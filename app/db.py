import sqlite3
import secrets
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

from app.config import Config

DB_PATH = str(Path(Config.DATA_DIR) / "bookings.db")


def _zips(*ranges: str | tuple[str, str]) -> list[str]:
    result: list[str] = []
    for r in ranges:
        if isinstance(r, str):
            result.append(r)
        else:
            s, e = r
            result.extend(str(z) for z in range(int(s), int(e) + 1))
    return result


SERVICE_AREAS: dict[str, dict[str, bool]] = {}
SERVICE_AREA_NOTES: dict[str, str] = {}

for z in _zips(("22030", "22039"), ("22041", "22044")):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": True}

for z in _zips(("22201", "22209"), "22213"):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": True}

for z in _zips(("22301", "22315")):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": False}
    SERVICE_AREA_NOTES[z] = "Electrical not available in Alexandria until Q2 2026"

for z in ("20147", "20148", "20164", "20165"):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": False, "electrical": False}
    SERVICE_AREA_NOTES[z] = "Only HVAC available in Loudoun (plumbing sub-contracted, electrical not available)"

for z in _zips(("20814", "20818"), "20832", "20833"):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": True}

for z in ("21042", "21043", "21044", "21045"):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": True}

for z in ("20706", "20707", "20708"):
    SERVICE_AREAS[z] = {"hvac": True, "plumbing": True, "electrical": False}
    SERVICE_AREA_NOTES[z] = "Electrical not licensed in Prince George's County — contact sister contractor EcoPower"

SERVICE_AREAS["20742"] = {"hvac": True, "plumbing": True, "electrical": True}
SERVICE_AREA_NOTES["20742"] = "University of Maryland campus requires facilities-office co-ordination — flag to scheduler before booking"


def check_service_area(zip_code: str, service_type: str) -> tuple[bool, Optional[str]]:
    entry = SERVICE_AREAS.get(zip_code)
    if entry is None:
        return False, f"ZIP {zip_code} is not in any Meridian service area"
    if not entry.get(service_type, False):
        note = SERVICE_AREA_NOTES.get(zip_code)
        return False, note or f"{service_type.title()} service not available for ZIP {zip_code}"
    note = SERVICE_AREA_NOTES.get(zip_code)
    if note:
        return True, note
    return True, None


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            booking_id TEXT UNIQUE NOT NULL,
            customer_id TEXT,
            customer_name TEXT,
            customer_phone TEXT,
            customer_email TEXT,
            service_type TEXT NOT NULL,
            job_type TEXT NOT NULL,
            zip_code TEXT NOT NULL,
            preferred_date TEXT NOT NULL,
            preferred_window TEXT NOT NULL,
            preferred_tech TEXT,
            status TEXT NOT NULL DEFAULT 'confirmed',
            appointment_window TEXT NOT NULL,
            confirmation_sent INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS waivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            waived_at TEXT NOT NULL
        );
    """)
    conn.close()


def _generate_id() -> str:
    return "BK-" + secrets.token_hex(4).upper()


def _appointment_window(date_str: str, window: str) -> str:
    if window == "morning":
        return f"{date_str} 08:00-10:00"
    return f"{date_str} 13:00-15:00"


def create_booking(
    customer_id: Optional[str],
    customer_name: Optional[str],
    customer_phone: Optional[str],
    customer_email: Optional[str],
    service_type: str,
    job_type: str,
    zip_code: str,
    preferred_date: str,
    preferred_window: str,
    preferred_tech: Optional[str],
) -> dict:
    booking_id = _generate_id()
    appointment = _appointment_window(preferred_date, preferred_window)
    now = datetime.now(UTC).isoformat()

    conn = get_conn()
    conn.execute(
        """INSERT INTO bookings
           (booking_id, customer_id, customer_name, customer_phone, customer_email,
            service_type, job_type, zip_code, preferred_date, preferred_window,
            preferred_tech, status, appointment_window, confirmation_sent, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed', ?, 1, ?, ?)""",
        (booking_id, customer_id, customer_name, customer_phone, customer_email,
         service_type, job_type, zip_code, preferred_date, preferred_window,
         preferred_tech, appointment, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "booking_id": booking_id,
        "status": "confirmed",
        "appointment_window": appointment,
        "confirmation_sent": True,
    }


def get_booking(booking_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def compute_reschedule(
    booking: dict, new_date: str, new_window: str
) -> tuple[int, bool, str]:
    preferred_date = booking["preferred_date"]
    preferred_window = booking["preferred_window"]

    hour = 8 if preferred_window == "morning" else 13
    apt_time = datetime.combine(
        date.fromisoformat(preferred_date), time(hour)
    )
    hours_notice = (apt_time - datetime.now()).total_seconds() / 3600

    if hours_notice > 24:
        fee = 0
    elif hours_notice >= 2:
        fee = 35
    else:
        fee = 75

    waiver_used = False
    if fee > 0 and booking.get("customer_id"):
        conn = get_conn()
        twelve_months_ago = (datetime.now() - timedelta(days=365)).isoformat()
        existing = conn.execute(
            "SELECT COUNT(*) FROM waivers WHERE customer_id = ? AND waived_at > ?",
            (booking["customer_id"], twelve_months_ago),
        ).fetchone()[0]
        if existing == 0:
            fee = 0
            waiver_used = True
            conn.execute(
                "INSERT INTO waivers (customer_id, waived_at) VALUES (?, ?)",
                (booking["customer_id"], datetime.now(UTC).isoformat()),
            )
        conn.commit()
        conn.close()

    new_appointment = _appointment_window(new_date, new_window)
    now = datetime.now(UTC).isoformat()

    conn = get_conn()
    conn.execute(
        """UPDATE bookings
           SET preferred_date = ?, preferred_window = ?, appointment_window = ?,
               status = 'confirmed', updated_at = ?
           WHERE booking_id = ?""",
        (new_date, new_window, new_appointment, now, booking["booking_id"]),
    )
    conn.commit()
    conn.close()

    return fee, waiver_used, new_appointment
