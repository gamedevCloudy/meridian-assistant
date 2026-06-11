"""
Agent Tools

Real tools bound to the LLM via langchain `@tool`. Each tool returns a
JSON-serialisable dict string so the model can reason over structured data.
"""

import json
import logging
import os
from typing import Optional

import httpx
from langchain.tools import tool

from app.data_loader.retriever import retrieve as _retrieve

logger = logging.getLogger(__name__)

API_BASE = os.getenv("MERIDIAN_API_BASE", "http://127.0.0.1:8000/api/v1")


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=10) as client:
        r = client.post(f"{API_BASE}{path}", json=payload)
        if r.status_code >= 400:
            return {"error": True, "status_code": r.status_code, "body": r.json()}
        return r.json()


def _patch(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=10) as client:
        r = client.patch(f"{API_BASE}{path}", json=payload)
        if r.status_code >= 400:
            return {"error": True, "status_code": r.status_code, "body": r.json()}
        return r.json()


@tool
def retrieve_kb(query: str, k: int = 4) -> str:
    """Search the Meridian knowledge base for FAQs, pricing, service-area
    rules, warranty terms, cancellation policy, and branch hours.

    Args:
        query: natural-language question to look up.
        k: number of chunks to return (default 4, max 8).

    Returns a JSON string with a list of {text, source, page} chunks.
    Always cite these sources in your final answer.
    """
    docs = _retrieve(query, k=min(k, 8))
    chunks = [
        {
            "text": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "doc_name": d.metadata.get("doc_name", "unknown"),
            "page": d.metadata.get("page", None),
        }
        for d in docs
    ]
    return json.dumps({"chunks": chunks})


@tool
def check_service_area(zip_code: str, service_type: str) -> str:
    """Check whether Meridian serves a given ZIP for a given service.

    Args:
        zip_code: 5-digit US ZIP.
        service_type: one of "hvac", "plumbing", "electrical".

    Returns a JSON string {"eligible": bool, "note": Optional[str]}.
    """
    params = {"zip_code": zip_code, "service_type": service_type}
    try:
        r = httpx.get(f"{API_BASE}/bookings/check", params=params, timeout=10)
        if r.status_code == 404:
            return json.dumps({"eligible": False, "note": "Endpoint not available, treating as not eligible"})
        return r.text
    except Exception as e:
        logger.warning("check_service_area failed: %s", e)
        return json.dumps({"eligible": False, "note": f"Lookup error: {e}"})


@tool
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
    preferred_tech: Optional[str] = None,
) -> str:
    """Create a new service booking via the Meridian mock API.

    Args:
        customer_id: existing customer id (optional if customer_name given).
        customer_name: name for new customers (optional if customer_id given).
        customer_phone: contact phone (recommended).
        customer_email: contact email (recommended).
        service_type: "hvac" | "plumbing" | "electrical".
        job_type: short description e.g. "diagnostic_visit", "warranty_return",
            "ac_repair", "drain_unclog", "panel_upgrade".
        zip_code: 5-digit ZIP; must be in a Meridian service area for the
            requested service_type (call check_service_area first).
        preferred_date: ISO date "YYYY-MM-DD", today or later, within 60 days.
        preferred_window: "morning" | "afternoon".
        preferred_tech: optional tech name.

    Returns a JSON string with booking_id / appointment_window or an error.
    Always ask the user to confirm details before calling this tool.
    """
    payload = {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_phone": customer_phone,
        "customer_email": customer_email,
        "service_type": service_type,
        "job_type": job_type,
        "zip_code": zip_code,
        "preferred_date": preferred_date,
        "preferred_window": preferred_window,
        "preferred_tech": preferred_tech,
    }
    return json.dumps(_post("/bookings", payload))


@tool
def reschedule_booking(booking_id: str, new_date: str, new_window: str) -> str:
    """Reschedule an existing booking to a new date and time window.

    Args:
        booking_id: the BK-XXXXXXXX id returned at booking time.
        new_date: ISO date "YYYY-MM-DD", today or later.
        new_window: "morning" | "afternoon".

    Returns a JSON string with new_appointment_window, fee_applied (0/35/75),
    and waiver_used (bool). Always warn the user about any fee before calling.
    """
    payload = {"new_date": new_date, "new_window": new_window}
    return json.dumps(_patch(f"/bookings/{booking_id}", payload))


@tool
def handoff_to_human(reason: str, context: str) -> str:
    """Escalate the conversation to a human contact-centre agent.

    Call this when:
    - the customer reports an emergency (gas leak, flooding, sparking, no heat in winter);
    - the request is out of scope (commercial accounts, billing disputes, legal);
    - the customer is angry or asks for a manager;
    - you have asked twice for required info and the customer has not provided it;
    - the request is ambiguous and you cannot reasonably clarify.

    Args:
        reason: short label, e.g. "emergency_gas_leak", "commercial_request",
            "missing_info", "low_confidence", "out_of_scope".
        context: a one-paragraph summary of the conversation so a human can
            pick up without re-asking.

    Returns a JSON string confirming the handoff. Do not call any other tool
    after this; the conversation ends here from your side.
    """
    logger.info("HANDOFF requested: %s | %s", reason, context)
    return json.dumps({"handoff": True, "reason": reason, "context": context})


tools = [retrieve_kb, check_service_area, create_booking, reschedule_booking, handoff_to_human]
tools_by_name = {t.name: t for t in tools}
