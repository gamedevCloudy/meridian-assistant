import logging

from fastapi import APIRouter, HTTPException, Query

from app.db import check_service_area, create_booking, get_booking, compute_reschedule
from app.models import (
    BookingCreate,
    BookingCreateResponse,
    BookingReschedule,
    BookingRescheduleResponse,
    BookingResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bookings", tags=["bookings"])


@router.get("/check")
def check_zip(zip_code: str, service_type: str):
    valid, message = check_service_area(zip_code, service_type)
    return {"zip_code": zip_code, "service_type": service_type, "eligible": valid, "note": message}


@router.post("", response_model=BookingCreateResponse, status_code=201)
def create(body: BookingCreate):
    valid, message = check_service_area(body.zip_code, body.service_type)
    if not valid:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                status="out_of_area",
                error=message or f"Service not available for ZIP {body.zip_code}",
                handoff_suggested=True,
            ).model_dump(),
        )

    result = create_booking(
        customer_id=body.customer_id,
        customer_name=body.customer_name,
        customer_phone=body.customer_phone,
        customer_email=body.customer_email,
        service_type=body.service_type,
        job_type=body.job_type,
        zip_code=body.zip_code,
        preferred_date=body.preferred_date.isoformat(),
        preferred_window=body.preferred_window,
        preferred_tech=body.preferred_tech,
    )

    return BookingCreateResponse(
        booking_id=result["booking_id"],
        status=result["status"],
        appointment_window=result["appointment_window"],
        confirmation_sent=result["confirmation_sent"],
        message="Booking confirmed" + (f" — Note: {message}" if message else ""),
    )


@router.get("/{booking_id}", response_model=BookingResponse)
def get(booking_id: str, customer_id: str = Query(None)):
    booking = get_booking(booking_id)
    if booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")
    if customer_id and booking["customer_id"] != customer_id:
        raise HTTPException(status_code=403, detail="Customer ID does not match booking")
    return BookingResponse(**booking)


@router.patch("/{booking_id}", response_model=BookingRescheduleResponse)
def reschedule(booking_id: str, body: BookingReschedule):
    booking = get_booking(booking_id)
    if booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    fee, waiver_used, new_appointment = compute_reschedule(
        booking, body.new_date.isoformat(), body.new_window
    )

    return BookingRescheduleResponse(
        booking_id=booking_id,
        status="confirmed",
        new_appointment_window=new_appointment,
        fee_applied=fee,
        waiver_used=waiver_used,
        message="Rescheduled" + (f" (${fee} cancellation fee applied)" if fee > 0 else ""),
    )
