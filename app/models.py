from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

ServiceType = Literal["hvac", "plumbing", "electrical"]
WindowType = Literal["morning", "afternoon"]


class BookingCreate(BaseModel):
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    service_type: ServiceType
    job_type: str
    zip_code: str
    preferred_date: date
    preferred_window: WindowType
    preferred_tech: Optional[str] = None

    @model_validator(mode="after")
    def validations(self) -> "BookingCreate":
        if not self.customer_id and not self.customer_name:
            raise ValueError("customer_id or customer_name required")
        if self.preferred_date < date.today():
            raise ValueError("preferred_date must be today or later")
        if self.preferred_date > date.today() + timedelta(days=60):
            raise ValueError("preferred_date must be within 60 days")
        return self


class BookingReschedule(BaseModel):
    new_date: date
    new_window: WindowType

    @model_validator(mode="after")
    def validations(self) -> "BookingReschedule":
        if self.new_date < date.today():
            raise ValueError("new_date must be today or later")
        return self


class BookingResponse(BaseModel):
    booking_id: str
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    service_type: ServiceType
    job_type: str
    zip_code: str
    preferred_date: str
    preferred_window: WindowType
    status: str
    appointment_window: str
    confirmation_sent: bool
    created_at: str


class BookingCreateResponse(BaseModel):
    booking_id: str
    status: str
    appointment_window: str
    confirmation_sent: bool
    message: str


class BookingRescheduleResponse(BaseModel):
    booking_id: str
    status: str
    new_appointment_window: str
    fee_applied: int
    waiver_used: bool
    message: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None
    handoff_suggested: bool = False
