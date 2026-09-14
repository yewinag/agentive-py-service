from datetime import datetime
from typing import Optional, Protocol

from pydantic import BaseModel

from app.core.config import Settings


class VehicleAvailability(BaseModel):
    """One vehicle the business system reports as available. Deliberately
    minimal - just enough for a tool result to be useful: no pricing, no
    booking state, no customer data. The business backend owns all of
    that; this is only what "is this vehicle free for this window" needs.
    """

    vehicle_id: str
    category: str
    model: str
    available_from: datetime
    available_until: datetime


class BusinessServiceClient(Protocol):
    """The boundary tools code against: WHAT they need from the business
    system. Concrete clients (fake today, an HTTP client calling the
    NestJS Business API later) implement this structurally - the same
    pattern as every other Protocol in this codebase.

    Only one operation exists today because only one tool exists;
    booking lookup/creation/modification would each add their own
    method here once a tool actually needs them, not before.
    """

    async def check_vehicle_availability(
        self, start: datetime, end: datetime, category: Optional[str] = None
    ) -> list[VehicleAvailability]: ...


def get_business_service_client(settings: Settings) -> BusinessServiceClient:
    """Composition point - a plain function, not FastAPI Depends()-wired,
    matching get_llm_provider()/get_vector_store()/etc. Only "fake" is
    implemented today; kept as a Settings-driven selector (same shape as
    every other provider setting) so a real HTTP client calling the
    NestJS Business API slots in later without touching any tool.
    """
    from app.tools.fake_business_client import FakeBusinessServiceClient

    if settings.business_service_provider == "fake":
        return FakeBusinessServiceClient()

    raise NotImplementedError(
        f"Business service provider '{settings.business_service_provider}' is not implemented yet"
    )
