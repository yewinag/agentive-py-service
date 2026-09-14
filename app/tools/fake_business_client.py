from datetime import datetime
from typing import Optional

from app.tools.business_client import VehicleAvailability


class FakeBusinessServiceClient:
    """Deterministic test double for BusinessServiceClient - not a second
    business database. Holds only whatever fixed vehicle list a test
    passes in (default: none); nothing is ever written to it at
    runtime, so it duplicates no real business/domain logic (no
    pricing, no booking state, no customer records). A real
    implementation will call the NestJS Business API over HTTP - not
    built in this step (see README's Tools section for why).

    Satisfies BusinessServiceClient structurally - it never needs to
    inherit from it.
    """

    def __init__(
        self,
        vehicles: Optional[list[VehicleAvailability]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._vehicles = vehicles if vehicles is not None else []
        self._error = error

    async def check_vehicle_availability(
        self, start: datetime, end: datetime, category: Optional[str] = None
    ) -> list[VehicleAvailability]:
        if self._error:
            raise self._error

        return [
            vehicle
            for vehicle in self._vehicles
            if vehicle.available_from <= start
            and end <= vehicle.available_until
            and (category is None or vehicle.category.lower() == category.lower())
        ]
