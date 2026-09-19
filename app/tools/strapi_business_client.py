from datetime import datetime
from typing import Optional

import httpx

from app.tools.availability_rules import car_is_available
from app.tools.business_client import VehicleAvailability
from app.tools.exceptions import BusinessServiceUnavailableError

# Fetching cars with their bookings populated and filtering in Python is
# the simplest approach that doesn't assume unverified Strapi filter
# query-string behavior (see Phase 3.1/3.2 notes) - populate is a
# well-documented, config-independent Strapi v5 REST feature, so relying
# on it alone (rather than also using filters[...] query params this
# specific instance's permissions were never exercised against) keeps
# this client's Strapi-side assumptions minimal.
_CARS_PATH = "/api/cars"


class StrapiBusinessServiceClient:
    """BusinessServiceClient implementation backed by the real Strapi API
    (see Phase 3.1's audit: Strapi + PostgreSQL owns Car/Booking/Payment
    data). Strapi-specific HTTP/auth/response-shape details live only in
    this module - CheckVehicleAvailabilityTool and the BusinessServiceClient
    Protocol know nothing about Strapi.

    Holds no availability business logic itself beyond mapping Strapi's
    response shape into the rules in app/tools/availability_rules.py.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=10.0,
        )

    async def check_vehicle_availability(
        self, start: datetime, end: datetime, category: Optional[str] = None
    ) -> list[VehicleAvailability]:
        try:
            response = await self._client.get(_CARS_PATH, params={"populate": "bookings"})
        except httpx.HTTPError as exc:
            raise BusinessServiceUnavailableError(f"Strapi request failed: {exc}") from exc

        if response.status_code != httpx.codes.OK:
            raise BusinessServiceUnavailableError(
                f"Strapi returned {response.status_code} for GET {_CARS_PATH}"
            )

        cars = response.json().get("data", [])
        requested_start = start.date()
        requested_end = end.date()

        results = []
        for car in cars:
            if category is not None and car["type"].lower() != category.lower():
                continue

            if not car_is_available(
                car_availability=car["availability"],
                bookings=car.get("bookings") or [],
                requested_start=requested_start,
                requested_end=requested_end,
            ):
                continue

            results.append(
                VehicleAvailability(
                    # documentId is Strapi v5's stable, API-facing
                    # identifier (see admin/src/index.ts seed script,
                    # which relates bookings to cars by documentId too) -
                    # preferred over the internal numeric id.
                    vehicle_id=car["documentId"],
                    category=car["type"],
                    model=car["model"],
                    available_from=start,
                    available_until=end,
                )
            )

        return results
