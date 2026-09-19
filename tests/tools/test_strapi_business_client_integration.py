"""Integration coverage against a real, running Strapi instance (admin/ -
see Phase 3.1's audit) and its seed data (admin/src/index.ts).

Skipped unless STRAPI_API_TOKEN is set - the rest of the suite (and CI by
default) never needs a running Strapi or a real token. As of Phase 3.2,
no such token has been created yet (see the phase report's manual-step
callout); this file exists so integration coverage is ready the moment
one is. To run this file locally, after creating a read-only Custom API
token in Strapi Admin (Settings > API Tokens) scoped to Car.find/findOne
and Booking.find/findOne:

    cd admin && npm run develop   # Strapi on :1337, seeds on first boot
    cd agentive-py-service
    STRAPI_URL=http://localhost:1337 STRAPI_API_TOKEN=<token> \
        python -m pytest tests/tools/test_strapi_business_client_integration.py -v
"""
import asyncio
import os
from datetime import datetime

import pytest

from app.tools.check_vehicle_availability import CheckVehicleAvailabilityTool
from app.tools.strapi_business_client import StrapiBusinessServiceClient

STRAPI_URL = os.environ.get("STRAPI_URL", "http://localhost:1337")
STRAPI_API_TOKEN = os.environ.get("STRAPI_API_TOKEN")

pytestmark = pytest.mark.skipif(
    not STRAPI_API_TOKEN,
    reason="STRAPI_API_TOKEN not set - skipping real Strapi integration test",
)


def _tool() -> CheckVehicleAvailabilityTool:
    client = StrapiBusinessServiceClient(base_url=STRAPI_URL, api_token=STRAPI_API_TOKEN)
    return CheckVehicleAvailabilityTool(client)


def test_honda_civic_is_available_after_its_seeded_approved_booking():
    tool = _tool()

    result = asyncio.run(
        tool.execute(
            {
                "pickup_at": datetime(2026, 9, 20).isoformat(),
                "return_at": datetime(2026, 9, 22).isoformat(),
                "category": "Sedan",
            }
        )
    )

    assert any(v.model == "Civic" for v in result.available_vehicles)


def test_honda_civic_is_not_available_during_its_seeded_approved_booking():
    tool = _tool()

    result = asyncio.run(
        tool.execute(
            {
                "pickup_at": datetime(2026, 9, 16).isoformat(),
                "return_at": datetime(2026, 9, 17).isoformat(),
            }
        )
    )

    assert not any(v.model == "Civic" for v in result.available_vehicles)


def test_honda_city_is_never_available_due_to_static_unavailable_flag():
    tool = _tool()

    result = asyncio.run(
        tool.execute(
            {
                "pickup_at": datetime(2026, 12, 1).isoformat(),
                "return_at": datetime(2026, 12, 2).isoformat(),
            }
        )
    )

    assert not any(v.model == "City" for v in result.available_vehicles)


def test_invalid_strapi_api_token_raises_business_service_unavailable_error():
    from app.tools.exceptions import BusinessServiceUnavailableError

    client = StrapiBusinessServiceClient(base_url=STRAPI_URL, api_token="definitely-not-a-real-token")
    tool = CheckVehicleAvailabilityTool(client)

    with pytest.raises(BusinessServiceUnavailableError):
        asyncio.run(
            tool.execute(
                {
                    "pickup_at": datetime(2026, 9, 20).isoformat(),
                    "return_at": datetime(2026, 9, 22).isoformat(),
                }
            )
        )
