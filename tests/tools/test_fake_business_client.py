import asyncio
from datetime import datetime, timedelta

import pytest

from app.tools.business_client import VehicleAvailability
from app.tools.fake_business_client import FakeBusinessServiceClient

NOW = datetime(2026, 6, 1, 10, 0)


def _vehicle(vehicle_id: str, category: str, from_offset_days: int, until_offset_days: int) -> VehicleAvailability:
    return VehicleAvailability(
        vehicle_id=vehicle_id,
        category=category,
        model="Test Model",
        available_from=NOW + timedelta(days=from_offset_days),
        available_until=NOW + timedelta(days=until_offset_days),
    )


def test_returns_empty_list_by_default():
    client = FakeBusinessServiceClient()

    results = asyncio.run(
        client.check_vehicle_availability(NOW, NOW + timedelta(days=1))
    )

    assert results == []


def test_returns_vehicles_whose_window_covers_the_requested_range():
    vehicle = _vehicle("v1", "Economy", -1, 10)
    client = FakeBusinessServiceClient(vehicles=[vehicle])

    results = asyncio.run(
        client.check_vehicle_availability(NOW, NOW + timedelta(days=2))
    )

    assert results == [vehicle]


def test_excludes_vehicles_whose_window_does_not_cover_the_requested_range():
    vehicle = _vehicle("v1", "Economy", 5, 6)  # only available days 5-6
    client = FakeBusinessServiceClient(vehicles=[vehicle])

    results = asyncio.run(
        client.check_vehicle_availability(NOW, NOW + timedelta(days=2))
    )

    assert results == []


def test_filters_by_category_case_insensitively():
    economy = _vehicle("v1", "Economy", -1, 10)
    suv = _vehicle("v2", "SUV", -1, 10)
    client = FakeBusinessServiceClient(vehicles=[economy, suv])

    results = asyncio.run(
        client.check_vehicle_availability(NOW, NOW + timedelta(days=1), category="economy")
    )

    assert results == [economy]


def test_raises_the_configured_error():
    client = FakeBusinessServiceClient(error=RuntimeError("business service down"))

    with pytest.raises(RuntimeError):
        asyncio.run(client.check_vehicle_availability(NOW, NOW + timedelta(days=1)))
