import asyncio
from datetime import datetime, timedelta

import pytest

from app.tools.business_client import VehicleAvailability
from app.tools.check_vehicle_availability import (
    TOOL_NAME,
    CheckVehicleAvailabilityInput,
    CheckVehicleAvailabilityTool,
)
from app.tools.exceptions import BusinessServiceUnavailableError, ToolInputError
from app.tools.fake_business_client import FakeBusinessServiceClient

NOW = datetime(2026, 6, 1, 10, 0)


def test_metadata_exposes_name_description_and_input_schema():
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient())

    metadata = tool.metadata

    assert metadata.name == TOOL_NAME
    assert "availability" in metadata.description.lower()
    assert metadata.input_schema["type"] == "object"
    assert set(metadata.input_schema["required"]) == {"pickup_at", "return_at"}


def test_input_model_rejects_return_at_before_pickup_at():
    with pytest.raises(Exception):
        CheckVehicleAvailabilityInput(
            pickup_at=NOW, return_at=NOW - timedelta(hours=1)
        )


def test_input_model_rejects_return_at_equal_to_pickup_at():
    with pytest.raises(Exception):
        CheckVehicleAvailabilityInput(pickup_at=NOW, return_at=NOW)


def test_input_model_accepts_a_valid_range_without_category():
    parsed = CheckVehicleAvailabilityInput(pickup_at=NOW, return_at=NOW + timedelta(days=1))

    assert parsed.category is None


def test_execute_returns_available_vehicles_for_a_valid_request():
    vehicle = VehicleAvailability(
        vehicle_id="v1",
        category="Economy",
        model="Toyota Yaris",
        available_from=NOW - timedelta(days=1),
        available_until=NOW + timedelta(days=10),
    )
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient(vehicles=[vehicle]))

    result = asyncio.run(
        tool.execute({"pickup_at": NOW.isoformat(), "return_at": (NOW + timedelta(days=2)).isoformat()})
    )

    assert result.available_vehicles == [vehicle]


def test_execute_raises_tool_input_error_for_missing_required_fields():
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient())

    with pytest.raises(ToolInputError):
        asyncio.run(tool.execute({"pickup_at": NOW.isoformat()}))


def test_execute_raises_tool_input_error_for_invalid_date_ordering():
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient())

    with pytest.raises(ToolInputError):
        asyncio.run(
            tool.execute(
                {
                    "pickup_at": NOW.isoformat(),
                    "return_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            )
        )


def test_execute_raises_tool_input_error_for_malformed_datetime():
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient())

    with pytest.raises(ToolInputError):
        asyncio.run(tool.execute({"pickup_at": "not-a-date", "return_at": "also-not-a-date"}))


def test_execute_delegates_to_business_client_with_parsed_arguments():
    class RecordingClient(FakeBusinessServiceClient):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def check_vehicle_availability(self, start, end, category=None):
            self.calls.append((start, end, category))
            return await super().check_vehicle_availability(start, end, category)

    client = RecordingClient()
    tool = CheckVehicleAvailabilityTool(client)
    return_at = NOW + timedelta(days=3)

    asyncio.run(
        tool.execute(
            {"pickup_at": NOW.isoformat(), "return_at": return_at.isoformat(), "category": "SUV"}
        )
    )

    assert client.calls == [(NOW, return_at, "SUV")]


def test_execute_propagates_business_service_unavailable_error_unwrapped():
    tool = CheckVehicleAvailabilityTool(
        FakeBusinessServiceClient(error=BusinessServiceUnavailableError("business API down"))
    )

    with pytest.raises(BusinessServiceUnavailableError):
        asyncio.run(
            tool.execute(
                {"pickup_at": NOW.isoformat(), "return_at": (NOW + timedelta(days=1)).isoformat()}
            )
        )


def test_execute_returns_empty_result_when_nothing_is_available():
    tool = CheckVehicleAvailabilityTool(FakeBusinessServiceClient(vehicles=[]))

    result = asyncio.run(
        tool.execute(
            {"pickup_at": NOW.isoformat(), "return_at": (NOW + timedelta(days=1)).isoformat()}
        )
    )

    assert result.available_vehicles == []
