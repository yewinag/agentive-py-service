"""End-to-end proof of the tool boundary working together, with no
external infrastructure:

    FakeBusinessServiceClient -> CheckVehicleAvailabilityTool -> ToolRegistry

No HTTP, no NestJS dependency, no database.
"""
import asyncio
from datetime import datetime, timedelta

from app.tools.business_client import VehicleAvailability
from app.tools.check_vehicle_availability import TOOL_NAME, CheckVehicleAvailabilityTool
from app.tools.fake_business_client import FakeBusinessServiceClient
from app.tools.registry import ToolRegistry

NOW = datetime(2026, 6, 1, 10, 0)


def test_registry_driven_tool_lookup_and_execution():
    vehicle = VehicleAvailability(
        vehicle_id="v1",
        category="SUV",
        model="Honda CR-V",
        available_from=NOW - timedelta(days=1),
        available_until=NOW + timedelta(days=14),
    )
    business_client = FakeBusinessServiceClient(vehicles=[vehicle])
    registry = ToolRegistry()
    registry.register(CheckVehicleAvailabilityTool(business_client))

    async def scenario():
        # An agent would discover the tool this way, without importing
        # CheckVehicleAvailabilityTool directly.
        metadata_list = registry.list_tools()
        assert [m.name for m in metadata_list] == [TOOL_NAME]

        tool = registry.resolve(TOOL_NAME)
        assert tool is not None

        return await tool.execute(
            {
                "pickup_at": NOW.isoformat(),
                "return_at": (NOW + timedelta(days=3)).isoformat(),
                "category": "SUV",
            }
        )

    result = asyncio.run(scenario())

    assert result.available_vehicles == [vehicle]


def test_registry_driven_lookup_of_unknown_tool_name_returns_none():
    registry = ToolRegistry()
    registry.register(CheckVehicleAvailabilityTool(FakeBusinessServiceClient()))

    assert registry.resolve("book_a_vehicle") is None
