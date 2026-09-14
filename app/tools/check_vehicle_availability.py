from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.tools.business_client import BusinessServiceClient, VehicleAvailability
from app.tools.exceptions import ToolInputError
from app.tools.tool import ToolMetadata

TOOL_NAME = "check_vehicle_availability"

TOOL_DESCRIPTION = (
    "Checks which rental vehicles are available for a given pickup/return "
    "date-time range, optionally filtered by vehicle category. Use this for "
    "live availability questions about specific dates - not for general "
    "policy, pricing, or requirement questions, which are answered from the "
    "knowledge base instead."
)


class CheckVehicleAvailabilityInput(BaseModel):
    """No category enum on purpose: vehicle categories are business
    inventory knowledge owned by the business backend, not something
    this service should hardcode a taxonomy for.
    """

    pickup_at: datetime
    return_at: datetime
    category: Optional[str] = None

    @field_validator("return_at")
    @classmethod
    def return_at_must_be_after_pickup_at(cls, value: datetime, info) -> datetime:
        pickup_at = info.data.get("pickup_at")
        if pickup_at is not None and value <= pickup_at:
            raise ValueError("return_at must be after pickup_at")
        return value


class CheckVehicleAvailabilityResult(BaseModel):
    available_vehicles: list[VehicleAvailability] = Field(default_factory=list)


class CheckVehicleAvailabilityTool:
    """Tool implementation: validates raw input, delegates to
    BusinessServiceClient, wraps the result. Holds no business logic of
    its own (no pricing, no booking state, no inventory) - it is an
    integration boundary, not a business-domain owner.
    """

    def __init__(self, business_client: BusinessServiceClient) -> None:
        self._business_client = business_client

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            input_schema=CheckVehicleAvailabilityInput.model_json_schema(),
        )

    async def execute(self, raw_input: dict) -> CheckVehicleAvailabilityResult:
        try:
            parsed_input = CheckVehicleAvailabilityInput.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolInputError(str(exc)) from exc

        # BusinessServiceUnavailableError propagates unwrapped - it's
        # already translated at the BusinessServiceClient boundary, the
        # same "don't re-wrap an already-translated error" pattern
        # Retriever uses for EmbeddingProviderError/VectorStoreError.
        vehicles = await self._business_client.check_vehicle_availability(
            start=parsed_input.pickup_at,
            end=parsed_input.return_at,
            category=parsed_input.category,
        )

        return CheckVehicleAvailabilityResult(available_vehicles=vehicles)
