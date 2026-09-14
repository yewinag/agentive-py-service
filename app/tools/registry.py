from typing import Optional

from app.core.config import Settings
from app.tools.tool import Tool, ToolMetadata


class ToolRegistry:
    """In-process registry of available tools. Deliberately not a plugin
    system: tools are registered explicitly (register()), never
    discovered via dynamic imports/runtime code loading, so the set of
    available tools is always the same deterministic list for a given
    process.
    """

    def __init__(self) -> None:
        self._tools: dict = {}

    def register(self, tool: Tool) -> None:
        name = tool.metadata.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = tool

    def resolve(self, name: str) -> Optional[Tool]:
        """Returns the tool, or None if name is unknown - a normal,
        expected outcome for a lookup (matching ConversationStore.get()'s
        precedent), not an error. What to do about "unknown tool" is a
        caller (eventually an agent) policy decision, not the
        registry's.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[ToolMetadata]:
        return [tool.metadata for tool in self._tools.values()]


def get_tool_registry(settings: Settings) -> ToolRegistry:
    """Composition point - a plain function, not FastAPI Depends()-wired:
    no endpoint consumes a ToolRegistry yet, matching every other
    composition function in this codebase. A fresh ToolRegistry per call
    is fine (unlike get_vector_store()/get_conversation_store()): tools
    and BusinessServiceClient hold no runtime-accumulated state here -
    nothing ever writes through FakeBusinessServiceClient - so there is
    nothing a later call could fail to see.
    """
    from app.tools.business_client import get_business_service_client
    from app.tools.check_vehicle_availability import CheckVehicleAvailabilityTool

    registry = ToolRegistry()
    registry.register(CheckVehicleAvailabilityTool(get_business_service_client(settings)))
    return registry
