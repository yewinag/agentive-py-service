import pytest

from app.tools.tool import ToolMetadata
from app.tools.registry import ToolRegistry


class StubTool:
    """Satisfies Tool structurally, unrelated to CheckVehicleAvailabilityTool -
    proves the registry depends on the Protocol, not a concrete tool.
    """

    def __init__(self, name: str = "stub_tool") -> None:
        self._name = name

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description="A stub tool.", input_schema={})

    async def execute(self, raw_input: dict):
        return {"echo": raw_input}


def test_register_and_resolve_by_name():
    registry = ToolRegistry()
    tool = StubTool()

    registry.register(tool)

    assert registry.resolve("stub_tool") is tool


def test_resolve_returns_none_for_unknown_tool():
    registry = ToolRegistry()

    assert registry.resolve("does_not_exist") is None


def test_list_tools_returns_metadata_for_every_registered_tool():
    registry = ToolRegistry()
    registry.register(StubTool("tool_a"))
    registry.register(StubTool("tool_b"))

    names = {metadata.name for metadata in registry.list_tools()}

    assert names == {"tool_a", "tool_b"}


def test_list_tools_returns_metadata_not_tool_instances():
    registry = ToolRegistry()
    registry.register(StubTool("tool_a"))

    [metadata] = registry.list_tools()

    assert isinstance(metadata, ToolMetadata)


def test_registering_a_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register(StubTool("tool_a"))

    with pytest.raises(ValueError):
        registry.register(StubTool("tool_a"))
