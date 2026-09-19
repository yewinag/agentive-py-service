import pytest

from app.core.config import Settings
from app.tools.business_client import get_business_service_client
from app.tools.check_vehicle_availability import TOOL_NAME
from app.tools.fake_business_client import FakeBusinessServiceClient
from app.tools.registry import ToolRegistry, get_tool_registry


def test_get_business_service_client_returns_fake_by_default():
    settings = Settings(business_service_provider="fake")

    client = get_business_service_client(settings)

    assert isinstance(client, FakeBusinessServiceClient)


def test_get_business_service_client_rejects_unknown_provider():
    settings = Settings(business_service_provider="not-a-real-provider")

    with pytest.raises(NotImplementedError):
        get_business_service_client(settings)


def test_get_business_service_client_requires_a_token_for_strapi():
    settings = Settings(business_service_provider="strapi", strapi_api_token=None)

    with pytest.raises(RuntimeError):
        get_business_service_client(settings)


def test_get_business_service_client_returns_strapi_client_when_configured():
    from app.tools.strapi_business_client import StrapiBusinessServiceClient

    settings = Settings(
        business_service_provider="strapi",
        strapi_url="http://localhost:1337",
        strapi_api_token="a-token",
    )

    client = get_business_service_client(settings)

    assert isinstance(client, StrapiBusinessServiceClient)


def test_get_tool_registry_returns_a_registry_with_the_availability_tool():
    settings = Settings()

    registry = get_tool_registry(settings)

    assert isinstance(registry, ToolRegistry)
    assert registry.resolve(TOOL_NAME) is not None
    assert TOOL_NAME in {metadata.name for metadata in registry.list_tools()}
