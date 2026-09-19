"""Unit tests for StrapiBusinessServiceClient against a mocked HTTP
transport (httpx.MockTransport - part of httpx itself, no real network
and no extra dependency). Mirrors the actual Strapi v5 response shape
(flat fields under "data", no v4-style "attributes" wrapper) confirmed
in Phase 3.1's audit against the real running instance.
"""
import asyncio
from datetime import datetime

import httpx
import pytest

from app.tools.business_client import VehicleAvailability
from app.tools.exceptions import BusinessServiceUnavailableError
from app.tools.strapi_business_client import StrapiBusinessServiceClient

CAMRY = {
    "id": 1,
    "documentId": "camry-doc-id",
    "brand": "Toyota",
    "model": "Camry",
    "type": "Sedan",
    "availability": "AVAILABLE",
    "bookings": [
        {"bookingStatus": "PENDING", "pickupDate": "2026-09-20", "returnDate": "2026-09-23"}
    ],
}

CIVIC = {
    "id": 2,
    "documentId": "civic-doc-id",
    "brand": "Honda",
    "model": "Civic",
    "type": "Sedan",
    "availability": "AVAILABLE",
    "bookings": [
        {"bookingStatus": "APPROVED", "pickupDate": "2026-09-15", "returnDate": "2026-09-18"}
    ],
}

FORTUNER = {
    "id": 3,
    "documentId": "fortuner-doc-id",
    "brand": "Toyota",
    "model": "Fortuner",
    "type": "SUV",
    "availability": "AVAILABLE",
    "bookings": [
        {"bookingStatus": "COMPLETED", "pickupDate": "2026-08-01", "returnDate": "2026-08-05"}
    ],
}

HONDA_CITY = {
    "id": 4,
    "documentId": "city-doc-id",
    "brand": "Honda",
    "model": "City",
    "type": "Sedan",
    "availability": "UNAVAILABLE",
    "bookings": [],
}

ALL_CARS = [CAMRY, CIVIC, FORTUNER, HONDA_CITY]


def _client_with(handler) -> StrapiBusinessServiceClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(
        base_url="http://localhost:1337",
        headers={"Authorization": "Bearer test-token"},
        transport=transport,
    )
    return StrapiBusinessServiceClient(base_url="http://localhost:1337", api_token="test-token", client=http_client)


def _ok_handler(cars: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/cars"
        assert request.url.params["populate"] == "bookings"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"data": cars, "meta": {}})

    return handler


def test_case_1_civic_after_its_booking_is_returned_as_available():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22))
    )

    assert VehicleAvailability(
        vehicle_id="civic-doc-id",
        category="Sedan",
        model="Civic",
        available_from=datetime(2026, 9, 20),
        available_until=datetime(2026, 9, 22),
    ) in results


def test_case_2_civic_during_its_approved_booking_is_excluded():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 16), datetime(2026, 9, 17))
    )

    assert "civic-doc-id" not in {v.vehicle_id for v in results}


def test_case_3_camry_during_its_pending_booking_is_excluded():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22))
    )

    assert "camry-doc-id" not in {v.vehicle_id for v in results}


def test_case_4_camry_after_its_pending_booking_is_included():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 24), datetime(2026, 9, 26))
    )

    assert "camry-doc-id" in {v.vehicle_id for v in results}


def test_case_5_fortuner_completed_booking_does_not_block_future_dates():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22))
    )

    assert "fortuner-doc-id" in {v.vehicle_id for v in results}


def test_case_6_honda_city_is_excluded_regardless_of_requested_dates():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 1, 1), datetime(2026, 1, 2))
    )

    assert "city-doc-id" not in {v.vehicle_id for v in results}


def test_category_filter_maps_to_car_type_case_insensitively():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 24), datetime(2026, 9, 26), category="suv")
    )

    assert {v.vehicle_id for v in results} == {"fortuner-doc-id"}


def test_unknown_category_returns_empty_list_not_an_error():
    client = _client_with(_ok_handler(ALL_CARS))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 24), datetime(2026, 9, 26), category="Van")
    )

    assert results == []


def test_no_cars_at_all_returns_empty_list():
    client = _client_with(_ok_handler([]))

    results = asyncio.run(
        client.check_vehicle_availability(datetime(2026, 9, 24), datetime(2026, 9, 26))
    )

    assert results == []


def test_403_from_strapi_raises_business_service_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"data": None, "error": {"status": 403, "name": "ForbiddenError"}})

    client = _client_with(handler)

    with pytest.raises(BusinessServiceUnavailableError):
        asyncio.run(client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22)))


def test_401_from_strapi_raises_business_service_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"data": None, "error": {"status": 401, "name": "UnauthorizedError"}})

    client = _client_with(handler)

    with pytest.raises(BusinessServiceUnavailableError):
        asyncio.run(client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22)))


def test_network_failure_raises_business_service_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with(handler)

    with pytest.raises(BusinessServiceUnavailableError):
        asyncio.run(client.check_vehicle_availability(datetime(2026, 9, 20), datetime(2026, 9, 22)))
