"""Unit tests for the deterministic overlap/availability rules, using the
same scenarios as Phase 3.2's audit trace against the real Strapi seed
data (see admin/src/index.ts) - no HTTP, no Strapi, no fakes involved.
"""
from datetime import date

from app.tools.availability_rules import booking_blocks_range, car_is_available

HONDA_CIVIC_BOOKING = {
    "bookingStatus": "APPROVED",
    "pickupDate": "2026-09-15",
    "returnDate": "2026-09-18",
}

TOYOTA_CAMRY_BOOKING = {
    "bookingStatus": "PENDING",
    "pickupDate": "2026-09-20",
    "returnDate": "2026-09-23",
}

TOYOTA_FORTUNER_BOOKING = {
    "bookingStatus": "COMPLETED",
    "pickupDate": "2026-08-01",
    "returnDate": "2026-08-05",
}


def test_case_1_honda_civic_after_its_booking_is_available():
    assert car_is_available(
        car_availability="AVAILABLE",
        bookings=[HONDA_CIVIC_BOOKING],
        requested_start=date(2026, 9, 20),
        requested_end=date(2026, 9, 22),
    )


def test_case_2_honda_civic_during_its_approved_booking_is_not_available():
    assert not car_is_available(
        car_availability="AVAILABLE",
        bookings=[HONDA_CIVIC_BOOKING],
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_case_3_toyota_camry_during_its_pending_booking_is_not_available():
    assert not car_is_available(
        car_availability="AVAILABLE",
        bookings=[TOYOTA_CAMRY_BOOKING],
        requested_start=date(2026, 9, 20),
        requested_end=date(2026, 9, 22),
    )


def test_case_4_toyota_camry_after_its_pending_booking_is_available():
    assert car_is_available(
        car_availability="AVAILABLE",
        bookings=[TOYOTA_CAMRY_BOOKING],
        requested_start=date(2026, 9, 24),
        requested_end=date(2026, 9, 26),
    )


def test_case_5_toyota_fortuner_completed_booking_does_not_block_future_dates():
    assert car_is_available(
        car_availability="AVAILABLE",
        bookings=[TOYOTA_FORTUNER_BOOKING],
        requested_start=date(2026, 9, 20),
        requested_end=date(2026, 9, 22),
    )


def test_case_6_honda_city_is_never_available_regardless_of_bookings():
    assert not car_is_available(
        car_availability="UNAVAILABLE",
        bookings=[],
        requested_start=date(2026, 9, 20),
        requested_end=date(2026, 9, 22),
    )


def test_declined_booking_does_not_block():
    assert not booking_blocks_range(
        booking_status="DECLINED",
        pickup_date=date(2026, 9, 15),
        return_date=date(2026, 9, 18),
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_cancelled_booking_does_not_block():
    assert not booking_blocks_range(
        booking_status="CANCELLED",
        pickup_date=date(2026, 9, 15),
        return_date=date(2026, 9, 18),
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_completed_booking_does_not_block():
    assert not booking_blocks_range(
        booking_status="COMPLETED",
        pickup_date=date(2026, 9, 15),
        return_date=date(2026, 9, 18),
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_pending_booking_blocks_an_overlapping_range():
    assert booking_blocks_range(
        booking_status="PENDING",
        pickup_date=date(2026, 9, 15),
        return_date=date(2026, 9, 18),
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_approved_booking_blocks_an_overlapping_range():
    assert booking_blocks_range(
        booking_status="APPROVED",
        pickup_date=date(2026, 9, 15),
        return_date=date(2026, 9, 18),
        requested_start=date(2026, 9, 16),
        requested_end=date(2026, 9, 17),
    )


def test_adjacent_ranges_touching_at_a_single_day_count_as_overlapping():
    """pickupDate <= requested_end AND returnDate >= requested_start is an
    inclusive comparison - a same-day pickup/return boundary is treated
    as a conflict, not as free. This is the documented formula from
    Phase 3.2, not an inferred rule.
    """
    assert booking_blocks_range(
        booking_status="APPROVED",
        pickup_date=date(2026, 9, 18),
        return_date=date(2026, 9, 20),
        requested_start=date(2026, 9, 15),
        requested_end=date(2026, 9, 18),
    )


def test_no_bookings_and_available_flag_means_available():
    assert car_is_available(
        car_availability="AVAILABLE",
        bookings=[],
        requested_start=date(2026, 9, 20),
        requested_end=date(2026, 9, 22),
    )
