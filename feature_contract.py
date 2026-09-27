"""Authoritative ordered Spark feature contract reproduced from upstream filters."""

SELECTED_FEATURES = [
    "DISTANCE", "_flight_timestamp_epoch", "scheduled_hour",
    "airline_distinct_tails_late_lag1d", "airline_late_flight_ratio_lag1d",
    "airline_delay_per_tail_lag1d", "airline_total_delay_minutes_lag3d",
    "airline_delay_per_tail_lag3d", "airline_flights_count_lag7d",
    "airline_late_flight_ratio_lag7d", "airport_scheduled_departures_same_day",
    "airport_scheduled_arrivals_same_day", "airport_avg_delay_lag3h",
    "airport_late_flights_count_lag6h", "airplane_flights_scheduled_same_day",
    "airplane_total_delay_minutes_lag1d", "airplane_total_delay_minutes_lag3d",
    "airplane_delay_per_km_lag7d", "airplane_prev_flight_arrival_delay",
    "flight_computed_scheduled_duration_minutes", "flight_scheduled_minutes_per_km",
    "is_weekend", "time_of_day_bucket_evening",
]

# Both fitted artifacts currently use the same verified ordered contract.
# Keeping task keys explicit prevents inference from assuming this forever.
FEATURES_BY_TASK = {
    "classification": SELECTED_FEATURES,
    "regression": SELECTED_FEATURES,
}
