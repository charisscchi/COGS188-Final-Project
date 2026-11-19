import argparse
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from road_network import positive_number, validate_id
from routing import checked_cost
from traffic_aware import NYC, TrafficObservation, TrafficSnapshot, normalize_time, valid_speed


DATA_DIR = Path(__file__).resolve().parent.parent / "Data Cleanup"
TRAFFIC_FILE = DATA_DIR / "DOT_Traffic_Speeds_20261116.csv"
TRIPS_FILE = DATA_DIR / "2024_High_Volume_FHV_Trip_Data_20261116.csv"
DEFAULT_LOOKBACK_MINUTES = 15
DOT_TIME_FORMAT = "%Y %b %d %I:%M:%S %p"
FHV_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def request_timestamp(value: datetime | str) -> datetime:
    """Accept datetimes, ISO timestamps, and the bundled FHV request format."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            try:
                value = datetime.strptime(value, FHV_TIME_FORMAT)
            except ValueError as exc:
                raise ValueError("Invalid request timestamp.") from exc
    return normalize_time(value)


def load_traffic_data(path: str | Path | TextIO = TRAFFIC_FILE) -> pd.DataFrame:
    """Read raw DOT speeds and quality fields, normalizing timestamps to UTC.

    Keep the last CSV row for identical link and timestamp keys. Keep a bad
    speed in the table so a fresh invalid row cannot revive an older reading.
    """
    traffic = pd.read_csv(path, dtype={"LINK_ID": str})
    required = {"LINK_ID", "SPEED", "DATA_AS_OF"}
    if missing := required - set(traffic.columns):
        raise ValueError(f"Traffic CSV is missing columns: {sorted(missing)!r}.")

    raw_times = traffic["DATA_AS_OF"]
    local_times = pd.to_datetime(raw_times, format=DOT_TIME_FORMAT, errors="coerce")
    times = local_times.dt.tz_localize(NYC, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

    def parse_iso(value: object) -> datetime | None:
        if isinstance(value, str):
            try:
                return normalize_time(datetime.fromisoformat(value))
            except ValueError:
                pass
        return None

    other = local_times.isna()
    times.loc[other] = pd.to_datetime(raw_times.loc[other].map(parse_iso), utc=True)
    traffic["DATA_AS_OF"] = times
    traffic["LINK_ID"] = traffic["LINK_ID"].str.strip()
    traffic["SPEED"] = pd.to_numeric(traffic["SPEED"], errors="coerce")
    if "TRAVEL_TIME" in traffic:
        traffic["TRAVEL_TIME"] = pd.to_numeric(traffic["TRAVEL_TIME"], errors="coerce")
    traffic = traffic.dropna(subset=["LINK_ID", "DATA_AS_OF"])
    traffic = traffic[traffic["LINK_ID"] != ""]
    return (
        traffic.drop_duplicates(subset=["LINK_ID", "DATA_AS_OF"], keep="last")
        .sort_values("DATA_AS_OF", kind="stable")
        .reset_index(drop=True)
    )


def get_traffic_snapshot(
    traffic: pd.DataFrame,
    request_time: datetime | str,
    max_age_minutes: float = DEFAULT_LOOKBACK_MINUTES,
) -> pd.DataFrame:
    """Return the latest row per DOT link within the inclusive lookback."""
    requested = request_timestamp(request_time)
    age = checked_cost(max_age_minutes, "Traffic lookback in minutes")
    try:
        earliest = pd.Timestamp(requested) - pd.Timedelta(minutes=age)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Traffic lookback is outside the supported time range.") from exc
    times = traffic["DATA_AS_OF"]
    window = traffic.loc[(times >= earliest) & (times <= requested)]
    return (
        window.sort_values("DATA_AS_OF", kind="stable")
        .drop_duplicates(subset="LINK_ID", keep="last")
        .copy()
    )


def historical_snapshot(
    traffic: pd.DataFrame,
    request_time: datetime | str,
    road_to_link: Mapping[str, str],
    *,
    meters_per_second_per_unit: float,
    max_age_minutes: float = DEFAULT_LOOKBACK_MINUTES,
    accept_observation: Callable[[pd.Series], bool] | None = None,
) -> TrafficSnapshot:
    """Adapt normalized DOT rows using a caller-verified crosswalk and unit scale.

    Each directed road maps to one DOT link. One link can cover several roads.
    The optional quality rule receives a raw row including STATUS when present.
    No DOT status code is interpreted as a closure.
    """
    scale = positive_number(meters_per_second_per_unit, "Source speed conversion factor")
    requested = request_timestamp(request_time)
    rows = get_traffic_snapshot(traffic, requested, max_age_minutes).set_index("LINK_ID")
    observations = {}
    for road_id, link_id in road_to_link.items():
        validate_id(road_id)
        validate_id(link_id)
        if link_id not in rows.index:
            continue
        row = rows.loc[link_id]
        measured_at = row["DATA_AS_OF"]
        raw_speed = row["SPEED"]
        speed = float(raw_speed) * scale if valid_speed(raw_speed) else None
        if valid_speed(speed) and (accept_observation is None or accept_observation(row)):
            observations[road_id] = TrafficObservation(speed, measured_at)
        else:
            observations[road_id] = TrafficObservation(None, measured_at, "missing")
    return TrafficSnapshot(requested, observations, "historical DOT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check raw historical traffic coverage.")
    parser.add_argument("--traffic", type=Path, default=TRAFFIC_FILE)
    parser.add_argument("--trips", type=Path, default=TRIPS_FILE)
    parser.add_argument("--lookback-minutes", type=float, default=DEFAULT_LOOKBACK_MINUTES)
    parser.add_argument("--at", action="append", default=[], help="Show coverage at an ISO timestamp.")
    args = parser.parse_args()
    lookback = checked_cost(args.lookback_minutes, "Traffic lookback in minutes")

    traffic = load_traffic_data(args.traffic)
    raw_positive = traffic.loc[traffic["SPEED"].map(valid_speed)]
    requests = pd.read_csv(args.trips, usecols=["request_datetime"])["request_datetime"]
    covered = invalid_requests = 0
    for request in requests:
        try:
            when = request_timestamp(request)
        except ValueError:
            invalid_requests += 1
            continue
        rows = get_traffic_snapshot(traffic, when, lookback)
        covered += bool(rows["SPEED"].map(valid_speed).any())

    print("Normalized traffic rows:", len(traffic))
    print("Rows with positive finite raw speeds:", len(raw_positive))
    print("First positive-speed observation (UTC):", raw_positive["DATA_AS_OF"].min())
    print("Last positive-speed observation (UTC):", raw_positive["DATA_AS_OF"].max())
    print(f"Requests with any recent positive raw speed: {covered}/{len(requests)}")
    print("Invalid request timestamps:", invalid_requests)
    if covered < len(requests) - invalid_requests:
        print("Historical coverage is missing for some requests.")
    for when in args.at:
        rows = get_traffic_snapshot(traffic, when, lookback)
        print(f"Links with recent positive raw speeds at {when}: {rows['SPEED'].map(valid_speed).sum()}")
    print("This check precedes unit, quality, and directed road mapping verification.")
