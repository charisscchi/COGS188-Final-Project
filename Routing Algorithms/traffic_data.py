
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data Cleanup"

TRAFFIC_FILE = DATA_DIR / "DOT_Traffic_Speeds_20261116.csv"


def load_traffic_data():

    traffic = pd.read_csv(
        TRAFFIC_FILE,
        dtype={"LINK_ID": str}
    )

    original_count = len(traffic)

    # Keep the information required for routing
    traffic = traffic[[
        "LINK_ID",
        "LINK_POINTS",
        "SPEED",
        "TRAVEL_TIME",
        "DATA_AS_OF",
        "BOROUGH"
    ]].copy()

    # Convert timestamps into datetime objects
    traffic["DATA_AS_OF"] = pd.to_datetime(
        traffic["DATA_AS_OF"],
        format="%Y %b %d %I:%M:%S %p",
        errors="coerce"
    )

    # Ensure speeds and travel times are numeric
    traffic["SPEED"] = pd.to_numeric(
        traffic["SPEED"], errors="coerce"
    )

    traffic["TRAVEL_TIME"] = pd.to_numeric(
        traffic["TRAVEL_TIME"], errors="coerce"
    )

    # Count zero-speed observations before filtering
    zero_count = (traffic["SPEED"] == 0).sum()

    # Remove missing essential information
    traffic = traffic.dropna(subset=[
        "LINK_ID",
        "LINK_POINTS",
        "SPEED",
        "TRAVEL_TIME",
        "DATA_AS_OF"
    ])

    # Keep positive, finite observations for this prototype
    traffic = traffic[
        np.isfinite(traffic["SPEED"])
        & np.isfinite(traffic["TRAVEL_TIME"])
        & (traffic["SPEED"] > 0)
        & (traffic["TRAVEL_TIME"] > 0)
    ].copy()

    traffic = traffic.drop_duplicates()

    traffic = traffic.sort_values(
        "DATA_AS_OF"
    ).reset_index(drop=True)

    print("Original records:", original_count)
    print("Zero-speed records:", zero_count)
    print("Usable records:", len(traffic))
    print("Removed records:", original_count - len(traffic))

    return traffic


def get_traffic_snapshot(
    traffic, request_time, max_age_minutes=15
):
    """
    Return the latest available observation for each DOT link.

    Only use observations available at or before the request.
    """

    request_time = pd.Timestamp(request_time)

    earliest_time = request_time - pd.Timedelta(
        minutes=max_age_minutes
    )

    snapshot = traffic[
        (traffic["DATA_AS_OF"] <= request_time)
        & (traffic["DATA_AS_OF"] >= earliest_time)
    ].copy()

    # Keep the latest observation for each road link
    snapshot = (
        snapshot.sort_values("DATA_AS_OF")
        .drop_duplicates(
            subset="LINK_ID",
            keep="last"
        )
    )

    return snapshot


if __name__ == "__main__":

    print("NYC Traffic Data Preparation")
    print("=" * 50)

    traffic = load_traffic_data()

    print("\nDataset Summary")
    print("-" * 50)

    print("Unique road links:", traffic["LINK_ID"].nunique())
    print("Earliest observation:", traffic["DATA_AS_OF"].min())
    print("Latest observation:", traffic["DATA_AS_OF"].max())

    print("\nUnique road links by borough:")
    print(
        traffic.groupby("BOROUGH")["LINK_ID"].nunique()
    )

    # Examine traffic conditions at different times
    test_times = [
        "2024-08-01 08:00:00",
        "2024-08-01 12:00:00",
        "2024-08-01 17:00:00",
        "2024-08-01 23:00:00"
    ]

    for request_time in test_times:

        snapshot = get_traffic_snapshot(
            traffic, request_time
        )

        print("\nTraffic snapshot:", request_time)
        print("Available road links:", len(snapshot))

        if not snapshot.empty:
            print(
                "Average recorded speed:",
                round(snapshot["SPEED"].mean(), 2)
            )

    print("\nTraffic data preparation completed!")
