from pathlib import Path
import pandas as pd

# Locate Krasi's data folder
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data Cleanup"

# Load the three datasets
traffic = pd.read_csv(
    DATA_DIR / "DOT_Traffic_Speeds_20261116.csv"
)

trips = pd.read_csv(
    DATA_DIR / "2024_High_Volume_FHV_Trip_Data_20261116.csv"
)

zones = pd.read_csv(
    DATA_DIR / "NYC_Taxi_Zones_20261115.csv"
)

# Inspect each dataset
datasets = {
    "Traffic": traffic,
    "FHV Trips": trips,
    "Taxi Zones": zones
}

for name, df in datasets.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())

# Check traffic information
print("\n--- Traffic Summary ---")
print("Unique road links:", traffic["LINK_ID"].nunique())
print("\nRecords by borough:")
print(traffic["BOROUGH"].value_counts())

print("\nSpeed statistics:")
print(traffic["SPEED"].describe())

# Check passenger requests
print("\n--- Passenger Request Summary ---")
print(trips[[
    "request_datetime",
    "pickup_datetime",
    "PULocationID",
    "DOLocationID"
]].to_string(index=False))
