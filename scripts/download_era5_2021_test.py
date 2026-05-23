import cdsapi
import os
import calendar

c = cdsapi.Client(
    retry_max=10,
    sleep_max=60
)

YEAR = "2021"

MONTH = "12"

DAYS = ["16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
        ]

AREA = [62.5, 27.0, 57.5, 35.0]  # N, W, S, E

TIMES = [
    "00:00", "03:00", "06:00", "09:00",
    "12:00", "15:00", "18:00", "21:00"
]

PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600",
    "500", "400", "300", "250", "200", "100"
]

os.makedirs("../data/ERA5/next_year/pressure", exist_ok=True)
os.makedirs("../data/ERA5/next_year/surface", exist_ok=True)
# =========================
# PRESSURE LEVELS
# =========================

filename = f"../data/ERA5/next_year/pressure/ERA5_pressure_{YEAR}_{MONTH}_second_half.grib"

if not os.path.exists(filename):
    print(f"Downloading pressure levels for {YEAR}-{MONTH}...")

    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": [
                "geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity"
            ],
            "pressure_level": PRESSURE_LEVELS,
            "year": YEAR,
            "month": MONTH,
            "day": DAYS,
            "time": TIMES,
            "area": AREA
        },
        filename
    )
else:
    print(f"File {filename} already exists. Skipping.")
# =========================
# SURFACE LEVELS
# =========================

filename = f"../data/ERA5/next_year/surface/ERA5_surface_{YEAR}_{MONTH}_second_half.grib"

if not os.path.exists(filename):
    print(f"Downloading surface levels for {YEAR}-{MONTH}...")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "geopotential",
                "mean_sea_level_pressure",
                "skin_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "snow_depth",
                "orography",
                "land_sea_mask",
            ],
            "year": YEAR,
            "month": MONTH,
            "day": DAYS,
            "time": TIMES,
            "area": AREA
        },
        filename
    )
else:
    print(f"File {filename} already exists. Skipping.")

print("Download completed.")

