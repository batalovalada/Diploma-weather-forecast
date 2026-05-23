import xarray as xr
import numpy as np
import glob

from config.data.features_config import features

# ========= paths =======================
# year !!!!!
# from config.data.split_year_config import *
# path_full = '../data/preprocessed/year/full/'
# path_selected = '../data/preprocessed/year/selected/'
# path_t2m = '../data/preprocessed/year/t2m/'
# path_wrf = sorted(glob.glob("../data/WRF_output/year/wrfout_d01_*"))
# paths_surface = sorted(glob.glob('../data/ERA5/year/surface/ERA5_surface_*.grib'))

# month !!!!!!
# from config.data.split_month_config import *
# path_full = '../data/preprocessed/month/full/'
# path_selected = '../data/preprocessed/month/selected/'
# path_t2m = '../data/preprocessed/month/t2m/'
# path_wrf = sorted(glob.glob("../data/WRF_output/month/wrfout_d01_*"))
# paths_surface = sorted(glob.glob('../data/ERA5/month/surface/ERA5_surface_*.grib'))

# year 2021 (only for tests)
from config.data.split_year_config import *
path_full = '../data/preprocessed/next_year_test/full/'
path_selected = '../data/preprocessed/next_year_test/selected/'
path_t2m = '../data/preprocessed/next_year_test/t2m/'
path_wrf = sorted(glob.glob("../data/WRF_output/next_year_test/wrfout_d01_*"))
paths_surface = sorted(glob.glob('../data/ERA5/next_year_test/surface/ERA5_surface_*.grib'))

# create dataset from .grib files function
def create_dataset_era5(paths, filter_arg):
    ds = xr.concat([xr.open_mfdataset(path, engine="cfgrib", backend_kwargs=filter_arg) for path in paths], dim='time')
    return ds


# =========== parameters =================
sn = [0, 15, 29, 45, 58]  #25 nodes
we = [0, 12, 23, 35, 46]

# =========== load wrf output and era5 data ==============
ds_wrf = xr.open_mfdataset(
    path_wrf,
    combine="nested",
    concat_dim="Time",
    parallel=True,
    chunks={'Time': 24},
)

ds_era5 = create_dataset_era5(paths_surface, {"filter_by_keys": {"typeOfLevel": "surface"}})

# ========== preprocessed wrf data: select features, create DataFrame, rename columns ========
# making 'XTIME' -> Coordinates and rename to 'time'
ds_wrf = ds_wrf.swap_dims({'Time': 'XTIME'})
ds_wrf = ds_wrf.rename({'XTIME': 'time'})

ds_wrf = ds_wrf[features]
df_wrf = ds_wrf.to_dataframe()
# save primary df[features] -------------------------------------
df_wrf.to_parquet(path_full + 'df_full_wrf.parquet', index=True) # -> eda.py

# ========== select nodes sn x we, time intersect ==============================
ds_wrf = ds_wrf.isel(south_north=sn, west_east=we)

# era5 has 6h step, wrf output has ~1h step
ds_wrf['time'] = ds_wrf.indexes['time'].floor('h') # wrf and era5 have ms difference
ds_era5['time'] = ds_era5.indexes['time'].floor('h')

common_time = np.intersect1d(ds_wrf.time, ds_era5.time)
ds_wrf = ds_wrf.sel(time=common_time)
ds_era5 = ds_era5.sel(time=common_time)

# selected nodes era5 interpolation
# get nodes coordinates (XLAT, XLONG)
lats = ds_wrf.XLAT.isel(time=0).reset_coords(drop=True)
lons = ds_wrf.XLONG.isel(time=0).reset_coords(drop=True)

ds_era5 = ds_era5.sel(time=common_time).interp(latitude=lats, longitude=lons)

# to save selected data in DataFrame -> eda.py
df_wrf = ds_wrf.to_dataframe().reset_index()
df_era5 = ds_era5.to_dataframe().reset_index()

# south_north, west_east -> node_id (using indexes from sn, we lists: sni-wei) for eda
df_eda = df_wrf.copy()
df_eda['node_id'] = df_eda['south_north'].astype(str) + '-' + df_eda['west_east'].astype(str)

# select data to check metrics (test) =================================================================
t2_era5_test = ds_era5['t2m'].sel(time=slice(test_start, test_end))
t2_wrf_test = ds_wrf['T2'].sel(time=slice(test_start, test_end))

# add features lat, lon, information about hour and day of year
lats = ds_wrf.XLAT.isel(time=0)
lons = ds_wrf.XLONG.isel(time=0)
ds_wrf['lat'] = lats.broadcast_like(ds_wrf['T2'])
ds_wrf['lon'] = lons.broadcast_like(ds_wrf['T2'])
ds_wrf = ds_wrf.drop_vars(['XLAT', 'XLONG'])

time = ds_wrf['time']

ds_wrf["hour_sin"] = np.sin(2 * np.pi * time.dt.hour / 24).broadcast_like(ds_wrf['T2'])
ds_wrf["hour_cos"] = np.cos(2 * np.pi * time.dt.hour / 24).broadcast_like(ds_wrf['T2'])

ds_wrf["day_sin"] = np.sin(2 * np.pi * time.dt.dayofyear / 365).broadcast_like(ds_wrf['T2'])
ds_wrf["day_cos"] = np.cos(2 * np.pi * time.dt.dayofyear / 365).broadcast_like(ds_wrf['T2'])

t2_wrf_test  = t2_wrf_test.drop_vars(['XLAT', 'XLONG'])

# save clean df, ds in selected nodes and intersected time -------------------------------------
df_eda.to_parquet(path_selected + 'df_selected_wrf_eda.parquet', index=False) # -> eda.py
df_era5.to_parquet(path_selected + 'df_selected_era5_eda.parquet', index=False) # -> eda.py
ds_wrf.to_netcdf(path_selected + "ds_selected_wrf.nc") # -> model preprocess
ds_era5.to_netcdf(path_selected + 'ds_selected_era5.nc') # -> model preprocess

# save t2 era5 and t2 wrf ----------------------------------------------------------------------
t2_era5_test.to_netcdf(path_t2m + "t2_era5_test.nc") # -> model preprocess
t2_wrf_test.to_netcdf(path_t2m + "t2_wrf_test.nc") # -> model preprocess