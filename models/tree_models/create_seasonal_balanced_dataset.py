import numpy as np
import xarray as xr

from config.data.features_config import features
from config.hyperparameters.tree_models import lags
from config.data.split_seasonal_year_config import *
# from config.data.split_next_year_test_config import test_blocks

path_selected = '../../data/preprocessed/year/selected/'
# path_selected = '../../data/preprocessed/next_year_test_seasonal_norm/selected/'
# ============= paths =======================

# year base !!!
# path_processed = '../../data/tree_models/seasonal_year/base/processed/'

# year spatial !!!
# path_processed = '../../data/tree_models/seasonal_year/spatial/processed/'

# year temporal !!!
# path_processed = '../../data/tree_models/seasonal_year/temporal/processed/'

# year spatiotemporal !!!
path_processed = '../../data/tree_models/seasonal_year/spatiotemporal/processed/'


# 2021 test
# year base !!!
# path_processed = '../../data/tree_models/next_year_test_seasonal_norm/base/processed/'

# year spatial !!!
# path_processed = '../../data/tree_models/next_year_test_seasonal_norm/spatial/processed/'

# year temporal !!!
# path_processed = '../../data/tree_models/next_year_test_seasonal_norm/temporal/processed/'

# year spatiotemporal !!!
# path_processed = '../../data/tree_models/next_year_test_seasonal_norm/spatiotemporal/processed/'

#===========functions =========================
def create_block_lags(ds, target, time_blocks):
    X_list = []
    y_list = []
    time_list = []

    for start_block, end_block in time_blocks:
        # lags only from continuous time interval
        ds_block = ds.sel(time=slice(start_block, end_block))
        target_block = target.sel(time=slice(start_block, end_block))

        for t in range(lags, len(ds_block.time)):
            X_t = []
            for lag in range(lags):
                X_t.append(ds_block[features].isel(time=t - lag).to_array().values)

            X_t = np.concatenate(X_t, axis=0)  # shape = (features*lags, H, W)
            y_t = target_block.isel(time=t).values  # shape = (H, W)

            X_list.append(X_t)
            y_list.append(y_t)
            time_list.append(ds_block.time.isel(time=t).values)

    # add time axis
    X = np.stack(X_list)  # (time, features*lags, H, W)
    y = np.stack(y_list)  # (time, H, W)
    return X, y, np.array(time_list)

def flatten_data(X, y):
    t, f, h, w = X.shape
    X_flat = X.transpose(0, 2, 3, 1).reshape(-1, f)  # (samples, features), samples = time x H x W
    y_flat = y.reshape(-1)  # (time x H x W, )
    return X_flat, y_flat, t, h, w

def add_coords(X, lat_flatten, lon_flatten, time):
    # add coordinate to each time moment
    lat_tiled = np.tile(lat_flatten, time)
    lon_tiled = np.tile(lon_flatten, time)

    return np.concatenate([X, lat_tiled[:, None], lon_tiled[:, None]], axis=1)

def add_time_features(X, hour_sin, hour_cos, day_sin, day_cos):
    hour_sin_flat = hour_sin.reshape(-1, 1)
    hour_cos_flat = hour_cos.reshape(-1, 1)
    day_sin_flat = day_sin.reshape(-1, 1)
    day_cos_flat = day_cos.reshape(-1, 1)
    return np.concatenate([
        X,
        hour_sin_flat,
        hour_cos_flat,
        day_sin_flat,
        day_cos_flat,
    ], axis=1)

def apply_nan_mask(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask], mask

# ============ load data ================
ds_wrf = xr.open_dataset(path_selected+'ds_selected_wrf.nc')
ds_era5 = xr.open_dataset(path_selected+'ds_selected_era5.nc')

# =============== create target =============
target = ds_era5['t2m'] - ds_wrf['T2']

#======================= split and create time lags ===============================
X_train, y_train, time_rf_train = create_block_lags(ds_wrf, target, train_blocks)
X_val, y_val, time_rf_val = create_block_lags(ds_wrf, target, val_blocks)
X_test, y_test, time_rf_test = create_block_lags(ds_wrf, target, test_blocks)

# ============= flatten ====================
X_train, y_train, T_train, H, W = flatten_data(X_train, y_train)
X_val, y_val, T_val, _, _ = flatten_data(X_val, y_val)
X_test, y_test, T_test, _, _ = flatten_data(X_test, y_test)

# ========= add extra features ================
# add coordinates features (spatial)
lats = ds_wrf.lat.isel(time=0).values # shape = (H, W)
lons = ds_wrf.lon.isel(time=0).values

lat_flatten = lats.flatten() # shape = (H*W, )
lon_flatten = lons.flatten()

X_train = add_coords(X_train, lat_flatten, lon_flatten, T_train)
X_val = add_coords(X_val, lat_flatten, lon_flatten, T_val)
X_test = add_coords(X_test, lat_flatten, lon_flatten, T_test)

# add time and day features (temporal)
hour_sin_train = ds_wrf.hour_sin.sel(time=time_rf_train).values
hour_cos_train = ds_wrf.hour_cos.sel(time=time_rf_train).values

hour_sin_val = ds_wrf.hour_sin.sel(time=time_rf_val).values
hour_cos_val = ds_wrf.hour_cos.sel(time=time_rf_val).values

hour_sin_test = ds_wrf.hour_sin.sel(time=time_rf_test).values
hour_cos_test = ds_wrf.hour_cos.sel(time=time_rf_test).values

day_sin_train = ds_wrf.day_sin.sel(time=time_rf_train).values
day_cos_train = ds_wrf.day_cos.sel(time=time_rf_train).values

day_sin_val = ds_wrf.day_sin.sel(time=time_rf_val).values
day_cos_val = ds_wrf.day_cos.sel(time=time_rf_val).values

day_sin_test = ds_wrf.day_sin.sel(time=time_rf_test).values
day_cos_test = ds_wrf.day_cos.sel(time=time_rf_test).values

X_train = add_time_features(X_train, hour_sin_train, hour_cos_train, day_sin_train, day_cos_train)
X_val = add_time_features(X_val, hour_sin_val, hour_cos_val, day_sin_val, day_cos_val)
X_test = add_time_features(X_test, hour_sin_test, hour_cos_test, day_sin_test, day_cos_test)

# ============ delete samples with nan values ========
# era5 has nan values => y has too
X_train, y_train, mask_train = apply_nan_mask(X_train, y_train)
X_val, y_val, mask_val = apply_nan_mask(X_val, y_val)
X_test, y_test, mask_test = apply_nan_mask(X_test, y_test)

# ========= T2 to ndarray, add mask (ERA5 has nan)=======================
t2_wrf_test = ds_wrf.sel(time=time_rf_test)['T2'].values # -> shape (time, H, W)
t2_era5_test = ds_era5.sel(time=time_rf_test)['t2m'].values # -> shape (time, H, W)
t2_mask_test = ~np.isnan(t2_era5_test) # shape (time, H, W)

# =========== save data ====================
np.save(path_processed+'t2_wrf_test.npy', t2_wrf_test)
np.save(path_processed+'t2_era5_test.npy', t2_era5_test)
np.save(path_processed+'t2_mask_test.npy', t2_mask_test)

np.savez(path_processed+'train.npz', X=X_train, y=y_train, mask=mask_train, time=time_rf_train)
np.savez(path_processed+'val.npz', X=X_val, y=y_val, mask=mask_val, time=time_rf_val)
np.savez(path_processed+'test.npz', X=X_test, y=y_test, mask=mask_test, time=time_rf_test)
