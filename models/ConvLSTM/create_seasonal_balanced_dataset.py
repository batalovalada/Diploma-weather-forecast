import xarray as xr
import numpy as np
import copy

from config.data.features_config import features, spatial_features, temporal_features
from config.data.split_seasonal_year_config import *

# from config.data.split_seasonal_year_config import train_blocks
# from config.data.split_next_year_test_config import test_blocks

path_selected = '../../data/preprocessed/year/selected/'
# path_selected_test = '../../data/preprocessed/next_year_test/selected/'

# year base
# path_norm = '../../data/ConvLSTM/seasonal_year/base/norm_params/'
# path_processed = '../../data/ConvLSTM/seasonal_year/base/processed/'

# year spatial
path_norm = '../../data/ConvLSTM/seasonal_year/spatial/norm_params/'
path_processed = '../../data/ConvLSTM/seasonal_year/spatial/processed/'

# year temporal
# path_norm = '../../data/ConvLSTM/seasonal_year/temporal/norm_params/'
# path_processed = '../../data/ConvLSTM/seasonal_year/temporal/processed/'

# year spatiotemporal
# path_norm = '../../data/ConvLSTM/seasonal_year/spatiotemporal/norm_params/'
# path_processed = '../../data/ConvLSTM/seasonal_year/spatiotemporal/processed/'




# 2021 test data
# year base
# path_norm = '../../data/ConvLSTM/next_year_test/base/norm_params/'
# path_processed = '../../data/ConvLSTM/next_year_test/base/processed/'

# year spatial
# path_norm = '../../data/ConvLSTM/next_year_test/spatial/norm_params/'
# path_processed = '../../data/ConvLSTM/next_year_test/spatial/processed/'

# year temporal
# path_norm = '../../data/ConvLSTM/next_year_test/temporal/norm_params/'
# path_processed = '../../data/ConvLSTM/next_year_test/temporal/processed/'

# year spatiotemporal
# path_norm = '../../data/ConvLSTM/next_year_test/spatiotemporal/norm_params/'
# path_processed = '../../data/ConvLSTM/next_year_test/spatiotemporal/processed/'
# ======= parameters ================================================
lookback = 4
horizon = 1

# normalize function
def normalize(data, mean, std):
    return (data - mean) / std

# ============== create norm_params function =============
def create_blocks_sequences(X, y, mask, time_blocks, lookback, horizon):
    X_seq, y_seq, mask_seq, time_seq = [], [], [], []

    for start_block, end_block in time_blocks:
        # sequences only from continuous time interval
        X_block = X.sel(time=slice(start_block, end_block))
        y_block = y.sel(time=slice(start_block, end_block))
        mask_block = mask.sel(time=slice(start_block, end_block))

        for i in range(len(X_block.time) - lookback - horizon + 1):
            X_seq.append(X_block.isel(time=slice(i, i+lookback)).values)

            y_t = i+lookback+horizon-1
            y_seq.append(y_block.isel(time=y_t).values)
            mask_seq.append(mask_block.isel(time=y_t).values)
            time_seq.append(X_block.time.values[y_t])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(mask_seq, dtype=bool),
        np.array(time_seq, dtype='datetime64[ns]'),
    )

# =============================== load data ============================
ds_wrf = xr.open_dataset(path_selected+'ds_selected_wrf.nc')
ds_era5 = xr.open_dataset(path_selected+'ds_selected_era5.nc')

# ds_wrf_test = xr.open_dataset(path_selected_test+'ds_selected_wrf.nc')
# ds_era5_test = xr.open_dataset(path_selected_test+'ds_selected_era5.nc')

# to choose, what dataset do you need:
# base
# ds_wrf = ds_wrf.drop_vars(spatial_features+temporal_features)
# ds_wrf_test = ds_wrf_test.drop_vars(spatial_features+temporal_features)
# spatial
ds_wrf = ds_wrf.drop_vars(temporal_features)
# ds_wrf_test = ds_wrf_test.drop_vars(temporal_features)
# # temporal
# ds_wrf = ds_wrf.drop_vars(spatial_features)
# ds_wrf_test = ds_wrf_test.drop_vars(spatial_features)
# spatiotemporal - nothing

# creating X, y with train, val, test ======================================
X_ds = copy.deepcopy(ds_wrf)
y_ds = ds_era5['t2m'] - ds_wrf['T2']

y_mask = (~np.isnan(y_ds)) # -> loss
y_ds = y_ds.fillna(0) # y_ds has nan

# X_test = ds_wrf_test
# y_test = ds_era5_test['t2m'] - ds_wrf_test['T2']
#
# y_mask_test = (~np.isnan(y_test)) # -> loss
# y_test = y_test.fillna(0) # y_ds has nan

# ========================== normalize =======================================
# select train data to define mean, std
train_X_list = []
train_y_list = []

for start_block, end_block in train_blocks:
    train_X_list.append(X_ds.sel(time=slice(start_block, end_block)))
    train_y_list.append(y_ds.sel(time=slice(start_block, end_block)))

X_train_blocks = xr.concat(train_X_list, dim='time')
y_train_blocks = xr.concat(train_y_list, dim='time')

# define std, mean for train data X and y
X_train_blocks = X_train_blocks.to_array().transpose("time","variable","south_north","west_east")
X_mean = X_train_blocks.sel(variable=features).mean(dim=['time', 'south_north', 'west_east'])
X_std = X_train_blocks.sel(variable=features).std(dim=['time', 'south_north', 'west_east'])

y_mean = y_train_blocks.mean()
y_std = y_train_blocks.std()

# normalize main features data
X_all = X_ds.to_array().transpose("time","variable","south_north","west_east")

X_all.loc[dict(variable=features)] = normalize(X_all.sel(variable=features), X_mean, X_std)

# normalize separated spatial features (comment if dataset is base/temporal)
X_spatial_mean = X_train_blocks.sel(variable=spatial_features).mean(dim=['time', 'south_north', 'west_east'])
X_spatial_std = X_train_blocks.sel(variable=spatial_features).std(dim=['time', 'south_north', 'west_east'])

X_all.loc[dict(variable=spatial_features)] = normalize(X_all.sel(variable=spatial_features), X_spatial_mean, X_spatial_std)

# normalize target
y_all = normalize(y_ds, y_mean, y_std)

# ================================= create sequences  =================================

X_train_seq, y_train_seq, mask_train_seq, time_train_seq = create_blocks_sequences(X_all, y_all, y_mask, train_blocks, lookback, horizon)
X_val_seq, y_val_seq, mask_val_seq, time_val_seq  = create_blocks_sequences(X_all, y_all, y_mask, val_blocks, lookback, horizon)
X_test_seq, y_test_seq, mask_test_seq, time_test_seq  = create_blocks_sequences(X_all, y_all, y_mask, test_blocks, lookback, horizon)

# T2 wrf, era5 to ndarray for check model result ============================
t2_wrf= ds_wrf.sel(time=time_test_seq)['T2'].values
t2_era5 = ds_era5.sel(time=time_test_seq)['t2m'].values

# t2_wrf= ds_wrf_test.sel(time=time_test_seq)['T2'].values
# t2_era5 = ds_era5_test.sel(time=time_test_seq)['t2m'].values

# ========================= save normalize parameters ====================
norm_params_X = {
    'mean': X_mean,
    'std': X_std
}

norm_params_y = {
    'mean': y_mean,
    'std': y_std
}
np.save(path_norm+'norm_params_X.npy', norm_params_X)
np.save(path_norm+'norm_params_y.npy', norm_params_y)

# ========================= save preprocess data ====================
np.savez(path_processed+'train.npz', X=X_train_seq, y=y_train_seq, mask=mask_train_seq, time=time_train_seq)
np.savez(path_processed+'val.npz', X=X_val_seq, y=y_val_seq, mask=mask_val_seq, time=time_val_seq)
np.savez(path_processed+'test.npz', X=X_test_seq, y=y_test_seq, mask=mask_test_seq, time=time_test_seq)

np.save(path_processed+'t2_wrf_test.npy', t2_wrf)
np.save(path_processed+'t2_era5_test.npy', t2_era5)