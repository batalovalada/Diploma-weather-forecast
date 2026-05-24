from torch.utils.data import DataLoader
import optuna

from model import *
from metrics_utils.metrics import *
from visualization.save_plots import *

# year
# base
# path_processed = '../../data/ConvLSTM/next_year_test_continuous_norm/base/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_continuous_norm/base/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/base/'

# # spatial
# path_processed = '../../data/ConvLSTM/next_year_test_continuous_norm/spatial/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_continuous_norm/spatial/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/spatial/'
#
# # temporal
# path_processed = '../../data/ConvLSTM/next_year_test_continuous_norm/temporal/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_continuous_norm/temporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/temporal/'

# # spatiotemporal
path_processed = '../../data/ConvLSTM/next_year_test_continuous_norm/spatiotemporal/processed/'
path_norm = '../../data/ConvLSTM/next_year_test_continuous_norm/spatiotemporal/norm_params/'
path_results = '../../reports/models/ConvLSTM/year/spatiotemporal/'

# year in seasonal balanced dataset
# base
# path_processed = '../../data/ConvLSTM/next_year_test_seasonal_norm/base/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_seasonal_norm/base/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/base/'

# # spatial
# path_processed = '../../data/ConvLSTM/next_year_test_seasonal_norm/spatial/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_seasonal_norm/spatial/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/spatial/'
#
# # temporal
# path_processed = '../../data/ConvLSTM/next_year_test_seasonal_norm/temporal/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_seasonal_norm/temporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/temporal/'

# # spatiotemporal
# path_processed = '../../data/ConvLSTM/next_year_test_seasonal_norm/spatiotemporal/processed/'
# path_norm = '../../data/ConvLSTM/next_year_test_seasonal_norm/spatiotemporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/spatiotemporal/'

# change in convlstm.py INPUT_DIM to 11, 13, 15, 17
path_results_testing = path_results+'test_next_year/'
# ======================================================
device = torch.device('cpu')

# ======================================================
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def load_norm_params(path):
    data = np.load(path, allow_pickle=True).item()
    return float(data['mean']), float(data['std'])

def load_model_params(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            if 'batch size:' in line:
                params['batch'] = int(line.split(': ')[1])
            elif 'hidden dim:' in line:
                params['hidden_dim'] = int(line.split(': ')[1])
            elif 'dropout:' in line:
                params['dropout'] = float(line.split(': ')[1])
    return params

# X, y, y_mask to tensor function
def tensor_data(X, y, mask):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool)
    )

def build_model(params):
    return ConvLSTM(hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)

#================= load data =======================
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')

params = load_model_params(path_results+'model_params.txt')

# ================= to tensors =========================
X_test, y_test, mask_test  = tensor_data(X_test, y_test, mask_test)

# ============= create Datasets, DataLoaders ================
Test_Dataset = WeatherDataset(X_test, y_test, mask_test)
Test_Loader = DataLoader(Test_Dataset, batch_size=params['batch'], shuffle=False)

# ===================== Test =======================
model = build_model(params)
model.load_state_dict(torch.load(path_results + 'model.pth',map_location=device))

model.eval()

all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for X_batch, y_batch, mask_batch in Test_Loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        pred = model(X_batch)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
        all_masks.append(mask_batch.cpu().numpy())

y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_targets, axis=0)
y_mask = np.concatenate(all_masks, axis=0)

# =========== remove normalize =======================
y_mean, y_std = load_norm_params(path_norm+'norm_params_y.npy')

y_pred = y_pred * y_std + y_mean
y_true = y_true * y_std + y_mean

# ============ recover T2 ==============================
T2_corrected = T2_wrf_test + y_pred
# return nan by mask, because model interp values in nodes, where era5 = nan
T2_corrected[~y_mask] = np.nan

# save and print metrics ==========================
define_and_save_metrics(y_pred, y_true, y_mask, T2_wrf_test, T2_corrected, T2_era5_test, path_results_testing)

# vizualization ========================================
save_scatter_plot(y_pred, y_true, y_mask, path_results_testing, 'ConvLSTM')
save_rmse_map(y_pred, y_true, y_mask, path_results_testing, 'ConvLSTM')
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results_testing, 'ConvLSTM')

# save ConvLSTM =======================================
# save results
np.savez(path_results_testing+"ConvLSTM.npz",
         y_pred=y_pred,
         y_true=y_true,
         mask=y_mask)
