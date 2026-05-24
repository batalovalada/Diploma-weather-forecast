import joblib
import optuna
from sklearn.metrics import mean_squared_error

from config.hyperparameters.tree_models import H, W
from metrics_utils.metrics import *
from visualization.save_plots import *

# model name
# name_model = 'RF'
name_model = 'XGBoost'

# chose test 2021 data (common for RF and XGB)
# base
# path_processed = '../../data/tree_models/next_year_test/base/processed/'
# spatial
# path_processed = '../../data/tree_models/next_year_test/spatial/processed/'
# # temporal
# path_processed = '../../data/tree_models/next_year_test/temporal/processed/'
# # spatiotemporal
path_processed = '../../data/tree_models/next_year_test/spatiotemporal/processed/'


# chose model
# Random Forest year
# #  base
# path_results = '../../reports/models/RF/year/base/'
# spatial
# path_results = '../../reports/models/RF/year/spatial/'
# # temporal
# path_results = '../../reports/models/RF/year/temporal/'
# # spatiotemporal
# path_results = '../../reports/models/RF/year/spatiotemporal/'



# XGBoost year
 # base
# path_results = '../../reports/models/XGB/year/base/'
# # spatial
# path_results = '../../reports/models/XGB/year/spatial/'
# # temporal
# path_results = '../../reports/models/XGB/year/temporal/'
# # spatiotemporal
# path_results = '../../reports/models/XGB/year/spatiotemporal/'



# Random Forest seasonal balanced year
#  base
# path_results = '../../reports/models/RF/seasonal_year/base/'

# spatial
# path_results = '../../reports/models/RF/seasonal_year/spatial/'

# # temporal
# path_results = '../../reports/models/RF/seasonal_year/temporal/'

# # spatiotemporal
# path_results = '../../reports/models/RF/seasonal_year/spatiotemporal/'


# XGBoost  seasonal balanced year
 # base
# path_results = '../../reports/models/XGB/seasonal_year/base/'

# # spatial
# path_results = '../../reports/models/XGB/seasonal_year/spatial/'

# # temporal
# path_results = '../../reports/models/XGB/seasonal_year/temporal/'

# # spatiotemporal
path_results = '../../reports/models/XGB/seasonal_year/spatiotemporal/'



# common
path_results_testing = path_results+'test_next_year/'

# ======== function ==============
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask'], data['time']

def restore_masked_grid(mask, samples, time, H, W):
    bias_pred = np.full(mask.shape, np.nan)
    bias_pred[mask] = samples
    bias_pred= bias_pred.reshape(time, H, W)
    return bias_pred

# ============ load data =========
X_test, y_test, mask_test, time_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')

# =========== test =========================
model = joblib.load(path_results+f'{name_model}_model.pkl')

test_pred = model.predict(X_test)

# #============= restore test data =============================
T_test = len(time_test)

test_pred_restored = restore_masked_grid(mask_test, test_pred, T_test, H, W)
y_test_restored = restore_masked_grid(mask_test, y_test, T_test, H, W)
mask_test_reshaped = mask_test.reshape(T_test, H, W)

T2_corrected = T2_wrf_test + test_pred_restored

# save and print metrics ==========================
define_and_save_metrics(test_pred_restored, y_test_restored, mask_test_reshaped, T2_wrf_test, T2_corrected, T2_era5_test, path_results_testing)

# vizualization ========================================
save_radial_distribution_plot(test_pred_restored,y_test_restored, mask_test_reshaped, path_results_testing, name_model)
save_scatter_plot(test_pred_restored, y_test_restored, mask_test_reshaped, path_results_testing, name_model)
save_rmse_map(test_pred_restored, y_test_restored, mask_test_reshaped, path_results_testing, name_model)
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results_testing, name_model)

# save model ============================================
# save results
np.savez(
    path_results_testing+f"results_{name_model}.npz",
    y_pred=test_pred_restored,
    y_true=y_test_restored,
    mask=mask_test_reshaped,
)