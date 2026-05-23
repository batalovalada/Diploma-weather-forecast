from torch.utils.data import DataLoader
import optuna
import copy

from model import *
from config.hyperparameters.convlstm import EPOCHS
from metrics_utils.metrics import *
from visualization.save_plots import *

# month base
# path_processed = '../../data/ConvLSTM/month/base/processed/'
# path_norm = '../../data/ConvLSTM/month/base/norm_params/'
# path_results = '../../reports/models/ConvLSTM/month/base/'

# month spatial
# path_processed = '../../data/ConvLSTM/month/spatial/processed/'
# path_norm = '../../data/ConvLSTM/month/spatial/norm_params/'
# path_results = '../../reports/models/ConvLSTM/month/spatial/'


# year
# base
# path_processed = '../../data/ConvLSTM/year/base/processed/'
# path_norm = '../../data/ConvLSTM/year/base/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/base/'

# # spatial
# path_processed = '../../data/ConvLSTM/year/spatial/processed/'
# path_norm = '../../data/ConvLSTM/year/spatial/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/spatial/'
#
# # temporal
# path_processed = '../../data/ConvLSTM/year/temporal/processed/'
# path_norm = '../../data/ConvLSTM/year/temporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/temporal/'

# # spatiotemporal
# path_processed = '../../data/ConvLSTM/year/spatiotemporal/processed/'
# path_norm = '../../data/ConvLSTM/year/spatiotemporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/year/spatiotemporal/'

# year in seasonal balanced dataset
# base
# path_processed = '../../data/ConvLSTM/seasonal_year/base/processed/'
# path_norm = '../../data/ConvLSTM/seasonal_year/base/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/base/'

# # spatial
# path_processed = '../../data/ConvLSTM/seasonal_year/spatial/processed/'
# path_norm = '../../data/ConvLSTM/seasonal_year/spatial/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/spatial/'
#
# # temporal
# path_processed = '../../data/ConvLSTM/seasonal_year/temporal/processed/'
# path_norm = '../../data/ConvLSTM/seasonal_year/temporal/norm_params/'
# path_results = '../../reports/models/ConvLSTM/seasonal_year/temporal/'

# # spatiotemporal
path_processed = '../../data/ConvLSTM/seasonal_year/spatiotemporal/processed/'
path_norm = '../../data/ConvLSTM/seasonal_year/spatiotemporal/norm_params/'
path_results = '../../reports/models/ConvLSTM/seasonal_year/spatiotemporal/'

# change in convlstm.py INPUT_DIM to 11, 13, 15, 17

# ============= parameters =============================
device = torch.device('cpu')

# ======================================================
# load npz data function
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y'], data['mask']

def load_norm_params(path):
    data = np.load(path, allow_pickle=True).item()
    return float(data['mean']), float(data['std'])

# X, y, y_mask to tensor function
def tensor_data(X, y, mask):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool)
    )

# loss function =====================================
def masked_mse(pred, target, mask):
    return torch.mean((pred[mask] - target[mask])**2)

def masked_rmse(pred, target, mask):
    return torch.sqrt(masked_mse(pred, target, mask))

def build_model(params):
    return ConvLSTM(hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)


def train_model(model, params, Train_Dataset, Val_Dataset):
    Train_Loader = DataLoader(Train_Dataset, batch_size=params['batch'], shuffle=True)
    Val_Loader = DataLoader(Val_Dataset, batch_size=params['batch'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # =================== Training ==============================
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float('inf')
    counter = 0
    patience = 5

    train_losses = []
    val_losses = []
    val_rmses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch, mask_batch in Train_Loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = masked_mse(pred, y_batch, mask_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(Train_Loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_rmse = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in Val_Loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                pred = model(X_batch)
                # loss function mse
                loss = masked_mse(pred, y_batch, mask_batch)
                # opt hyperparams by optuna with rmse
                rmse = masked_rmse(pred, y_batch, mask_batch)
                val_loss += loss.item()
                val_rmse += rmse.item()

        val_loss /= len(Val_Loader)
        val_rmse /= len(Val_Loader)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        # early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_state)
    return train_losses, val_losses, val_rmses


# optuna function ====================================
def objective(trial):
    params={
        'batch': trial.suggest_categorical('batch', [4, 8, 16]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 96]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
    }
    model = build_model(params)
    _, _, val_rmses = train_model(model, params, Train_Dataset, Val_Dataset)
    val_rmse = min(val_rmses)
    return val_rmse

#================= load data =======================
X_train, y_train, mask_train = load_npz(path_processed + 'train.npz')
X_val, y_val, mask_val = load_npz(path_processed + 'val.npz')
X_test, y_test, mask_test = load_npz(path_processed + 'test.npz')

T2_wrf_test = np.load(path_processed+'t2_wrf_test.npy')
T2_era5_test = np.load(path_processed+'t2_era5_test.npy')

# ================= to tensors =========================
X_train, y_train, mask_train = tensor_data(X_train, y_train, mask_train)
X_val, y_val, mask_val = tensor_data(X_val, y_val, mask_val)
X_test, y_test, mask_test  = tensor_data(X_test, y_test, mask_test)

# ============= create Datasets, DataLoaders ================
Train_Dataset = WeatherDataset(X_train, y_train, mask_train)
Val_Dataset = WeatherDataset(X_val, y_val, mask_val)
Test_Dataset = WeatherDataset(X_test, y_test, mask_test)

# ================ select model's params with Optuna ====================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

print("Best params:", best_params)
print("Best value:", study.best_value)

# ===================== training ConvLSTM model ======================
best_model = build_model(best_params)
train_losses, val_losses, _ = train_model(best_model, best_params, Train_Dataset, Val_Dataset)

# ===================== Test =======================
Test_Loader = DataLoader(Test_Dataset, batch_size=best_params['batch'], shuffle=False)

best_model.eval()

all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for X_batch, y_batch, mask_batch in Test_Loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        pred = best_model(X_batch)

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
define_and_save_metrics(y_pred, y_true, y_mask, T2_wrf_test, T2_corrected, T2_era5_test, path_results)

# vizualization ========================================
save_loss_plot(train_losses, val_losses, path_results)
save_scatter_plot(y_pred, y_true, y_mask, path_results, 'ConvLSTM')
save_rmse_map(y_pred, y_true, y_mask, path_results, 'ConvLSTM')
save_corrected_map(T2_wrf_test, T2_era5_test, T2_corrected, path_results, 'ConvLSTM')

# save ConvLSTM =======================================
# save model
torch.save(best_model.state_dict(), path_results+"model.pth")

# save results
np.savez(path_results+"ConvLSTM.npz",
         y_pred=y_pred,
         y_true=y_true,
         mask=y_mask)

# save train params
with open(path_results + "model_params.txt", "w") as f:
    f.write("Optimized params by Optuna:\n")
    f.write(f"batch size: {best_params['batch']}\n")
    f.write(f"hidden dim: {best_params['hidden_dim']}\n")
    f.write(f"lr: {best_params['lr']}\n")
    f.write(f"dropout: {best_params['dropout']}\n\n")
    f.write("Fixed params:\n")
    f.write(f"epochs: {EPOCHS}\n")
    f.write(f"input dim: {INPUT_DIM}\n")
    f.write(f"kernel size: {KERNEL_SIZE}\n")