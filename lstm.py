import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback

# Параметры
features = ['temparature', 'humidity', 'pressure', 'windspeed', 'cloud']
target = 'rainfall'
n_days = 7

# --- Функция создания последовательностей ---
def create_sequences(df, features, target, n_days):
    X, y = [], []
    for i in range(n_days, len(df)):
        X.append(df[features].iloc[i - n_days:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)

# --- Callback для расчета F1 на валидации ---
class F1ScoreCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.y_val = val_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = (self.model.predict(self.X_val) > 0.5).astype(int)
        f1 = f1_score(self.y_val, val_pred)
        self.f1_scores.append(f1)
        print(f" — val_f1: {f1:.4f}")

# --- Загрузка и подготовка данных ---
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

# Преобразование даты и установка индекса
for df in [train_df, test_df]:
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)

# Разделение train на train/val
val_size = 300
val_df = train_df[-val_size:]
train_df = train_df[:-val_size]

# Обзор train
print("Train data preview:")
print(train_df.head())
print(train_df.info())
print(train_df.describe())
print("\nMissing values in train:")
print(train_df.isnull().sum())
print("\nTarget distribution in train:")
print(train_df[target].value_counts(normalize=True).rename("proportion"))

# Визуализация распределения
train_df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Распределение признаков (train)')
plt.show()

# Корреляция
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица (train)')
plt.show()

# Создание последовательностей
X_train, y_train = create_sequences(train_df, features, target, n_days)
X_val, y_val = create_sequences(val_df, features, target, n_days)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(-1, n_days, len(features))
X_val_scaled = scaler.transform(X_val.reshape(-1, len(features))).reshape(-1, n_days, len(features))

# Модель
model = Sequential([
    LSTM(64, activation='relu', input_shape=(n_days, len(features))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение
f1_callback = F1ScoreCallback(val_data=(X_val_scaled, y_val))
history = model.fit(
    X_train_scaled, y_train,
    epochs=10, batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[f1_callback]
)

# Финальная F1-оценка
y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
f1 = f1_score(y_val, y_pred)
print(f"\nF1-Score на валидации: {f1:.4f}")

# Визуализация Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], label='Accuracy на валидации')
plt.title('Accuracy модели')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация F1-Score
plt.figure(figsize=(10, 6))
plt.plot(f1_callback.f1_scores, label='F1-score на валидации', color='green')
plt.title('F1-score модели')
plt.xlabel('Эпоха')
plt.ylabel('F1-score')
plt.legend()
plt.grid(True)
plt.show()

# --- Предсказание на тесте ---
test_df['rainfall'] = 0  # временно
X_test, _ = create_sequences(test_df, features, target, n_days)

X_test_scaled = scaler.transform(X_test.reshape(-1, len(features))).reshape(-1, n_days, len(features))
predictions = (model.predict(X_test_scaled) > 0.5).astype(int)

# Сохранение
submission = pd.DataFrame({
    'day': test_df.index[n_days:],
    'predicted_rainfall': predictions.ravel()
})
submission.to_csv('submission.csv', index=False)
print("\nФайл submission.csv успешно сохранён.")