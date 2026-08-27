import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import re
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# --- Initial Setup ---
tf.random.set_seed(42)
np.random.seed(42)

print("=== SIREN for Mach Prediction V17.0 (Advanced Tuning - TensorFlow) ===")

# --- 1. Custom Layers (Identical to previous versions) ---
class RowdyAdaptiveActivationLayer(layers.Layer):
    """Custom Rowdy Adaptive Activation Layer in TensorFlow."""
    def __init__(self, units, initial_a=0.1, initial_c=0.1, initial_a1=0.0, initial_F1=0.1, initial_c1=0.0, **kwargs):
        super(RowdyAdaptiveActivationLayer, self).__init__(**kwargs)
        self.units = units
        self.initial_a = initial_a
        self.initial_c = initial_c
        self.initial_a1 = initial_a1
        self.initial_F1 = initial_F1
        self.initial_c1 = initial_c1

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.a = self.add_weight(name='a', shape=(), initializer=keras.initializers.Constant(self.initial_a), trainable=True)
        self.c = self.add_weight(name='c', shape=(), initializer=keras.initializers.Constant(self.initial_c), trainable=True)
        self.a1 = self.add_weight(name='a1', shape=(), initializer=keras.initializers.Constant(self.initial_a1), trainable=True)
        self.F1 = self.add_weight(name='F1', shape=(), initializer=keras.initializers.Constant(self.initial_F1), trainable=True)
        self.c1 = self.add_weight(name='c1', shape=(), initializer=keras.initializers.Constant(self.initial_c1), trainable=True)
        self.linear_weight = self.add_weight(shape=(input_dim, self.units), initializer=keras.initializers.GlorotUniform(), trainable=True, name='linear_weight')
        self.linear_bias = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='linear_bias')
        super(RowdyAdaptiveActivationLayer, self).build(input_shape)

    def call(self, inputs):
        linear_output = tf.matmul(inputs, self.linear_weight) + self.linear_bias
        base_activation = tf.tanh(10 * self.a * linear_output + self.c)
        sin_component = 10 * self.a1 * tf.sin(10 * self.F1 * linear_output + self.c1)
        return base_activation + sin_component

class FourierFeatureLayer(layers.Layer):
    """Maps input coordinates to a higher-dimensional space using Fourier features."""
    def __init__(self, num_frequencies=256, sigma=10.0, trainable=True, **kwargs):
        super(FourierFeatureLayer, self).__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.is_trainable = trainable

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.B = self.add_weight(name='fourier_matrix', shape=(input_dim, self.num_frequencies),
                                 initializer=keras.initializers.RandomNormal(stddev=self.sigma), trainable=self.is_trainable)
        super(FourierFeatureLayer, self).build(input_shape)

    def call(self, inputs):
        x_proj = tf.matmul(inputs, self.B)
        return tf.concat([tf.sin(2 * np.pi * x_proj), tf.cos(2 * np.pi * x_proj)], axis=-1)

# --- 2. Optimized Model Architecture (Identical to V16.0) ---
def build_optimized_ffn(spatial_input_dim, output_dim, hidden_units=512, num_hidden_layers=8):
    """Builds an optimized Feed-Forward Network (FFN)."""
    inputs = keras.Input(shape=(spatial_input_dim,))
    x = FourierFeatureLayer(num_frequencies=256, sigma=10.0)(inputs)
    for _ in range(num_hidden_layers):
        x = RowdyAdaptiveActivationLayer(hidden_units)(x)
    outputs = layers.Dense(output_dim, activation='linear', name='output_coefficients')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='Optimized_FFN_Model')
    return model

# --- 3. Data Loading and Processing (Identical to previous versions) ---
def load_structured_grid_data(filename="structured_grid.DAT"):
    if not os.path.exists(filename): return None, None
    with open(filename, 'r') as f: lines = f.readlines()
    variables_line, zone_line_idx = None, -1
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('VARIABLES'): variables_line = line
        elif line.strip().upper().startswith('ZONE'): zone_line_idx = i; break
    if variables_line is None or zone_line_idx == -1: return None, None
    column_names = re.findall(r'"([^"]*)"', variables_line.split('=', 1)[1])
    if not column_names: return None, None
    header_lines = lines[:zone_line_idx + 1]
    all_data_text = "".join(line.strip() + " " for line in lines[zone_line_idx + 1:] if line.strip() and not line.startswith('#'))
    numeric_data = [float(val) for val in all_data_text.split() if val]
    num_cols = len(column_names)
    if num_cols == 0: return None, None
    num_rows = len(numeric_data) // num_cols
    data_array = np.array(numeric_data[:num_rows * num_cols]).reshape(num_rows, num_cols)
    df = pd.DataFrame(data_array, columns=column_names)
    print(f"Loaded {len(df)} data points from {filename}")
    return df, header_lines

def filter_valid_points(df, large_value=1.0e+30):
    valid_mask = np.abs(df['MA'] - large_value) > 1e20
    filtered_df = df[valid_mask].copy()
    print(f"Removed {len(df) - len(filtered_df)} points. Remaining {len(filtered_df)} valid points.")
    return filtered_df

def write_tecplot_file(output_filename, header_lines, df_with_predictions):
    try:
        with open(output_filename, 'w') as f:
            f.writelines(header_lines)
            df_with_predictions.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n', float_format='%.8e')
        print(f"\n✅ Successfully saved predictions to '{output_filename}'.")
    except Exception as e:
        print(f"\n🔴 Error writing file '{output_filename}': {e}")


# --- 4. Main Training Script ---
print("\n--- Training Optimized FFN for Mach Prediction (V17.0) ---")

# --- Configuration ---
INPUT_FEATURES = ['X', 'Y']
OUTPUT_FEATURES = ['MA']
TRAIN_SPLIT_RATIO = 0.85
BATCH_SIZE = 4096
HIDDEN_UNITS = 512
NUM_HIDDEN_LAYERS = 8
LEARNING_RATE = 5e-4 # Slightly higher initial learning rate
EPOCHS = 2000
EARLY_STOP_PATIENCE = 500
LR_SCHEDULER_PATIENCE = 50

# --- Data Loading ---
data, header = load_structured_grid_data("structured_grid.DAT")
if data is not None and not data.empty:
    filtered_data = filter_valid_points(data)
    
    # Prepare data
    inputs_data = filtered_data[INPUT_FEATURES].values
    targets = filtered_data[OUTPUT_FEATURES].values
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    inputs_norm = input_scaler.fit_transform(inputs_data)
    targets_norm = target_scaler.fit_transform(targets)

    # --- بهبود یافته: استراتژی وزن‌دهی پیشرفته با مشتق اول و دوم ---
    print("Calculating advanced spatial derivatives for intelligent weighting...")
    mach_values = filtered_data['MA'].values
    coords = filtered_data[['X', 'Y']].values
    
    # Sort data points to approximate derivatives along a path
    sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
    mach_sorted = mach_values[sorted_indices]
    
    # First derivative (gradient)
    grad1_sorted = np.gradient(mach_sorted)
    # Second derivative (approximated as gradient of gradient)
    grad2_sorted = np.gradient(grad1_sorted)
    
    # Restore original order and get magnitude
    grad1_mag = np.zeros_like(mach_values)
    grad1_mag[sorted_indices] = np.abs(grad1_sorted)
    grad2_mag = np.zeros_like(mach_values)
    grad2_mag[sorted_indices] = np.abs(grad2_sorted)
    
    # Scale derivatives to [0, 1] range
    grad1_scaled = MinMaxScaler().fit_transform(grad1_mag.reshape(-1, 1)).flatten()
    grad2_scaled = MinMaxScaler().fit_transform(grad2_mag.reshape(-1, 1)).flatten()

    # Combine weights
    w1 = 100.0  # Weight for gradient (sharpness)
    w2 = 50.0   # Weight for second derivative (curvature)
    sample_weights = 1.0 + (w1 * grad1_scaled) + (w2 * grad2_scaled)
    print("Advanced sample weights calculated.")

    # Data Splitting
    n_samples = len(inputs_norm)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_index = int(n_samples * TRAIN_SPLIT_RATIO)
    train_indices, val_indices = indices[:split_index], indices[split_index:]

    x_train, x_val = inputs_norm[train_indices], inputs_norm[val_indices]
    y_train, y_val = targets_norm[train_indices], targets_norm[val_indices]
    sw_train, sw_val = sample_weights[train_indices], sample_weights[val_indices]
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

    # === Build and Compile Model ===
    model = build_optimized_ffn(
        spatial_input_dim=len(INPUT_FEATURES),
        output_dim=len(OUTPUT_FEATURES),
        hidden_units=HIDDEN_UNITS,
        num_hidden_layers=NUM_HIDDEN_LAYERS
    )
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=keras.losses.Huber(), weighted_metrics=['mae'])

    # --- بهبود یافته: استفاده از Callbacks برای آموزش هوشمند ---
    callbacks = [
        # توقف زود هنگام برای جلوگیری از اتلاف وقت
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # کاهش خودکار نرخ یادگیری در صورت عدم بهبود
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, # نرخ یادگیری را نصف می‌کند
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --- بهبود یافته: استفاده از model.fit برای سادگی و بهره‌گیری از Callbacks ---
    print("\n--- Starting Training with Advanced Callbacks ---")
    history = model.fit(
        x_train, y_train,
        sample_weight=sw_train,
        validation_data=(x_val, y_val, sw_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # === Generate Final Output & Analysis ===
    print("\n--- Generating Output File from Best Trained Model ---")
    all_inputs_norm = input_scaler.transform(data[INPUT_FEATURES].values)
    predictions_norm = model.predict(all_inputs_norm, batch_size=8192)
    predictions_original_scale = target_scaler.inverse_transform(predictions_norm)

    predicted_df = data.copy()
    predicted_df['MA_predicted'] = predictions_original_scale.flatten()
    large_value_mask = np.abs(data['MA'] - 1.0e+30) < 1e20
    predicted_df.loc[large_value_mask, 'MA_predicted'] = 1.0e+30

    output_df = predicted_df.copy()
    output_df['MA'] = output_df['MA_predicted']
    output_df = output_df.drop('MA_predicted', axis=1)

    output_filename = "structured_grid_NN_predicted_v17.0.dat"
    write_tecplot_file(output_filename, header, output_df)
    
    # --- Performance Analysis ---
    print("\n--- Performance Analysis ---")
    valid_mask = ~large_value_mask
    true_values = data.loc[valid_mask, 'MA'].values
    pred_values = predicted_df.loc[valid_mask, 'MA_predicted'].values
    mse = np.mean((true_values - pred_values) ** 2)
    mae = np.mean(np.abs(true_values - pred_values))
    mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-9))) * 100
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # (Optional Visualization Code can be added here)
else:
    print("Failed to load or process data. Exiting.")
