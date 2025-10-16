import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
import itertools

# ---------------------- Configuration Parameters (Modify Before Use) ----------------------
# Input/Output Paths (Replace with your actual paths, avoid Chinese characters in paths)
IMAGE_FOLDER = "./dataset/images"  # Folder storing images (JPG format)
LABEL_FILE = "./dataset/labels.xlsx"  # Excel file with image names and regression labels
SAVE_ROOT = "./results"  # Root folder to save models, results and visualizations

# Basic Training Parameters
BATCH_SIZE = 16
EPOCHS = 20
TARGET_SIZE = (224, 224)  # Image resize dimension (height, width)
TEST_SIZE = 0.1  # Proportion of test set (0-1)
RANDOM_STATE = 42  # For reproducibility

# Cross-Validation Parameters
KFOLD_SPLITS = 5  # Number of folds for K-fold cross-validation

# Hyperparameter Search Space (for RandomizedSearchCV)
PARAM_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'conv_filters': [[32, 64, 128], [16, 32, 64], [64, 128, 256]],  # Filters for each Conv layer
    'dense_units': [64, 128, 256],  # Units in fully connected layer
    'dropout_rate': [0.1, 0.2, 0.4]  # Dropout rate for overfitting prevention
}

# Hyperparameter Search Settings
SEARCH_ITERATIONS = 10  # Number of random combinations to test

# Visualization Parameters
VISUALIZATION_SAMPLES = 5  # Number of test samples for visualization
LAYER_VISUALIZATION = [0, 2, 4]  # Indices of Conv layers to visualize (activation maps)


# -------------------------------------------------------
def data_generator(df, image_folder, target_size=(224, 224), batch_size=16, shuffle=True):
    """
    Generic data generator for image regression tasks.
    Dynamically reads images and corresponding labels (supports multiple regression metrics).

    Args:
        df: DataFrame with 'imagename' column and label columns
        image_folder: Path to folder containing images
        target_size: Tuple (height, width) for image resizing
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples per epoch

    Yields:
        batch_images: Numpy array of preprocessed images (batch_size, H, W, 3)
        batch_labels: Numpy array of corresponding labels (batch_size, num_metrics)
    """
    indices = df.index.tolist()
    # Get label columns (exclude 'imagename' which stores image filenames)
    metric_cols = [col for col in df.columns if col != 'imagename']

    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                row = df.loc[idx]
                image_name = f"{row['imagename']}.jpg"  # Assume images are in JPG format
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)

                # Skip invalid images (corrupted or non-existent)
                if image is None:
                    print(f"Skipping invalid image: {image_path}")
                    continue
                # Image preprocessing: resize + normalize to [0,1]
                image = cv2.resize(image, target_size)
                image = image.astype(np.float32) / 255.0
                batch_images.append(image)

                # Extract labels (handle multiple metrics dynamically)
                labels = [row[col] for col in metric_cols]
                # Skip samples with NaN/Inf labels
                if np.isnan(labels).any() or np.isinf(labels).any():
                    print(f"Skipping invalid labels: {labels} (Sample: {row['imagename']})")
                    continue
                batch_labels.append(labels)

            # Ensure batch is not empty before yielding
            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)


def check_data_validity(df, image_folder):
    """
    Check validity of dataset (image existence + label validity) and return valid samples.

    Args:
        df: Raw DataFrame with 'imagename' and label columns
        image_folder: Path to image folder

    Returns:
        DataFrame containing only valid samples
    """
    metric_cols = [col for col in df.columns if col != 'imagename']
    valid_indices = []

    for idx, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{row['imagename']}.jpg")
        # Check if image exists and is readable
        if not os.path.exists(image_path) or cv2.imread(image_path) is None:
            print(f"Invalid image: {image_path}")
            continue

        # Check if labels are valid (no NaN/Inf)
        labels = [row[col] for col in metric_cols]
        if np.isnan(labels).any() or np.isinf(labels).any():
            print(f"Invalid labels: {labels} (Sample: {row['imagename']})")
            continue

        valid_indices.append(idx)

    print(f"Valid samples: {len(valid_indices)}/{len(df)}")
    return df.loc[valid_indices]


def build_model(input_shape, num_metrics, learning_rate=0.001,
                conv_filters=[32, 64, 128], dense_units=128, dropout_rate=0.2):
    """
    Build a CNN model for multi-metric regression.
    Used as a wrapper for KerasRegressor (compatible with scikit-learn hyperparameter search).

    Args:
        input_shape: Tuple (H, W, 3) for input image dimension
        num_metrics: Number of regression targets (output dimension)
        learning_rate: Learning rate for Adam optimizer
        conv_filters: List of filters for each Conv2D layer
        dense_units: Number of units in the fully connected layer
        dropout_rate: Dropout rate after the fully connected layer

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()

    # Add convolutional layers (dynamic number based on conv_filters)
    for i, filters in enumerate(conv_filters):
        if i == 0:
            # First Conv layer needs input_shape
            model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape, name=f'conv_{i}'))
        else:
            model.add(Conv2D(filters, (3, 3), activation='relu', name=f'conv_{i}'))
        model.add(MaxPooling2D((2, 2), name=f'pool_{i}'))  # Max pooling after each Conv layer

    # Add fully connected layers for regression
    model.add(Flatten(name='flatten'))  # Convert 2D features to 1D vector
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout'))  # Prevent overfitting
    model.add(Dense(num_metrics, name='output'))  # Output layer (no activation for regression)

    # Compile model (MSE loss for regression tasks)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


def evaluate_model(model, test_gen, validation_steps, metric_cols):
    """
    Evaluate model performance on test set using RMSE, R², and MAE.

    Args:
        model: Trained Keras model
        test_gen: Data generator for test set
        validation_steps: Number of steps to iterate through test_gen
        metric_cols: List of regression metric names

    Returns:
        results: Dictionary of metrics (RMSE/R²/MAE) for each regression target
        test_labels: True labels of test set (numpy array)
        test_preds: Predicted labels of test set (numpy array)
    """
    test_preds = []
    test_labels = []

    # Collect predictions and true labels from test generator
    for _ in range(validation_steps):
        x, y = next(test_gen)
        preds = model.predict(x)
        test_preds.extend(preds)
        test_labels.extend(y)

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # Calculate metrics for each regression target
    results = {}
    for i, metric in enumerate(metric_cols):
        rmse = np.sqrt(mean_squared_error(test_labels[:, i], test_preds[:, i]))
        r2 = r2_score(test_labels[:, i], test_preds[:, i])
        mae = mean_absolute_error(test_labels[:, i], test_preds[:, i])
        results[metric] = {"RMSE": rmse, "R²": r2, "MAE": mae}

    return results, test_labels, test_preds


def visualize_intermediate_layers(model, image, layer_indices, save_path=None):
    """
    Visualize activation maps of intermediate convolutional layers.

    Args:
        model: Trained Keras model
        image: Preprocessed image (H, W, 3) for visualization
        layer_indices: List of Conv layer indices to visualize
        save_path: Path to save the visualization (if None, only display)
    """
    # Create a model to extract outputs of target layers
    layer_outputs = [model.layers[i].output for i in layer_indices]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))  # Add batch dimension

    # Plot activation maps in a grid
    images_per_row = 16
    plt.figure(figsize=(15, 10))

    for layer_idx, layer_activation in zip(layer_indices, activations):
        n_features = layer_activation.shape[-1]  # Number of feature maps (channels)
        size = layer_activation.shape[1]  # Size of each feature map (H=W)

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # Fill grid with normalized feature maps
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Normalize to [0, 255] for better visualization
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)  # Avoid division by zero
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        # Plot single layer's activation maps
        plt.subplot(len(layer_indices), 1, layer_indices.index(layer_idx) + 1)
        plt.title(f'Layer {layer_idx}: {model.layers[layer_idx].name}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Intermediate layer visualization saved to: {save_path}")
    plt.close()


def grad_cam(model, img_array, metric_index, layer_name=None):
    """
    Implement Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize feature importance.
    Reference: https://arxiv.org/pdf/1610.02391.pdf

    Args:
        model: Trained Keras model
        img_array: Preprocessed image (1, H, W, 3) with batch dimension
        metric_index: Index of the regression target to analyze
        layer_name: Name of the Conv layer to use (default: last Conv layer)

    Returns:
        heatmap: Numpy array of Grad-CAM heatmap (H, W)
    """
    # Use last convolutional layer if layer_name is not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                layer_name = layer.name
                break

    # Create a model to get gradients of target metric w.r.t. Conv layer outputs
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Calculate gradients using automatic differentiation
    with K.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, metric_index]  # Loss = predicted value of target metric

    # Compute gradients of loss w.r.t. Conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling (GAP) on gradients to get channel importance weights
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Weight Conv layer outputs by channel importance
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., np.newaxis]  # (H, W, C) @ (C, 1) = (H, W, 1)
    heatmap = K.squeeze(heatmap, axis=-1)  # (H, W)

    # Keep only positive contributions (ReLU activation)
    heatmap = K.maximum(heatmap, 0)

    # Normalize heatmap to [0, 1]
    heatmap /= K.max(heatmap)
    return heatmap.numpy()


def visualize_feature_importance(model, img, img_array, metric_index, metric_name, save_path=None):
    """
    Visualize feature importance for a specific regression metric using Grad-CAM.
    Combines original image, heatmap, and superimposed result.

    Args:
        model: Trained Keras model
        img: Original image (BGR format, H, W, 3) without normalization
        img_array: Preprocessed image (1, H, W, 3) with batch dimension
        metric_index: Index of the regression target to analyze
        metric_name: Name of the regression target (for plot title)
        save_path: Path to save the visualization (if None, only display)
    """
    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, img_array, metric_index)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB with colormap (Jet)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[..., :3]  # (H, W, 3)

    # Superimpose heatmap on original image (adjust transparency with 0.4)
    superimposed_img = heatmap * 0.4 + img / 255.0  # Normalize img to [0,1] first
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img) * 255)

    # Plot 3 subplots: original image, heatmap, superimposed result
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title(f'Grad-CAM Heatmap\n({metric_name})')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Superimposed Result\n({metric_name})')
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved to: {save_path}")
    plt.close()


def main():
    # Create result directories (auto-create if not exist)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    vis_dir = os.path.join(SAVE_ROOT, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    layer_vis_dir = os.path.join(vis_dir, "intermediate_layers")
    os.makedirs(layer_vis_dir, exist_ok=True)
    feature_vis_dir = os.path.join(vis_dir, "feature_importance")
    os.makedirs(feature_vis_dir, exist_ok=True)

    # 1. Load labels and auto-detect regression metrics
    labels_df = pd.read_excel(io=LABEL_FILE, engine='openpyxl')
    labels_df['imagename'] = labels_df['imagename'].astype(str)  # Ensure image names are strings
    metric_cols = [col for col in labels_df.columns if col != 'imagename']
    num_metrics = len(metric_cols)
    print(f"Auto-detected regression metrics: {metric_cols} (Total: {num_metrics})")

    # 2. Validate data and split into train/val/test sets
    valid_df = check_data_validity(labels_df, IMAGE_FOLDER)
    train_val_df, test_df = train_test_split(
        valid_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Prepare test set generator (no shuffle for consistent evaluation)
    test_gen = data_generator(
        test_df,
        IMAGE_FOLDER,
        TARGET_SIZE,
        BATCH_SIZE,
        shuffle=False
    )
    validation_steps = max(1, len(test_df) // BATCH_SIZE)  # Avoid zero steps

    # 3. Hyperparameter search using RandomizedSearchCV
    print("\nStarting hyperparameter search...")
    input_shape = (*TARGET_SIZE, 3)  # (H, W, 3) for RGB images

    # Wrap Keras model for scikit-learn compatibility
    model_wrapper = KerasRegressor(
        build_fn=build_model,
        input_shape=input_shape,
        num_metrics=num_metrics,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model_wrapper,
        param_distributions=PARAM_GRID,
        n_iter=SEARCH_ITERATIONS,
        cv=KFOLD_SPLITS,
        scoring='neg_mean_squared_error',  # Higher = better (negative MSE)
        random_state=RANDOM_STATE,
        verbose=1
    )

    # Prepare data for hyperparameter search (load all at once for small datasets)
    def prepare_search_data(df):
        images, labels = [], []
        metric_cols = [col for col in df.columns if col != 'imagename']
        for _, row in df.iterrows():
            img_path = os.path.join(IMAGE_FOLDER, f"{row['imagename']}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, TARGET_SIZE) / 255.0  # Preprocess
                images.append(img)
                labels.append([row[col] for col in metric_cols])
        return np.array(images), np.array(labels)

    X_search, y_search = prepare_search_data(train_val_df)

    # Run hyperparameter search
    random_search.fit(X_search, y_search)

    # Print and save search results
    print(f"Best hyperparameters: {random_search.best_params_}")
    print(f"Best cross-validation score (MSE): {-random_search.best_score_:.4f}")  # Convert to positive MSE

    # Save hyperparameter search results to Excel
    search_results_df = pd.DataFrame(random_search.cv_results_)
    search_results_df.to_excel(os.path.join(SAVE_ROOT, "hyperparameter_search_results.xlsx"), index=False)

    # 4. K-fold cross-validation with best hyperparameters
    print("\nStarting K-fold cross-validation...")
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {metric: [] for metric in metric_cols}  # Store results per metric

    fold = 1
    for train_idx, val_idx in kf.split(train_val_df):
        print(f"\n----- Fold {fold}/{KFOLD_SPLITS} -----")

        # Split into fold-specific train/val sets
        fold_train_df = train_val_df.iloc[train_idx]
        fold_val_df = train_val_df.iloc[val_idx]

        # Create generators for current fold
        fold_train_gen = data_generator(
            fold_train_df,
            IMAGE_FOLDER,
            TARGET_SIZE,
            BATCH_SIZE,
            shuffle=True
        )
        fold_val_gen = data_generator(
            fold_val_df,
            IMAGE_FOLDER,
            TARGET_SIZE,
            BATCH_SIZE,
            shuffle=False
        )

        # Calculate steps per epoch (avoid zero)
        steps_per_epoch = max(1, len(fold_train_df) // BATCH_SIZE)
        fold_val_steps = max(1, len(fold_val_df) // BATCH_SIZE)

        # Build model with best hyperparameters
        fold_model = build_model(
            input_shape=input_shape,
            num_metrics=num_metrics,
            **random_search.best_params_
        )

        # Train model on current fold
        fold_model.fit(
            fold_train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=fold_val_gen,
            validation_steps=fold_val_steps,
            verbose=1
        )

        # Evaluate current fold model
        fold_val_results, _, _ = evaluate_model(fold_model, fold_val_gen, fold_val_steps, metric_cols)

        # Store results for current fold
        for metric, metrics_dict in fold_val_results.items():
            cv_results[metric].append(metrics_dict)

        fold += 1

    # Calculate and print average cross-validation results
    print("\nAverage cross-validation results:")
    cv_avg_results = {}
    for metric, results_list in cv_results.items():
        avg_rmse = np.mean([r['RMSE'] for r in results_list])
        avg_r2 = np.mean([r['R²'] for r in results_list])
        avg_mae = np.mean([r['MAE'] for r in results_list])
        cv_avg_results[metric] = {"Average RMSE": avg_rmse, "Average R²": avg_r2, "Average MAE": avg_mae}
        print(f"{metric}:")
        print(f"  Average RMSE: {avg_rmse:.4f}")
        print(f"  Average R²: {avg_r2:.4f}")
        print(f"  Average MAE: {avg_mae:.4f}\n")

    # Save average cross-validation results to Excel
    cv_results_df = pd.DataFrame()
    for metric, avg_dict in cv_avg_results.items():
        for key, value in avg_dict.items():
            cv_results_df.at[metric, key] = value
    cv_results_df.to_excel(os.path.join(SAVE_ROOT, "cross_validation_results.xlsx"))

    # 5. Train final model on full train/val set with best hyperparameters
    print("\nTraining final model on full train/validation set...")
    final_model = build_model(
        input_shape=input_shape,
        num_metrics=num_metrics,
        **random_search.best_params_
    )

    # Prepare generator for full train/val set
    final_train_gen = data_generator(
        train_val_df,
        IMAGE_FOLDER,
        TARGET_SIZE,
        BATCH_SIZE,
        shuffle=True
    )
    final_steps_per_epoch = max(1, len(train_val_df) // BATCH_SIZE)

    # Train final model
    final_model.fit(
        final_train_gen,
        steps_per_epoch=final_steps_per_epoch,
        epochs=EPOCHS,
        validation_data=test_gen,
        validation_steps=validation_steps
    )

    # 6. Save final model (Keras format, compatible with TensorFlow >=2.4)
    model_save_path = os.path.join(SAVE_ROOT, f"final_model_{num_metrics}_metrics.keras")
    final_model.save(model_save_path)
    print(f"Final model saved to: {model_save_path}")

    # 7. Evaluate final model on test set
    print("\nFinal model evaluation on test set:")
    test_results, test_labels, test_preds = evaluate_model(
        final_model,
        test_gen,
        validation_steps,
        metric_cols
    )

    # Print test set results
    for metric, metrics_dict in test_results.items():
        print(f"{metric}:")
        print(f"  RMSE: {metrics_dict['RMSE']:.4f}")
        print(f"  R²: {metrics_dict['R²']:.4f}")
        print(f"  MAE: {metrics_dict['MAE']:.4f}\n")

    # 8. Generate full predictions (train + test sets)
    def get_full_predictions(df):
        """Helper function to get predictions for a given DataFrame"""
        images, labels, img_paths = [], [], []
        metric_cols = [col for col in df.columns if col != 'imagename']
        for _, row in df.iterrows():
            img_path = os.path.join(IMAGE_FOLDER, f"{row['imagename']}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img_normalized = cv2.resize(img, TARGET_SIZE) / 255.0
                images.append(img_normalized)
                labels.append([row[col] for col in metric_cols])
                img_paths.append(img_path)
        images = np.array(images)
        labels = np.array(labels)
        preds = final_model.predict(images)
        return images, labels, preds, img_paths

    # Get predictions for train and test sets
    train_imgs, train_labels, train_preds, _ = get_full_predictions(train_val_df)
    test_imgs, test_labels, test_preds, test_img_paths = get_full_predictions(test_df)

    # 9. Visualize intermediate layer activations (on test samples)
    print("\nGenerating intermediate layer visualizations...")
    # Select random valid test samples (avoid duplicate indices)
    sample_indices = np.random.choice(
        len(test_imgs),
        min(VISUALIZATION_SAMPLES, len(test_imgs)),
        replace=False
    )

    for i, idx in enumerate(sample_indices):
        img = test_imgs[idx]
        img_name = os.path.basename(test_img_paths[idx]).split('.')[0]  # Get image name without extension

        # Save activation map visualization
        visualize_intermediate_layers(
            final_model,
            img,
            LAYER_VISUALIZATION,
            save_path=os.path.join(layer_vis_dir, f"sample_{img_name}_activations.png")
        )

    # 10. Visualize feature importance (Grad-CAM) for each metric
    print("\nGenerating feature importance visualizations...")
    for i, idx in enumerate(sample_indices):
        img_array = np.expand_dims(test_imgs[idx], axis=0)  # Add batch dimension
        img_original = cv2.imread(test_img_paths[idx])  # Load original image (no normalization)
        img_original = cv2.resize(img_original, TARGET_SIZE)  # Match model input size
        img_name = os.path.basename(test_img_paths[idx]).split('.')[0]

        # Generate Grad-CAM for each regression metric
        for metric_idx, metric_name in enumerate(metric_cols):
            visualize_feature_importance(
                final_model,
                img_original,
                img_array,
                metric_idx,
                metric_name,
                save_path=os.path.join(feature_vis_dir, f"sample_{img_name}_{metric_name}_gradcam.png")
            )

    # 11. Visualize true vs predicted values (scatter plot)
    plt.figure(figsize=(14, 6))

    # Train set: True vs Predicted
    plt.subplot(1, 2, 1)
    for i, metric in enumerate(metric_cols):
        plt.scatter(train_labels[:, i], train_preds[:, i], label=metric, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Train Set: True vs Predicted Values")
    plt.legend()
    plt.grid(alpha=0.3)

    # Test set: True vs Predicted
    plt.subplot(1, 2, 2)
    for i, metric in enumerate(metric_cols):
        plt.scatter(test_labels[:, i], test_preds[:, i], label=metric, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Test Set: True vs Predicted Values")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "true_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("True vs Predicted visualization saved to: ./results/true_vs_predicted.png")

    # 12. Save full predictions to Excel (train + test sets)
    def create_results_df(df, true_labels, preds):
        """Create DataFrame with image names, true labels, and predicted labels"""
        data = {"imagename": df['imagename'].values}
        for i, metric in enumerate(metric_cols):
            data[f"True_{metric}"] = true_labels[:, i]
            data[f"Pred_{metric}"] = preds[:, i]
        return pd.DataFrame(data)

    # Create results DataFrames for train and test sets
    train_results_df = create_results_df(train_val_df, train_labels, train_preds)
    test_results_df = create_results_df(test_df, test_labels, test_preds)

    # Save to Excel with multiple sheets
    excel_save_path = os.path.join(SAVE_ROOT, f"predictions_{num_metrics}_metrics.xlsx")
    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
        train_results_df.to_excel(writer, sheet_name="Train_Set", index=False)
        test_results_df.to_excel(writer, sheet_name="Test_Set", index=False)
    print(f"Full predictions saved to: {excel_save_path}")
    print(f"All visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()