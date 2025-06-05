import os 

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def create_output_directory():
    output_dir = f"neural_network_evaluation"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir

def load_and_preprocess_data():
    """
    Load and preprocess the Alzheimer's disease dataset.
    This function performs the following steps:
    1. Loads the dataset from a CSV file.
    2. Drops unnecessary columns (PataintID and DoctorInCharge).
    3. Encodes categorical features using One-Hot Encoding.
    4. Standardizes clinical features using Z-score normalization.
    5. Scales lifestyle features using Min-Max scaling.
    
    Returns:
        X (DataFrame): Preprocessed feature set.
        y (Series): Target variable (Alzheimer's diagnosis).
    """
   
    # Load the dataset
    df = pd.read_csv("alzheimers_disease_data.csv")

    df.drop(columns=["PatientID"], inplace=True)
    df.drop(columns=["DoctorInCharge"], inplace=True)

    X = df.drop(columns=["Diagnosis"])  # Features
    y = df["Diagnosis"]  # (Alzheimer's diagnosis: 0 = No, 1 = Yes)

    # ✅ Convert y to ensure it's numeric (some datasets store 0/1 as strings)
    y = y.astype(int)  # Ensure y is integer type

    # One-Hot Encoding for Categorical Features
    categorical_features = ["Ethnicity", "EducationLevel"]
    encoder = OneHotEncoder(sparse_output=False)  # Drop first to avoid redundancy
    X_encoded = encoder.fit_transform(df[categorical_features])

    # Convert encoded data to DataFrame
    encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Drop original categorical columns & concatenate encoded ones
    X = X.drop(columns=categorical_features)
    X = pd.concat([X, encoded_df], axis=1)

    # Standardization (Z-score normalization) for Clinical Features
    standard_features = [
        "BMI", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
        "MMSE", "ADL", "FunctionalAssessment"
    ]
    scaler_standard = StandardScaler()
    X[standard_features] = scaler_standard.fit_transform(X[standard_features])

    # Min-Max Scaling for Lifestyle Features
    minmax_features = ["DietQuality", "SleepQuality", "PhysicalActivity", "AlcoholConsumption"]
    scaler_minmax = MinMaxScaler()
    X[minmax_features] = scaler_minmax.fit_transform(X[minmax_features])

    print("Data preprocessing complete! Preprocessed files saved.")
    #print the fist ten lines of the processed data
    #print("First ten lines of the processed data:")
    #print(X.head(10))

    return X, y

def build_neural_network(input_shape):
    """
    Build a neural network model for Alzheimer's disease diagnosis.
    
    Args:
        input_shape (int): The number of features in the input data.
        
    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = keras.Sequential([
    layers.Input(shape=(input_shape,)),

    layers.Dense(76, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  
    layers.Dense(38, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  
    layers.Dense(38, activation="elu", kernel_regularizer=regularizers.l2(0.001)), 
    
    layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)) 
])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.6),
                  loss="binary_crossentropy",
                  metrics=['accuracy', 
                           'binary_crossentropy', 
                           'mse',
                           'precision',
                           'recall',
                           'f1_score'
                           ])
    
    return model

def build_neural_network(input_shape, feature_mask=None):
    """
    Build a neural network model with optional feature masking.
    
    Args:
        input_shape (int): The number of features in the input data.
        feature_mask (array-like): Binary mask for features (1=use, 0=ignore)
        
    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        
        # Add masking layer if feature_mask is provided
        layers.Lambda(lambda x: x * feature_mask if feature_mask is not None else x, 
                     name='feature_masking') if feature_mask is not None else layers.Lambda(lambda x: x),
        
        layers.Dense(76, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  
        layers.Dense(38, activation="elu", kernel_regularizer=regularizers.l2(0.001)),  
        layers.Dense(38, activation="elu", kernel_regularizer=regularizers.l2(0.001)), 
        layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)) 
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.6),
                  loss="binary_crossentropy",
                  metrics=['accuracy', 'binary_crossentropy', 'mse', 'precision', 'recall', 'f1_score'])
    
    return model

def train_and_evaluate_model(X, y, trained_weights=None):
    """
    Train and evaluate the neural network model using Stratified K-Fold cross-validation.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target variable.
        trained_weights (optional): Pre-trained weights to use instead of training from scratch.
        selected_features (optional): List of selected feature indices.
        
    Returns:
        results (dict): Dictionary containing evaluation metrics and trained weights.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Early Stopping based on validation loss 
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    input_shape = X.shape[1]  

    # Initialize dictionaries to store results
    results = {
        'val_accuracy': [],
        'val_loss': [],
        'val_binary_crossentropy': [],
        'val_mse': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': [],
        'epochs_trained': [],
        'train_accuracy': [],
        'generalization_gap': [],  
        'all_histories': [],
        'trained_weights': None,  # To store the best model weights
        'model': None  # To store the best model
    }

    fold_accuracies = []
    fold_losses = []
    fold_bce = []
    fold_mse = []
    fold_precission = []
    fold_recall = []
    fold_f1_score = []
    fold_epochs = []
    all_fold_histories = []
    val_scores = []
    fold_train_acc = []

    trained_model = None  # To store the best model
    best_val_accuracy = 0

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"  TRAINING FOLD {fold + 1}/5")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = build_neural_network(input_shape)
        
        # If pre-trained weights are provided, use them
        if trained_weights is not None:
            try:
                model.set_weights(trained_weights)
                print("  Using pre-trained weights")
            except Exception as e:
                print(f"  Warning: Could not load pre-trained weights: {e}")
                print("  Training from scratch instead")
        
        # Calculate class weights
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)
        class_weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            verbose=0,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[early_stopping]
        )

        all_fold_histories.append(history.history)
        val_scores = model.evaluate(X_test, y_test, verbose=0)
        train_scores = model.evaluate(X_train, y_train, verbose=0)
        val_loss = val_scores[0]  # loss (from loss function - BCE)
        val_accuracy = val_scores[1]  
        val_bce = val_scores[2]  
        val_mse = val_scores[3]  
        val_precission = val_scores[4]  
        val_recall = val_scores[5]
        val_f1_score = val_scores[6]
        epochs = len(history.history['loss'])

        # Store individual fold results
        fold_accuracies.append(val_accuracy)
        fold_losses.append(val_loss)
        fold_bce.append(val_bce)
        fold_mse.append(val_mse)
        fold_precission.append(val_precission)
        fold_recall.append(val_recall)
        fold_f1_score.append(val_f1_score)
        fold_epochs.append(epochs)
        fold_train_acc.append(train_scores[1])

        # Keep track of the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trained_model = model

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate average scores across folds
    avg_accuracy = np.mean(fold_accuracies)
    avg_loss = np.mean(fold_losses)
    avg_bce = np.mean(fold_bce)
    avg_mse = np.mean(fold_mse)
    avg_precission = np.mean(fold_precission)
    avg_recall = np.mean(fold_recall)
    avg_f1_score = np.mean(fold_f1_score)
    avg_epochs = np.mean(fold_epochs)
    avg_train_acc = np.mean(fold_train_acc)

    # Store results
    results['val_accuracy'].append(avg_accuracy)
    results['val_loss'].append(avg_loss)
    results['val_binary_crossentropy'].append(avg_bce)
    results['val_mse'].append(avg_mse)
    results['val_precision'].append(avg_precission)
    results['val_recall'].append(avg_recall)
    results['val_f1_score'].append(avg_f1_score)
    results['epochs_trained'].append(avg_epochs)
    results['all_histories'].append(all_fold_histories)
    results['train_accuracy'].append(avg_train_acc)

    generalization_gap = avg_train_acc - avg_accuracy
    results['generalization_gap'].append(generalization_gap)

    if trained_model is not None:
        results['trained_weights'] = trained_model.get_weights()
        results['model'] = trained_model

    print(f"RESULTS: Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, Avg Epochs: {avg_epochs:.1f}, Precision: {avg_precission:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}, Generalization Gap: {generalization_gap:.4f}")

    output_dir = create_output_directory()

    save_results_to_csv(results, output_dir)
    plot_convergence_graphs(results, y_pred_classes, y_test, output_dir)

    # Add trained weights to results
    if trained_model is not None:
        results['trained_weights'] = trained_model.get_weights()
        results['model'] = trained_model

    return results

def train_and_evaluate_model(X, y, trained_weights=None, selected_features=None):
    """
    Train and evaluate the neural network model using masking for feature selection.
    
    Args:
        X (DataFrame): Feature set (always full feature set).
        y (Series): Target variable.
        trained_weights (optional): Pre-trained weights to use.
        selected_features (optional): List of selected feature indices for masking.
        
    Returns:
        results (dict): Dictionary containing evaluation metrics and trained weights.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create feature mask
    feature_mask = None
    if selected_features is not None:
        feature_mask = np.zeros(X.shape[1], dtype=np.float32)
        feature_mask[selected_features] = 1.0
        feature_mask = tf.constant(feature_mask)
        print(f"  Using feature mask: {len(selected_features)}/{X.shape[1]} features selected")
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    input_shape = X.shape[1]  # Always use full input shape

    # Initialize results dictionary
    results = {
        'val_accuracy': [],
        'val_loss': [],
        'val_binary_crossentropy': [],
        'val_mse': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': [],
        'epochs_trained': [],
        'train_accuracy': [],
        'generalization_gap': [],  
        'all_histories': [],
        'trained_weights': None,
        'model': None
    }

    fold_accuracies = []
    fold_losses = []
    fold_bce = []
    fold_mse = []
    fold_precission = []
    fold_recall = []
    fold_f1_score = []
    fold_epochs = []
    all_fold_histories = []
    fold_train_acc = []

    trained_model = None
    best_val_accuracy = 0

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"  TRAINING FOLD {fold + 1}/5")
        
        # Use full feature set but apply masking
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build model with masking
        model = build_neural_network(input_shape, feature_mask)
        
        # Load pre-trained weights (now compatible since architecture is identical)
        if trained_weights is not None:
            try:
                model.set_weights(trained_weights)
                print("  Successfully loaded pre-trained weights with masking")
            except Exception as e:
                print(f"  Warning: Could not load pre-trained weights: {e}")
                print("  Training from scratch instead")
        
        # Calculate class weights
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)
        class_weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            verbose=0,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[early_stopping]
        )

        all_fold_histories.append(history.history)
        val_scores = model.evaluate(X_test, y_test, verbose=0)
        train_scores = model.evaluate(X_train, y_train, verbose=0)
        
        val_accuracy = val_scores[1]
        
        # Store fold results
        fold_accuracies.append(val_scores[1])
        fold_losses.append(val_scores[0])
        fold_bce.append(val_scores[2])
        fold_mse.append(val_scores[3])
        fold_precission.append(val_scores[4])
        fold_recall.append(val_scores[5])
        fold_f1_score.append(val_scores[6])
        fold_epochs.append(len(history.history['loss']))
        fold_train_acc.append(train_scores[1])

        # Keep track of best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trained_model = model

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate average scores across folds
    avg_accuracy = np.mean(fold_accuracies)
    avg_loss = np.mean(fold_losses)
    avg_bce = np.mean(fold_bce)
    avg_mse = np.mean(fold_mse)
    avg_precission = np.mean(fold_precission)
    avg_recall = np.mean(fold_recall)
    avg_f1_score = np.mean(fold_f1_score)
    avg_epochs = np.mean(fold_epochs)
    avg_train_acc = np.mean(fold_train_acc)

    # Store results
    results['val_accuracy'].append(avg_accuracy)
    results['val_loss'].append(avg_loss)
    results['val_binary_crossentropy'].append(avg_bce)
    results['val_mse'].append(avg_mse)
    results['val_precision'].append(avg_precission)
    results['val_recall'].append(avg_recall)
    results['val_f1_score'].append(avg_f1_score)
    results['epochs_trained'].append(avg_epochs)
    results['all_histories'].append(all_fold_histories)
    results['train_accuracy'].append(avg_train_acc)

    generalization_gap = avg_train_acc - avg_accuracy
    results['generalization_gap'].append(generalization_gap)

    if trained_model is not None:
        results['trained_weights'] = trained_model.get_weights()
        results['model'] = trained_model

    print(f"RESULTS: Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, Avg Epochs: {avg_epochs:.1f}, Precision: {avg_precission:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}, Generalization Gap: {generalization_gap:.4f}")

    output_dir = create_output_directory()

    save_results_to_csv(results, output_dir)
    plot_convergence_graphs(results, y_pred_classes, y_test, output_dir)

    # Add trained weights to results
    if trained_model is not None:
        results['trained_weights'] = trained_model.get_weights()
        results['model'] = trained_model

    return results

def save_results_to_csv(results, output_dir):
    output_dir = f"neural_network_evaluation"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Create DataFrame for summary results (averaged across folds)
    summary_df = pd.DataFrame({
        'val_accuracy': results['val_accuracy'],
        'val_loss': results['val_loss'],
        'val_binary_crossentropy': results['val_binary_crossentropy'],
        'val_mse': results['val_mse'],
        'val_precision': results['val_precision'],
        'val_recall': results['val_recall'],
        'val_f1_score': results['val_f1_score'],
        'epochs_trained': results['epochs_trained'],
        'train_accuracy': results['train_accuracy'],
        'generalization_gap': results['generalization_gap']
    })

    summary_filename = os.path.join(output_dir, "neural_network_evaluation.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to {summary_filename}")

    return summary_filename

# Function to plot convergence graphs and confusion matrix
def plot_convergence_graphs(results, y_pred_classes, y_test, output_dir):
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Use the last fold's history for plotting
    if results['all_histories'] and results['all_histories'][-1]:
        last_history = results['all_histories'][-1][0]  # Get the first history from the last fold
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(last_history['accuracy'])
        plt.plot(last_history['val_accuracy'])
        plt.title(f'Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(last_history['loss'])
        plt.plot(last_history['val_loss'])
        plt.title(f'Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_history.png')
    
    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not patient', 'Patient'],
                yticklabels=['Not patient', 'Patient'])
    plt.title(f'Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')

    return

# if __name__ == "__main__":
#     X, y = load_and_preprocess_data()
#     # Save the first 10 rows of preprocessed data to CSV
#     X.head(10).to_csv("preprocessed_data_sample.csv", index=False)
#     print(f"Saved first 10 rows of preprocessed data to preprocessed_data_sample.csv")
#     results = train_and_evaluate_model(X, y)
#     print("Neural network evaluation completed successfully!")

