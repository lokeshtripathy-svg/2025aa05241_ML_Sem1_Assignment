"""
ML Classification Models Training Script
This script trains 6 different classification models and saves them for use in the Streamlit app.
Dataset: Heart Disease Dataset from UCI (or any dataset with 12+ features, 500+ instances)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of instances: {df.shape[0]}")
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Encode target if categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        label_encoders['target'] = le_target
    
    return X, y, label_encoders


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    
    # For AUC calculation
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For multi-class, use weighted average
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }
    
    print(f"\n{model_name} Metrics:")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"  {key}: {value}")
    
    return metrics, confusion_matrix(y_test, y_pred)


def train_all_models(X_train, X_test, y_train, y_test, save_dir='./'):
    """Train all 6 classification models."""
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    confusion_matrices = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("=" * 60)
    print("Training and Evaluating Classification Models")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that benefit from scaling
        if name in ['Logistic Regression', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            metrics, cm = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            model.fit(X_train, y_train)
            metrics, cm = evaluate_model(model, X_test, y_test, name)
        
        results.append(metrics)
        confusion_matrices[name] = cm
        
        # Save model
        model_filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
        with open(os.path.join(save_dir, model_filename), 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_filename}")
    
    return results, confusion_matrices


def main():
    """Main function to run the training pipeline."""
    
    # Create sample dataset if not exists (using Heart Disease data structure)
    # In practice, you would download from Kaggle/UCI
    
    # For demonstration, we'll create a synthetic dataset
    # Replace this with your actual dataset loading
    
    print("=" * 60)
    print("ML Classification Models Training Pipeline")
    print("=" * 60)
    
    # Check if dataset exists, otherwise create sample data
    data_path = 'data.csv'
    
    if not os.path.exists(data_path):
        print("\nNo dataset found. Creating sample dataset for demonstration...")
        print("Please replace with your actual dataset from Kaggle/UCI")
        
        # Create synthetic dataset with 15 features and 1000 instances
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        # Generate features
        data = {}
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        for i, name in enumerate(feature_names):
            if i < 10:
                data[name] = np.random.randn(n_samples)
            else:
                data[name] = np.random.randint(0, 5, n_samples)
        
        # Generate target (binary classification)
        # Target depends on some features
        prob = 1 / (1 + np.exp(-(
            0.5 * data['feature_1'] + 
            0.3 * data['feature_2'] - 
            0.4 * data['feature_3'] +
            0.2 * data['feature_11'] +
            np.random.randn(n_samples) * 0.5
        )))
        data['target'] = (prob > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print(f"Sample dataset created: {data_path}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, y, label_encoders = load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Save feature names
    feature_names = list(X.columns)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save test data for Streamlit app
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)
    print("\nTest data saved: test_data.csv")
    
    # Train all models
    results, confusion_matrices = train_all_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    print("\n" + "=" * 60)
    print("Model Comparison Table")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Save confusion matrices
    with open('confusion_matrices.pkl', 'wb') as f:
        pickle.dump(confusion_matrices, f)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nSaved files:")
    print("  - scaler.pkl")
    print("  - logistic_regression.pkl")
    print("  - decision_tree.pkl")
    print("  - k_nearest_neighbors.pkl")
    print("  - naive_bayes.pkl")
    print("  - random_forest.pkl")
    print("  - xgboost.pkl")
    print("  - model_results.csv")
    print("  - confusion_matrices.pkl")
    print("  - feature_names.pkl")
    print("  - test_data.csv")


if __name__ == "__main__":
    main()
