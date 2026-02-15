"""
ML Classification Models - Streamlit Web Application
This app allows users to:
1. Upload a dataset (CSV)
2. Select a classification model
3. View evaluation metrics
4. Display confusion matrix and classification report
"""

import streamlit as st
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 ML Classification Models Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare 6 Different Classification Algorithms</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model selection
MODEL_OPTIONS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

selected_model = st.sidebar.selectbox(
    "🔍 Select Classification Model",
    list(MODEL_OPTIONS.keys()),
    help="Choose a model to train and evaluate"
)

# File upload
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload your dataset. The last column should be the target variable."
)

# Test size slider
test_size = st.sidebar.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Proportion of data to use for testing"
)


def load_and_preprocess_data(df):
    """Load and preprocess the dataset."""
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target (last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Store original feature names
    feature_names = list(X.columns)
    
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
    
    return X, y, feature_names, label_encoders


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
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }
    
    return metrics, y_pred, confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models and return comparison."""
    results = []
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in MODEL_OPTIONS.items():
        # Use scaled data for models that benefit from scaling
        if name in ['Logistic Regression', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            metrics, _, _ = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            model.fit(X_train, y_train)
            metrics, _, _ = evaluate_model(model, X_test, y_test, name)
        
        metrics['Model'] = name
        results.append(metrics)
    
    return pd.DataFrame(results)


# Main content
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1] - 1)
    with col3:
        st.metric("Target Classes", df.iloc[:, -1].nunique())
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show data preview
    with st.expander("📋 View Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Preprocess data
    X, y, feature_names, label_encoders = load_and_preprocess_data(df.copy())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    st.info(f"📌 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    # Scale data for certain models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Single Model Evaluation", "📈 Model Comparison", "📊 Feature Analysis"])
    
    with tab1:
        st.header(f"🎯 {selected_model} - Model Evaluation")
        
        # Train selected model
        model = MODEL_OPTIONS[selected_model]
        
        if selected_model in ['Logistic Regression', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            metrics, y_pred, cm = evaluate_model(model, X_test_scaled, y_test, selected_model)
        else:
            model.fit(X_train, y_train)
            metrics, y_pred, cm = evaluate_model(model, X_test, y_test, selected_model)
        
        # Display metrics
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("📈 AUC Score", f"{metrics['AUC']:.4f}")
        
        with col2:
            st.metric("🎯 Precision", f"{metrics['Precision']:.4f}")
            st.metric("📊 Recall", f"{metrics['Recall']:.4f}")
        
        with col3:
            st.metric("⚖️ F1 Score", f"{metrics['F1 Score']:.4f}")
            st.metric("📉 MCC", f"{metrics['MCC']:.4f}")
        
        # Confusion Matrix and Classification Report
        st.subheader("📊 Confusion Matrix & Classification Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_confusion_matrix(cm, f"{selected_model} - Confusion Matrix")
            st.pyplot(fig)
        
        with col2:
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    
    with tab2:
        st.header("📈 Model Comparison")
        
        with st.spinner("Training all models..."):
            comparison_df = train_all_models(X_train, X_test, y_train, y_test)
        
        # Reorder columns
        comparison_df = comparison_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']]
        
        # Display comparison table
        st.subheader("📊 Metrics Comparison Table")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                color='lightgreen'
            ).format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'MCC': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Bar chart comparison
        st.subheader("📊 Visual Comparison")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
        
        for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, alpha=0.8)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, val in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model summary
        best_accuracy_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_f1_model = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
        
        st.success(f"🏆 **Best Accuracy**: {best_accuracy_model} ({comparison_df['Accuracy'].max():.4f})")
        st.success(f"🏆 **Best F1 Score**: {best_f1_model} ({comparison_df['F1 Score'].max():.4f})")
    
    with tab3:
        st.header("📊 Feature Analysis")
        
        # Feature importance for tree-based models
        if selected_model in ['Decision Tree', 'Random Forest', 'XGBoost']:
            st.subheader(f"🌳 {selected_model} - Feature Importance")
            
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature', 
                       palette='viridis', ax=ax)
            ax.set_title(f'{selected_model} - Top 15 Feature Importance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            st.pyplot(fig)
            
            # Show table
            st.dataframe(feature_importance_df, use_container_width=True)
        else:
            st.info(f"Feature importance is available for tree-based models (Decision Tree, Random Forest, XGBoost). Selected model: {selected_model}")
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        
        # Limit to top features for readability
        n_features = min(15, len(feature_names))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = X.iloc[:, :n_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=ax, fmt='.2f', annot_kws={"size": 8})
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)

else:
    # No file uploaded - show instructions
    st.header("🚀 Getting Started")
    
    st.markdown("""
    ### Welcome to the ML Classification Models Dashboard!
    
    This application allows you to:
    
    1. **📂 Upload a Dataset**: Upload your CSV file using the sidebar
    2. **🔍 Select a Model**: Choose from 6 different classification algorithms
    3. **📊 View Metrics**: See accuracy, AUC, precision, recall, F1, and MCC scores
    4. **📈 Compare Models**: Compare all models side-by-side
    5. **🔥 Analyze Features**: View feature importance and correlations
    
    ### 📋 Dataset Requirements:
    - **Format**: CSV file
    - **Target Variable**: Last column should be the target/label
    - **Minimum Features**: 12
    - **Minimum Samples**: 500
    
    ### 🤖 Available Models:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Logistic Regression**: Linear model for binary/multiclass classification
        - **Decision Tree**: Tree-based model with interpretable rules
        - **K-Nearest Neighbors**: Instance-based learning algorithm
        """)
    
    with col2:
        st.markdown("""
        - **Naive Bayes**: Probabilistic classifier based on Bayes theorem
        - **Random Forest**: Ensemble of decision trees
        - **XGBoost**: Gradient boosting ensemble method
        """)
    
    st.markdown("""
    ### 📊 Evaluation Metrics:
    | Metric | Description |
    |--------|-------------|
    | **Accuracy** | Proportion of correct predictions |
    | **AUC** | Area Under the ROC Curve |
    | **Precision** | Proportion of true positives among predicted positives |
    | **Recall** | Proportion of true positives among actual positives |
    | **F1 Score** | Harmonic mean of precision and recall |
    | **MCC** | Matthews Correlation Coefficient - balanced measure |
    """)
    
    st.info("👈 **Upload a CSV file using the sidebar to get started!**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | ML Classification Models Dashboard</p>
    <p>BITS Pilani - M.Tech (AIML/DSE) - Machine Learning Assignment 2</p>
</div>
""", unsafe_allow_html=True)
