# ML Classification Models Dashboard

A comprehensive machine learning classification project that implements 6 different classification algorithms and provides an interactive Streamlit web application for model evaluation and comparison.

## 📌 Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models on a chosen dataset and deploy an interactive web application for demonstration. This project covers the complete end-to-end ML deployment workflow: data preprocessing, model training, evaluation, UI design, and cloud deployment.

The application allows users to:
- Upload their own classification datasets
- Select and train different ML models
- View comprehensive evaluation metrics
- Compare model performances visually

## 📊 Dataset Description

**Dataset Name**: [Your Dataset Name - e.g., Heart Disease UCI / Bank Marketing / etc.]

**Source**: [Kaggle / UCI Machine Learning Repository]

**Dataset Characteristics**:
| Property | Value |
|----------|-------|
| Number of Instances | [e.g., 1000+] |
| Number of Features | [e.g., 15] |
| Target Variable | [e.g., Binary Classification - 0/1] |
| Missing Values | [e.g., None / Handled] |

**Feature Description**:
| Feature Name | Type | Description |
|--------------|------|-------------|
| feature_1 | Numeric | [Description] |
| feature_2 | Numeric | [Description] |
| feature_3 | Categorical | [Description] |
| ... | ... | ... |
| target | Binary | Target variable (0/1) |

**Data Preprocessing Steps**:
1. Handled missing values using [method]
2. Encoded categorical variables using Label Encoding
3. Standardized numerical features for certain models (Logistic Regression, KNN)
4. Split data into training (80%) and testing (20%) sets

## 🤖 Models Used

The following 6 classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for classification
2. **Decision Tree Classifier** - Tree-based interpretable model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble

### 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| K-Nearest Neighbors | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

> **Note**: Replace the `0.XXXX` values with your actual results after running the models on your chosen dataset.

### 📝 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | [e.g., "Performs well as a baseline model. Good for linearly separable data. Fast training time and interpretable coefficients."] |
| Decision Tree | [e.g., "Prone to overfitting without proper pruning. Provides interpretable decision rules. Feature importance is easily extractable."] |
| K-Nearest Neighbors | [e.g., "Sensitive to feature scaling and curse of dimensionality. Performance depends on optimal k value. Slow prediction for large datasets."] |
| Naive Bayes | [e.g., "Fast training and prediction. Works well with high-dimensional data. Assumes feature independence which may not hold."] |
| Random Forest (Ensemble) | [e.g., "Reduces overfitting compared to single decision tree. Good handling of imbalanced data. Provides robust feature importance."] |
| XGBoost (Ensemble) | [e.g., "Best performing model with highest accuracy. Handles missing values well. Requires careful hyperparameter tuning."] |

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

1. **📂 Dataset Upload** - Upload CSV files for classification
2. **🔍 Model Selection** - Dropdown to select from 6 different models
3. **📊 Evaluation Metrics Display** - Shows Accuracy, AUC, Precision, Recall, F1, MCC
4. **📈 Confusion Matrix** - Visual confusion matrix heatmap
5. **📋 Classification Report** - Detailed per-class metrics
6. **📊 Model Comparison** - Side-by-side comparison of all models
7. **🌳 Feature Importance** - For tree-based models

## 📁 Project Structure

```
project-folder/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── model/                    # Model files and training scripts
│   └── train_models.py       # Script to train all models
│
└── data/                     # Dataset files (optional)
    ├── data.csv              # Training dataset
    └── test_data.csv         # Test dataset
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Community Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository and branch
6. Choose `app.py` as the main file
7. Click "Deploy"

**Live App Link**: [Your Streamlit App URL]

## 📊 How to Use

1. **Upload Dataset**: Click on "Upload CSV file" in the sidebar
   - Ensure your CSV has the target variable as the last column
   - Minimum 12 features and 500 instances recommended

2. **Select Model**: Choose a classification model from the dropdown

3. **View Results**: 
   - Single Model Evaluation tab shows detailed metrics
   - Model Comparison tab compares all 6 models
   - Feature Analysis tab shows feature importance

## 📚 Evaluation Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions |
| **AUC** | Area Under the ROC Curve - measures discrimination ability |
| **Precision** | Proportion of true positives among predicted positives |
| **Recall** | Proportion of true positives among actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **MCC** | Matthews Correlation Coefficient - balanced measure for imbalanced data |

## 🔗 Links

- **GitHub Repository**: [Your GitHub Repo Link]
- **Live Streamlit App**: [Your Streamlit App Link]

## 👨‍💻 Author

**Name**: [Your Name]  
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani  
**Course**: Machine Learning - Assignment 2

## 📝 License

This project is for educational purposes as part of the Machine Learning course at BITS Pilani.

---

*Built with ❤️ using Python, Scikit-learn, and Streamlit*
