# 🤖 AutoML - Data Analyzer & ML Recommender

An intelligent machine learning system that automates the complete data analysis pipeline—from preprocessing to model recommendation—helping users quickly understand datasets and select suitable ML algorithms.

![AutoML Dashboard](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

### 1. 📊 Dataset Overview
- **Comprehensive Metrics**: Rows, columns, missing values, duplicates, file size
- **Feature Type Distribution**: Visual breakdown of numeric vs categorical features
- **Data Quality Score**: Automated quality assessment with visual gauge (0-100)
- **Dataset Preview**: Quick view of first 5 rows
- **Correlation Heatmap**: Interactive correlation matrix for numeric features
- **Problem Type Detection**: Automatic detection of classification/regression/clustering

### 2. ⚙️ Data Preprocessing
- **Missing Value Handling**: Multiple strategies (auto, mean, median, mode)
- **Duplicate Removal**: Automatic detection and removal
- **Outlier Treatment**: IQR-based outlier detection and capping
- **Categorical Encoding**: Label encoding and one-hot encoding
- **ID Column Detection**: Automatic removal of non-informative columns
- **Preprocessing Log**: Detailed log of all cleaning steps applied

### 3. 📈 EDA & Visualizations
- **Distribution Plots**: Histograms for numeric features
- **Box Plots**: Outlier detection visualization
- **Count Plots**: Categorical feature distributions
- **Scatter Plots**: Relationship exploration
- **Interactive Selection**: Choose visualization type and columns

### 4. 🤖 ML Recommendation Engine
- **Automatic Problem Detection**:
  - Classification (binary/multiclass)
  - Regression (continuous prediction)
  - Clustering (unsupervised learning)
- **Smart Algorithm Suggestions**: Top 5 algorithms with:
  - Detailed descriptions
  - Pros and cons
  - Use case recommendations
  - Why it's suitable for your data
- **Context-Aware**: Recommendations based on:
  - Dataset size
  - Number of features
  - Class imbalance
  - Feature correlations

### 5. 🏆 Model Evaluation
- **Automated Training**: Train multiple models with one click
- **Performance Metrics**:
  - Classification: Accuracy, Precision, Recall, F1-Score
  - Regression: MAE, RMSE, R² Score
  - Clustering: Silhouette Score, Inertia
- **Best Model Identification**: Automatic selection of top performer
- **Visual Comparison**: Bar charts comparing all models
- **Train-Test Split**: Proper evaluation on held-out test set

### 6. 📥 Export Results
- **Download Cleaned Data**: Export preprocessed dataset as CSV
- **Analysis Report**: Comprehensive markdown report with:
  - Dataset summary
  - Preprocessing steps
  - Key insights
  - Recommendations

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.8+ |
| **Data Handling** | Pandas, NumPy |
| **ML Models** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | Streamlit |
| **File Support** | CSV, Excel (openpyxl) |

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload Dataset
1. Click "Browse files" in the sidebar
2. Select a CSV or Excel file
3. Supported formats: `.csv`, `.xlsx`

### Step 2: Configure Settings
- **Missing Values**: Choose imputation strategy (auto/mean/median/mode)
- **Remove Outliers**: Toggle IQR-based outlier removal
- **Encoding**: Select label or one-hot encoding for categorical features

### Step 3: Explore Tabs

#### 📊 Overview
- View dataset statistics and metrics
- Check data quality score
- Explore feature correlations
- Preview your data

#### ⚙️ Preprocessing
- Review cleaning steps applied
- Check before/after statistics
- Preview cleaned data

#### 📈 EDA & Visualizations
- Select visualization type
- Explore data distributions
- Identify patterns and outliers

#### 🤖 ML Recommendations
- Select target column (or use auto-detect)
- View detected problem type
- Read algorithm recommendations
- Understand pros/cons of each approach

#### 🏆 Model Evaluation
- Train multiple models automatically
- Compare performance metrics
- Identify best model
- View comparison charts

#### 📥 Export Results
- Download cleaned dataset
- Download analysis report

## 📊 Example Datasets

The system works with any tabular dataset. Example use cases:

- **Classification**: Customer churn, disease diagnosis, spam detection
- **Regression**: House price prediction, sales forecasting, temperature prediction
- **Clustering**: Customer segmentation, anomaly detection, pattern discovery

### Sample Dataset Format

```csv
Age,Salary,Department,Performance,Target
25,50000,IT,Good,1
30,60000,HR,Excellent,0
35,75000,Sales,Good,1
```

## 🎨 Dashboard Design

The application features a modern, dark-themed interface with:
- **Gradient backgrounds** for visual appeal
- **Card-based layouts** for organized information
- **Interactive charts** with consistent color schemes
- **Responsive design** that works on different screen sizes
- **Smooth transitions** and hover effects

## 🔧 Configuration

### Preprocessing Options

| Option | Values | Description |
|--------|--------|-------------|
| Missing Values | auto, mean, median, mode | Strategy for filling missing values |
| Remove Outliers | True/False | Apply IQR-based outlier removal |
| Encoding | label, onehot | Method for encoding categorical features |

### Model Parameters

All models use optimized default parameters:
- **Random Forest**: 150 estimators, sqrt max_features
- **Gradient Boosting**: 150 estimators, 0.1 learning rate
- **SVM**: RBF kernel, C=10
- **Logistic Regression**: max_iter=2000, C=1.0

## 🐛 Troubleshooting

### Common Issues

**Issue**: "File not found" error
- **Solution**: Ensure the file path is correct and file is not corrupted

**Issue**: "Not enough data to train models"
- **Solution**: Dataset needs at least 10 rows with numeric features

**Issue**: Slow performance
- **Solution**: Large datasets (>100k rows) may take longer to process

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt` to install all dependencies

## 📝 Project Structure

```
ML/
├── app.py                 # Main Streamlit application
├── data_analyzer.py       # Core analysis and ML engine
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML powered by [Scikit-learn](https://scikit-learn.org/)
- Visualizations by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for the Data Science Community**
