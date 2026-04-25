# Auto Data Analyzer & ML Recommender

An intelligent machine learning system that automates the complete data analysis pipeline—from preprocessing to model recommendation—helping users quickly understand datasets and select suitable ML algorithms.

## 🎯 Features

### 1. 📥 Data Input
- Accepts datasets in CSV/Excel format
- Loads data using pandas into a DataFrame

### 2. 🧹 Data Preprocessing
- Handles missing values (mean/median/mode)
- Encodes categorical features (Label / One-Hot Encoding)
- Removes duplicates
- Detects and handles outliers

### 3. 📊 Exploratory Data Analysis (EDA)
- Identifies feature types (numerical/categorical)
- Computes correlations
- Detects dataset issues (imbalance, skewness)
- Generates basic feature importance

### 4. 📈 Data Visualization
- Histograms
- Box plots
- Heatmaps
- Count plots

### 5. 🤖 ML Recommendation Engine
- Automatically detects problem type:
  - Classification
  - Regression
  - Clustering
- Suggests suitable algorithms with pros/cons

### 6. ⚡ Model Evaluation
- Trains multiple models
- Compares performance metrics
- Visualizes results

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python |
| Data Handling | Pandas, NumPy |
| ML Models | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Frontend | Streamlit |

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Use the Application

1. Upload your dataset (CSV or Excel)
2. Configure preprocessing options in the sidebar
3. Explore different tabs:
   - **Dataset Overview**: View raw data and column info
   - **Preprocessing**: Clean and transform data
   - **EDA & Visualizations**: Analyze patterns
   - **ML Recommendations**: Get algorithm suggestions
   - **Model Evaluation**: Train and compare models

## 📁 Project Structure

```
ML/
├── app.py              # Streamlit UI
├── data_analyzer.py    # Core ML pipeline
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 💡 Usage Tips

- **Target Column**: Select the column you want to predict
- **Problem Type**: Automatically detected based on target column
- **Preprocessing**: Adjust strategies based on your data quality
- **Model Evaluation**: Compare multiple algorithms to find the best

## 🔧 Configuration Options

### Missing Value Strategy
- `auto`: Median for numeric, mode for categorical
- `mean`: Fill with column mean
- `median`: Fill with column median
- `mode`: Fill with most frequent value

### Outlier Handling
- `Remove Outliers`: Cap outliers using IQR method

### Categorical Encoding
- `label`: Label encoding (ordinal)
- `onehot`: One-hot encoding (dummy variables)

## 📊 Example Datasets

The system works well with:
- Customer churn datasets
- Sales forecasting data
- Classification problems
- Regression problems
- Clustering analysis

## 🏆 Key Benefits

- ✅ Automatic problem type detection
- ✅ Intelligent model recommendation
- ✅ Dataset issue detection
- ✅ Performance-based model comparison
- ✅ Natural language insights
- ✅ Interactive visualizations

## 📝 License

MIT License