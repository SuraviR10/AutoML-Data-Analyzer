"""
Auto Data Analyzer & ML Recommender - Core Engine
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Non-informative column patterns to auto-drop
_ID_PATTERNS = ['id', 'index', 'row', 'unnamed', 'serial', 'no.', 'sr.', 'sno']


def _tof(x: object) -> float:
    """Safely convert any pandas/numpy scalar to Python float."""
    return float(np.asarray(x).item())


def _toi(x: object) -> int:
    """Safely convert any pandas/numpy scalar to Python int."""
    return int(np.asarray(x).item())


def _is_id_column(col: str, series: pd.Series) -> bool:
    """Detect non-informative ID/index columns."""
    col_lower = col.lower().strip()
    if any(col_lower == p or col_lower.startswith(p) for p in _ID_PATTERNS):
        return True
    # Numeric column where every value is unique and sequential-ish
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() == len(series):
            diffs = series.sort_values().diff().dropna()
            if len(diffs) > 0 and float(diffs.std()) < 1e-6:
                return True
    return False


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.target_col: str | None = None
        self.problem_type: str | None = None
        self.recommendations: list[dict] = []
        self.insights: list[str] = []
        self.dropped_cols: list[str] = []
        self.preprocessing_log: list[str] = []

    def drop_id_columns(self) -> list[str]:
        """Auto-detect and drop non-informative ID/index columns."""
        to_drop = [col for col in self.df.columns if _is_id_column(col, self.df[col])]
        if to_drop:
            self.df = self.df.drop(columns=to_drop)
            self.dropped_cols.extend(to_drop)
            self.preprocessing_log.append(f"Dropped non-informative columns: {', '.join(to_drop)}")
        return to_drop

    def analyze_types(self):
        """Correctly detect numeric vs categorical columns based on dtype."""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        return self.numeric_cols, self.categorical_cols

    def handle_missing_values(self, strategy: str = 'auto'):
        missing_info: dict = {}
        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            if missing_count > 0:
                pct = round(missing_count / len(self.df) * 100, 2)
                missing_info[col] = {'count': missing_count, 'percentage': pct}
                if strategy == 'auto':
                    if col in self.numeric_cols:
                        fill = _tof(self.df[col].median())
                        self.df[col] = self.df[col].fillna(fill)
                        self.preprocessing_log.append(f"Filled {missing_count} missing in '{col}' with median ({fill:.2f})")
                    else:
                        fill_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                        self.df[col] = self.df[col].fillna(fill_val)
                        self.preprocessing_log.append(f"Filled {missing_count} missing in '{col}' with mode ('{fill_val}')")
                elif strategy == 'mean' and col in self.numeric_cols:
                    fill = _tof(self.df[col].mean())
                    self.df[col] = self.df[col].fillna(fill)
                elif strategy == 'median' and col in self.numeric_cols:
                    fill = _tof(self.df[col].median())
                    self.df[col] = self.df[col].fillna(fill)
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    self.df[col] = self.df[col].fillna(fill_val)
        return missing_info

    def remove_duplicates(self) -> int:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed > 0:
            self.preprocessing_log.append(f"Removed {removed} duplicate rows")
        return removed

    def handle_outliers(self, method: str = 'iqr'):
        outlier_info: dict = {}
        for col in self.numeric_cols:
            Q1 = _tof(self.df[col].quantile(0.25))
            Q3 = _tof(self.df[col].quantile(0.75))
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            if n_outliers > 0:
                outlier_info[col] = {'count': n_outliers, 'lower_bound': lower, 'upper_bound': upper}
                if method == 'iqr':
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                    self.preprocessing_log.append(f"Capped {n_outliers} outliers in '{col}' using IQR")
                elif method == 'remove':
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        return outlier_info

    def encode_categorical(self, method: str = 'label'):
        encoded_cols: dict = {}
        for col in list(self.categorical_cols):
            if col == self.target_col:
                continue  # Don't encode target here
            if method == 'label':
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                self.df[col] = encoded.tolist()  # assign as list for pandas compatibility
                encoded_cols[col] = 'label'
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                encoded_cols[col] = 'onehot'
        if encoded_cols:
            self.preprocessing_log.append(f"Encoded {len(encoded_cols)} categorical columns using {method} encoding")
        return encoded_cols

    def compute_correlations(self):
        if len(self.numeric_cols) > 1:
            return self.df[self.numeric_cols].corr()
        return None

    def get_statistical_summary(self) -> pd.DataFrame:
        if not self.numeric_cols:
            return pd.DataFrame()
        stats = self.df[self.numeric_cols].describe().T
        stats['skewness'] = self.df[self.numeric_cols].skew()
        stats['kurtosis'] = self.df[self.numeric_cols].kurt()
        stats['missing_%'] = (self.df[self.numeric_cols].isnull().sum() / len(self.df) * 100).round(2)
        return stats.round(4)

    def detect_problem_type(self, target_col: str | None = None) -> str:
        """
        Correct problem type detection:
        - object/category dtype → classification
        - numeric with ≤15 unique values → classification
        - numeric with >15 unique values → regression
        - no target → clustering
        """
        self.target_col = target_col
        
        if not self.numeric_cols and not self.categorical_cols:
            self.analyze_types()

        if self.target_col is None:
            # Auto-infer: look for common target column names
            for col in reversed(self.df.columns): # Search from end (often target is last)
                if col.lower() in ['target', 'label', 'class', 'y', 'output', 'result', 'outcome', 'survived', 'churn']:
                    self.target_col = col
                    break
            if self.target_col is None and self.numeric_cols:
                self.target_col = self.numeric_cols[-1]

        if self.target_col and self.target_col in self.df.columns:
            col_dtype = self.df[self.target_col].dtype
            n_unique = int(self.df[self.target_col].nunique())
            n_rows = len(self.df)
            
            if n_rows == 0:
                self.problem_type = 'clustering'
                return self.problem_type
                
            # Rule 1: object/category/bool dtype → always classification
            if col_dtype == object or str(col_dtype) in ['category', 'bool']:
                self.problem_type = 'classification'
            # Rule 2: numeric with few unique values → classification
            elif n_unique <= 15 or (n_unique / n_rows < 0.05 and n_unique <= 50):
                self.problem_type = 'classification'
            # Rule 3: numeric with many unique values → regression
            else:
                self.problem_type = 'regression'
        else:
            self.problem_type = 'clustering'

        return self.problem_type  # type: ignore[return-value]

    def detect_data_issues(self) -> list[str]:
        issues: list[str] = []
        total_cells = self.df.shape[0] * self.df.shape[1]
        if total_cells > 0:
            missing_pct = _tof(self.df.isnull().sum().sum()) / total_cells * 100
            if missing_pct > 5:
                issues.append(f"⚠️ High missing values: {missing_pct:.1f}% of all cells")

        dupes = int(self.df.duplicated().sum())
        if dupes > 0:
            issues.append(f"⚠️ {dupes} duplicate rows detected")

        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1:
                ratio = _tof(vc.max()) / _tof(vc.min())
                if ratio > 5:
                    issues.append(f"⚠️ Class imbalance: majority/minority ratio = {ratio:.1f}x")

        for col in self.numeric_cols[:8]:
            skew = abs(_tof(self.df[col].skew()))
            if skew > 2:
                issues.append(f"⚠️ High skewness in '{col}': {_tof(self.df[col].skew()):.2f}")

        return issues

    def recommend_algorithms(self) -> list[dict]:
        self.recommendations = []
        n_samples = len(self.df)
        n_features = len(self.numeric_cols)

        if self.problem_type == 'classification':
            is_imbalanced = False
            n_classes = 2
            if self.target_col and self.target_col in self.df.columns:
                vc = self.df[self.target_col].value_counts()
                n_classes = len(vc)
                if len(vc) > 1:
                    is_imbalanced = _tof(vc.min()) / _tof(vc.max()) < 0.2

            self.recommendations = [
                {
                    'algorithm': 'Random Forest',
                    'description': f'Ensemble of 200 decision trees. Best for {n_samples} samples, {n_features} features.',
                    'pros': 'Robust to overfitting, handles missing data, provides feature importance',
                    'cons': 'Less interpretable, slower than single tree',
                    'use_when': 'General purpose — recommended starting point',
                    'why': f'With {n_samples} samples and {n_features} features, Random Forest provides the best balance of accuracy and robustness.'
                },
                {
                    'algorithm': 'Gradient Boosting',
                    'description': f'Sequential boosting correcting errors. Ideal for {"imbalanced" if is_imbalanced else "balanced"} {n_classes}-class problem.',
                    'pros': 'Highest accuracy, handles class imbalance, built-in regularization',
                    'cons': 'Slower training, more hyperparameters to tune',
                    'use_when': 'When maximum accuracy is required',
                    'why': 'Gradient Boosting typically achieves the highest accuracy by iteratively correcting prediction errors.'
                },
                {
                    'algorithm': 'Logistic Regression',
                    'description': f'Linear classifier. Fast baseline for {n_classes}-class problem.',
                    'pros': 'Fast, interpretable, probabilistic output, works well with scaled features',
                    'cons': 'Assumes linear decision boundary, may underfit complex patterns',
                    'use_when': 'Need interpretable model or quick baseline',
                    'why': 'Provides a fast, interpretable baseline to compare against complex models.'
                },
                {
                    'algorithm': 'SVM (RBF Kernel)',
                    'description': f'Finds optimal hyperplane. Effective for {n_features}-dimensional space.',
                    'pros': 'Effective in high dimensions, robust to outliers with proper C parameter',
                    'cons': 'Slow on large datasets (>10k rows), requires feature scaling',
                    'use_when': 'High-dimensional data, small to medium datasets',
                    'why': f'With {n_features} features, SVM can find complex non-linear boundaries effectively.'
                },
                {
                    'algorithm': 'Decision Tree',
                    'description': 'Single tree with depth limit. Fully interpretable rules.',
                    'pros': 'Fully interpretable, no feature scaling needed, handles non-linearity',
                    'cons': 'Prone to overfitting, unstable with small data changes',
                    'use_when': 'Need explainable rules or quick visualization',
                    'why': 'Best choice when you need to explain every prediction as a set of if-else rules.'
                },
            ]

        elif self.problem_type == 'regression':
            corr = self.compute_correlations()
            high_corr = 0
            if corr is not None:
                high_corr = int(((corr.abs() > 0.8) & (corr.abs() < 1.0)).sum().sum() // 2)

            self.recommendations = [
                {
                    'algorithm': 'Gradient Boosting',
                    'description': f'Sequential boosting for regression. Best for {n_samples} samples.',
                    'pros': 'Highest accuracy, handles outliers and non-linear relationships',
                    'cons': 'Slower training, requires tuning',
                    'use_when': 'Need maximum accuracy',
                    'why': 'Gradient Boosting consistently achieves the lowest error on regression tasks.'
                },
                {
                    'algorithm': 'Random Forest',
                    'description': f'Ensemble regression with {n_features} features.',
                    'pros': 'Robust, handles complexity, provides feature importance',
                    'cons': 'Less interpretable than linear models',
                    'use_when': 'Complex non-linear relationships',
                    'why': f'With {n_features} features, Random Forest captures complex interactions without overfitting.'
                },
                {
                    'algorithm': 'Ridge Regression',
                    'description': f'L2-regularized linear regression. {"Handles " + str(high_corr) + " correlated feature pairs." if high_corr > 0 else "Stable linear baseline."}',
                    'pros': 'Prevents overfitting, stable with correlated features, fast',
                    'cons': 'Assumes linearity, may underfit complex patterns',
                    'use_when': 'Multicollinearity present, need stable coefficients',
                    'why': f'{"Detected " + str(high_corr) + " highly correlated feature pairs — Ridge handles this better than plain Linear Regression." if high_corr > 0 else "Good regularized baseline for linear relationships."}'
                },
                {
                    'algorithm': 'Linear Regression',
                    'description': 'Ordinary least squares. Simplest interpretable baseline.',
                    'pros': 'Simple, interpretable, fast, coefficient significance testing',
                    'cons': 'Assumes linearity, sensitive to outliers and multicollinearity',
                    'use_when': 'Linear relationship exists, need interpretability',
                    'why': 'Provides the most interpretable model — each coefficient directly shows feature impact.'
                },
                {
                    'algorithm': 'Decision Tree Regressor',
                    'description': 'Tree-based regression with depth limit.',
                    'pros': 'Handles non-linearity, no scaling needed, interpretable',
                    'cons': 'Prone to overfitting, high variance',
                    'use_when': 'Non-linear data, need interpretable rules',
                    'why': 'Captures non-linear patterns while remaining interpretable through tree visualization.'
                },
            ]

        elif self.problem_type == 'clustering':
            self.recommendations = [
                {
                    'algorithm': 'K-Means',
                    'description': f'Partitions {n_samples} samples into K clusters by minimizing inertia.',
                    'pros': 'Simple, fast, scales well, easy to interpret',
                    'cons': 'Must specify K, assumes spherical clusters, sensitive to outliers',
                    'use_when': 'Known or estimated number of clusters, large datasets',
                    'why': 'Most efficient clustering algorithm — use elbow method to find optimal K.'
                },
                {
                    'algorithm': 'DBSCAN',
                    'description': 'Density-based clustering — finds clusters of arbitrary shape.',
                    'pros': 'No need to specify K, detects outliers, finds arbitrary shapes',
                    'cons': 'Sensitive to eps and min_samples parameters',
                    'use_when': 'Unknown number of clusters, noisy data',
                    'why': 'Best when clusters have irregular shapes or when outlier detection is important.'
                },
                {
                    'algorithm': 'Hierarchical Clustering',
                    'description': 'Agglomerative clustering building a dendrogram.',
                    'pros': 'No need to specify K, provides dendrogram, deterministic',
                    'cons': 'O(n²) complexity, slow for large datasets',
                    'use_when': 'Small datasets, need hierarchical structure',
                    'why': 'Provides a complete picture of cluster relationships through dendrogram visualization.'
                },
            ]

        return self.recommendations

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """Train models on train set, evaluate ONLY on held-out test set."""
        results: list[dict] = []

        if self.problem_type == 'classification' and self.target_col:
            X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
            y = self.df[self.target_col]
            if X.empty or len(y) < 5: # Lower threshold for tiny datasets
                return pd.DataFrame()

            # Encode target if categorical
            if y.dtype == object or str(y.dtype) == 'category':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                try:
                    y = pd.Series(y_encoded, index=y.index, dtype=int)
                except Exception:
                    y = pd.Series(y_encoded, index=y.index)

            try:
                strat = y if y.nunique() <= 20 and y.value_counts().min() >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split( # Logic fix for tiny datasets
                    X, y, test_size=test_size, random_state=random_state, stratify=strat
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models: dict = {
                'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=2000, C=1.0),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=8, min_samples_split=5),
                'Random Forest': RandomForestClassifier(n_estimators=150, random_state=random_state, max_features='sqrt', n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=random_state, learning_rate=0.1, max_depth=4),
                'SVM': SVC(random_state=random_state, kernel='rbf', C=10, gamma='scale'),
            }
            for name, model in models.items():
                try:
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)  # Evaluate on TEST set only
                    results.append({
                        'Model': name,
                        'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                        'Precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                        'Recall': round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                        'F1-Score': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                    })
                except Exception:
                    pass

        elif self.problem_type == 'regression' and self.target_col:
            X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
            y = self.df[self.target_col]
            if X.empty or len(y) < 10:
                return pd.DataFrame()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            reg_models: dict = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=0.5),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=8),
                'Random Forest': RandomForestRegressor(n_estimators=150, random_state=random_state, max_features='sqrt', n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=random_state, learning_rate=0.1, max_depth=4),
            }
            for name, model in reg_models.items():
                try:
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)  # Evaluate on TEST set only
                    results.append({
                        'Model': name,
                        'MAE': round(float(mean_absolute_error(y_test, y_pred)), 4),
                        'RMSE': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                        'R2-Score': round(float(r2_score(y_test, y_pred)), 4),
                    })
                except Exception:
                    pass

        elif self.problem_type == 'clustering':
            X = self.df.select_dtypes(include=[np.number])
            if X.empty:
                return pd.DataFrame()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            for k in range(2, min(9, len(X) // 5 + 2)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
                    labels = kmeans.fit_predict(X_scaled)
                    if len(np.unique(labels)) > 1:
                        results.append({
                            'K': k,
                            'Silhouette Score': round(float(silhouette_score(X_scaled, labels)), 4),
                            'Inertia': round(float(kmeans.inertia_), 2),
                        })
                except Exception:
                    pass

        return pd.DataFrame(results)

    def generate_insights(self) -> list[str]:
        self.insights = []
        self.insights.append(f"📊 Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        self.insights.append(f"🔢 Numeric Features: {len(self.numeric_cols)} | 📝 Categorical: {len(self.categorical_cols)}")
        if self.problem_type:
            self.insights.append(f"🎯 Problem Type: {self.problem_type.title()}")
        missing = int(self.df.isnull().sum().sum())
        self.insights.append(f"{'⚠️' if missing > 0 else '✅'} Missing Values: {missing}")
        dupes = int(self.df.duplicated().sum())
        self.insights.append(f"{'⚠️' if dupes > 0 else '✅'} Duplicates: {dupes}")
        for issue in self.detect_data_issues():
            self.insights.append(issue)
        if not self.detect_data_issues():
            self.insights.append("✅ Data Quality: No significant issues detected")
        return self.insights


def preprocess_data(
    df: pd.DataFrame,
    missing_strategy: str = 'auto',
    remove_outliers: bool = True,
    encode_method: str = 'label',
    drop_id_cols: bool = True
) -> 'DataAnalyzer':
    analyzer = DataAnalyzer(df)
    analyzer.analyze_types()
    if drop_id_cols:
        analyzer.drop_id_columns()
        analyzer.analyze_types()  # Re-analyze after dropping
    analyzer.handle_missing_values(strategy=missing_strategy)
    analyzer.remove_duplicates()
    if remove_outliers:
        analyzer.handle_outliers(method='iqr')
    analyzer.encode_categorical(method=encode_method)
    analyzer.analyze_types()  # Final re-analyze for consistency
    return analyzer
