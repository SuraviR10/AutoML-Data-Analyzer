"""
Auto Data Analyzer & ML Recommender - Core Engine
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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


def _tof(x: object) -> float:
    """Safely convert any pandas/numpy scalar to Python float."""
    return float(np.asarray(x).item())


def _toi(x: object) -> int:
    """Safely convert any pandas/numpy scalar to Python int."""
    return int(np.asarray(x).item())


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.target_col: str | None = None
        self.problem_type: str | None = None
        self.recommendations: list[dict] = []
        self.insights: list[str] = []

    def analyze_types(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return self.numeric_cols, self.categorical_cols

    def handle_missing_values(self, strategy: str = 'auto'):
        missing_info: dict = {}
        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            if missing_count > 0:
                missing_info[col] = {'count': missing_count, 'percentage': missing_count / len(self.df) * 100}
                if strategy == 'auto':
                    if col in self.numeric_cols:
                        self.df[col] = self.df[col].fillna(_tof(self.df[col].median()))
                    else:
                        fill_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                        self.df[col] = self.df[col].fillna(fill_val)
                elif strategy == 'mean' and col in self.numeric_cols:
                    self.df[col] = self.df[col].fillna(_tof(self.df[col].mean()))
                elif strategy == 'median' and col in self.numeric_cols:
                    self.df[col] = self.df[col].fillna(_tof(self.df[col].median()))
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    self.df[col] = self.df[col].fillna(fill_val)
        return missing_info

    def remove_duplicates(self) -> int:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return before - len(self.df)

    def handle_outliers(self, method: str = 'iqr'):
        outlier_info: dict = {}
        for col in self.numeric_cols:
            Q1 = _tof(self.df[col].quantile(0.25))
            Q3 = _tof(self.df[col].quantile(0.75))
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            if n_outliers > 0:
                outlier_info[col] = {'count': n_outliers, 'lower_bound': lower, 'upper_bound': upper}
                if method == 'iqr':
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                elif method == 'remove':
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        return outlier_info

    def encode_categorical(self, method: str = 'label'):
        encoded_cols: dict = {}
        for col in list(self.categorical_cols):
            if method == 'label':
                le = LabelEncoder()
                encoded = le.fit_transform(self.df[col].astype(str))
                self.df[col] = pd.Series(encoded, index=self.df.index)
                encoded_cols[col] = 'label'
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                encoded_cols[col] = 'onehot'
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

    def compute_data_health_score(self) -> tuple[float, dict]:
        score = 100.0
        penalties: dict = {}

        missing_pct = _tof(self.df.isnull().sum().sum()) / (self.df.shape[0] * self.df.shape[1]) * 100
        mp = min(missing_pct * 3, 30)
        score -= mp
        penalties['Missing Values'] = round(mp, 1)

        dup_pct = _tof(self.df.duplicated().sum()) / len(self.df) * 100
        dp = min(dup_pct * 2, 20)
        score -= dp
        penalties['Duplicates'] = round(dp, 1)

        skew_count = sum(1 for col in self.numeric_cols if abs(_tof(self.df[col].skew())) > 2)
        sp = min(skew_count * 4, 20)
        score -= sp
        penalties['High Skewness'] = round(sp, 1)

        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1:
                ratio = _tof(vc.max()) / _tof(vc.min())
                ip = min((ratio - 1) * 1.5, 15)
                score -= ip
                penalties['Class Imbalance'] = round(ip, 1)

        low_var = sum(1 for col in self.numeric_cols if _tof(self.df[col].std()) < 1e-6)
        lp = min(low_var * 5, 15)
        score -= lp
        penalties['Low Variance'] = round(lp, 1)

        return max(round(score, 1), 0), penalties

    def get_feature_engineering_tips(self) -> list[str]:
        tips: list[str] = []
        for col in self.numeric_cols:
            skew = abs(_tof(self.df[col].skew()))
            if skew > 2:
                tips.append(f"🔄 **{col}**: High skew ({skew:.2f}) — consider log/sqrt transform")
        if len(self.numeric_cols) >= 2:
            corr = self.df[self.numeric_cols].corr()
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = abs(_tof(corr.iloc[i, j]))
                    if val > 0.9:
                        tips.append(f"⚠️ **{cols[i]}** & **{cols[j]}**: Highly correlated ({_tof(corr.iloc[i,j]):.2f}) — consider dropping one")
        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1 and _tof(vc.max()) / _tof(vc.min()) > 5:
                tips.append("⚖️ **Target Column**: Class imbalance detected — consider SMOTE or class weighting")
        if not tips:
            tips.append("✅ No critical feature engineering issues detected!")
        return tips

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self.target_col is None or self.target_col not in self.df.columns:
            return None
        X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
        y = self.df[self.target_col]
        if X.empty:
            return None
        try:
            model = (RandomForestClassifier(n_estimators=100, random_state=42)
                     if self.problem_type == 'classification'
                     else RandomForestRegressor(n_estimators=100, random_state=42))
            model.fit(X, y)
            return pd.DataFrame({
                'Feature': X.columns.tolist(),
                'Importance': model.feature_importances_.tolist()
            }).sort_values('Importance', ascending=False)
        except Exception:
            return None

    def detect_data_issues(self) -> list[str]:
        issues: list[str] = []
        missing_pct = _tof(self.df.isnull().sum().sum()) / (self.df.shape[0] * self.df.shape[1]) * 100
        if missing_pct > 5:
            issues.append(f"High missing values: {missing_pct:.2f}%")
        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1 and _tof(vc.max()) / _tof(vc.min()) > 10:
                issues.append(f"Class imbalance detected: ratio {_tof(vc.max()) / _tof(vc.min()):.2f}")
        for col in self.numeric_cols[:5]:
            skew = abs(_tof(self.df[col].skew()))
            if skew > 1:
                issues.append(f"High skewness in {col}: {_tof(self.df[col].skew()):.2f}")
        return issues

    def detect_problem_type(self, target_col: str | None = None) -> str:
        """Intelligently detect problem type based on data characteristics."""
        self.target_col = target_col
        
        # If target is specified, use it
        if target_col and target_col in self.df.columns:
            self.target_col = target_col
        elif target_col is None and self.numeric_cols:
            # Auto-detect: look for a column that looks like a target
            self.target_col = self._infer_target_column()
        
        if self.target_col and self.target_col in self.df.columns:
            unique_values = int(self.df[self.target_col].nunique())
            total_rows = len(self.df)
            dtype = str(self.df[self.target_col].dtype)
            
            # Classification detection rules
            if unique_values <= 20 and unique_values >= 2:
                # Check if it's binary or multiclass
                class_ratio = self.df[self.target_col].value_counts(normalize=True)
                if unique_values == 2:
                    self.problem_type = 'classification'
                elif unique_values <= 10 and class_ratio.min() > 0.01:
                    # Has reasonable class distribution
                    self.problem_type = 'classification'
                else:
                    self.problem_type = 'regression'
            elif unique_values > 20:
                # Could be regression (continuous values)
                # Check if values are discrete or continuous
                if self._is_categorical(self.target_col):
                    self.problem_type = 'classification'
                else:
                    self.problem_type = 'regression'
            else:
                self.problem_type = 'clustering'
        else:
            self.problem_type = 'clustering'
        
        return self.problem_type  # type: ignore[return-value]

    def _infer_target_column(self) -> str | None:
        """Infer the most likely target column based on data patterns."""
        if not self.numeric_cols:
            return None
        
        # Look for common target column patterns
        target_patterns = ['target', 'label', 'class', 'y', 'output', 'dependent', 'result', 'outcome']
        
        for col in self.df.columns:
            col_lower = col.lower()
            for pattern in target_patterns:
                if pattern in col_lower:
                    return col
        
        # If no pattern match, use the last numeric column as potential target
        # (often the last column is the target in datasets)
        return self.numeric_cols[-1] if self.numeric_cols else None

    def _is_categorical(self, col: str) -> bool:
        """Check if a column should be treated as categorical."""
        if col not in self.df.columns:
            return False
        unique_vals = self.df[col].nunique()
        total_rows = len(self.df)
        # If unique values are less than 5% of total rows, treat as categorical
        return unique_vals < max(10, total_rows * 0.05)

    def recommend_algorithms(self) -> list[dict]:
        """Generate data-driven ML algorithm recommendations based on dataset characteristics."""
        self.recommendations = []
        
        # Get dataset characteristics for smart recommendations
        n_samples = len(self.df)
        n_features = len(self.numeric_cols) + len(self.categorical_cols)
        has_missing = self.df.isnull().sum().sum() > 0
        has_outliers = self._detect_outliers_present()
        
        if self.problem_type == 'classification':
            # Get class distribution info
            if self.target_col and self.target_col in self.df.columns:
                class_dist = self.df[self.target_col].value_counts()
                n_classes = len(class_dist)
                is_imbalanced = class_dist.min() / class_dist.max() < 0.2 if len(class_dist) > 1 else False
            else:
                n_classes = 2
                is_imbalanced = False
            
            # Data-driven recommendations
            base_recommendations = []
            
            # Random Forest - always good starting point
            rf_desc = "Ensemble of decision trees for robust predictions"
            if n_features > 20:
                rf_desc += f" (optimized for high-dimensional data with {n_features} features)"
            base_recommendations.append({
                'algorithm': 'Random Forest', 
                'description': rf_desc,
                'pros': 'Handles overfitting, feature importance, works with messy data, robust to outliers', 
                'cons': 'Less interpretable, can be slow with many trees', 
                'use_when': 'General purpose — best starting point for most classification problems'
            })
            
            # Gradient Boosting - for accuracy
            gb_desc = "Sequential ensemble that corrects previous errors"
            if is_imbalanced:
                gb_desc += " (handles class imbalance well with proper tuning)"
            base_recommendations.append({
                'algorithm': 'Gradient Boosting', 
                'description': gb_desc,
                'pros': 'Highest accuracy, handles mixed data types, good with imbalanced classes', 
                'cons': 'Slower to train, requires tuning', 
                'use_when': 'Need maximum accuracy on complex datasets'
            })
            
            # Logistic Regression - for interpretability
            lr_desc = "Linear model for binary/multiclass classification"
            if n_features < 10:
                lr_desc += f" (ideal for {n_features} features with clear decision boundaries)"
            base_recommendations.append({
                'algorithm': 'Logistic Regression', 
                'description': lr_desc,
                'pros': 'Fast, interpretable, probabilistic output, works well with linearly separable data', 
                'cons': 'Assumes linear boundary, may underfit complex patterns', 
                'use_when': 'Need interpretable model or linearly separable data'
            })
            
            # SVM - for high dimensions
            if n_features > 15 or n_samples < 1000:
                svm_desc = "Finds optimal hyperplane to separate classes"
                if n_features > n_samples:
                    svm_desc += f" (effective in high-dimensional space with {n_features} features > {n_samples} samples)"
                base_recommendations.append({
                    'algorithm': 'SVM', 
                    'description': svm_desc,
                    'pros': 'Effective in high dimensions, good for small datasets', 
                    'cons': 'Slow on large datasets, requires feature scaling', 
                    'use_when': 'High-dimensional data, small to medium datasets'
                })
            
            # Decision Tree - for interpretability
            dt_desc = "Tree-based model splitting on feature values"
            if n_classes > 2:
                dt_desc += f" (natural fit for {n_classes}-class problem)"
            base_recommendations.append({
                'algorithm': 'Decision Tree', 
                'description': dt_desc,
                'pros': 'Highly interpretable, handles non-linear relationships', 
                'cons': 'Prone to overfitting, unstable with small data', 
                'use_when': 'Need full explainability or rule extraction'
            })
            
            # XGBoost - if available and data is suitable
            if n_samples > 100 and not is_imbalanced:
                base_recommendations.append({
                    'algorithm': 'XGBoost', 
                    'description': 'Extreme gradient boosting for maximum performance',
                    'pros': 'State-of-the-art accuracy, built-in regularization, handles missing values',
                    'cons': 'More complex tuning, may overfit on small data',
                    'use_when': 'Competitive ML tasks, need best accuracy'
                })
            
            self.recommendations = base_recommendations
            
        elif self.problem_type == 'regression':
            # Get regression-specific metrics
            target_std = 0
            if self.target_col and self.target_col in self.df.columns:
                target_std = float(self.df[self.target_col].std())
            
            base_recommendations = []
            
            # Gradient Boosting
            gb_desc = "Sequential boosting for regression"
            if target_std > 100:
                gb_desc += f" (handles wide range of target values with std={target_std:.2f})"
            base_recommendations.append({
                'algorithm': 'Gradient Boosting', 
                'description': gb_desc,
                'pros': 'Highest accuracy, handles outliers and non-linear relationships', 
                'cons': 'Slower training, requires tuning', 
                'use_when': 'Need maximum accuracy on complex regression problems'
            })
            
            # Random Forest
            rf_desc = "Ensemble of decision trees for regression"
            if n_features > 10:
                rf_desc += f" (handles {n_features} features well)"
            base_recommendations.append({
                'algorithm': 'Random Forest', 
                'description': rf_desc,
                'pros': 'Robust, handles complexity, provides feature importance', 
                'cons': 'Less interpretable than linear models', 
                'use_when': 'Complex non-linear relationships in data'
            })
            
            # Ridge Regression
            corr = self.compute_correlations()
            has_high_corr = False
            high_corr_pairs = 0
            if corr is not None:
                high_corr_pairs = ((corr.abs() > 0.8) & (corr.abs() < 1.0)).sum().sum() // 2
                has_high_corr = high_corr_pairs > 0
            
            ridge_desc = "L2 regularized linear regression"
            if has_high_corr:
                ridge_desc += f" (handles {high_corr_pairs} highly correlated feature pairs)"
            base_recommendations.append({
                'algorithm': 'Ridge Regression', 
                'description': ridge_desc,
                'pros': 'Prevents overfitting with correlated features, stable coefficients', 
                'cons': 'May underfit complex patterns', 
                'use_when': 'Multicollinearity present, need stable predictions'
            })
            
            # Linear Regression
            lr_desc = "Models linear relationships between features and target"
            if n_features < 5:
                lr_desc += f" (simple relationship with {n_features} features)"
            base_recommendations.append({
                'algorithm': 'Linear Regression', 
                'description': lr_desc,
                'pros': 'Simple, interpretable, fast, provides coefficient significance', 
                'cons': 'Assumes linearity, sensitive to outliers', 
                'use_when': 'Linear relationship exists, need interpretability'
            })
            
            # SVR
            if n_samples < 2000:
                svr_desc = "Support Vector Regression"
                if has_outliers:
                    svr_desc += " (robust to outliers)"
                base_recommendations.append({
                    'algorithm': 'SVR', 
                    'description': svr_desc,
                    'pros': 'Robust to outliers, effective in transformed feature space', 
                    'cons': 'Slow on large datasets, requires scaling', 
                    'use_when': 'Small to medium datasets, outliers present'
                })
            
            self.recommendations = base_recommendations
            
        elif self.problem_type == 'clustering':
            # Get clustering-specific metrics
            n_numeric = len(self.numeric_cols)
            
            base_recommendations = []
            
            # K-Means
            km_desc = "Partitions data into K clusters based on distance"
            if n_numeric > 5:
                km_desc += f" (works well with {n_numeric} numeric features)"
            base_recommendations.append({
                'algorithm': 'K-Means', 
                'description': km_desc,
                'pros': 'Simple, fast, scales well to large datasets', 
                'cons': 'Need to specify K, assumes spherical clusters', 
                'use_when': 'Known or estimated number of clusters, large datasets'
            })
            
            # DBSCAN
            dbscan_desc = "Density-based clustering algorithm"
            if has_outliers:
                dbscan_desc += " (naturally detects and handles outliers)"
            base_recommendations.append({
                'algorithm': 'DBSCAN', 
                'description': dbscan_desc,
                'pros': 'Finds arbitrary shaped clusters, detects outliers, no need to specify K', 
                'cons': 'Sensitive to parameters (eps, min_samples)', 
                'use_when': 'Unknown number of clusters, noisy data, arbitrary cluster shapes'
            })
            
            # Hierarchical
            if n_samples < 500:
                hier_desc = "Builds hierarchy of clusters using agglomerative or divisive approach"
                base_recommendations.append({
                    'algorithm': 'Hierarchical Clustering', 
                    'description': hier_desc,
                    'pros': 'No need to specify K, provides dendrogram for visualization', 
                    'cons': 'Expensive for large data, sensitive to noise', 
                    'use_when': 'Small datasets, need hierarchical structure, dendrogram analysis'
                })
            
            # Gaussian Mixture
            if n_samples > 50:
                gmm_desc = "Probabilistic clustering assuming mixture of Gaussian distributions"
                base_recommendations.append({
                    'algorithm': 'Gaussian Mixture Model', 
                    'description': gmm_desc,
                    'pros': 'Soft clustering, handles uncertainty, flexible cluster shapes', 
                    'cons': 'May converge to local optima, assumes Gaussian distributions', 
                    'use_when': 'Need soft cluster assignments, overlapping clusters'
                })
            
            self.recommendations = base_recommendations
        
        return self.recommendations

    def _detect_outliers_present(self) -> bool:
        """Detect if outliers are present in the dataset."""
        for col in self.numeric_cols[:5]:  # Check first 5 numeric columns
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                    if outliers > len(self.df) * 0.01:  # More than 1% outliers
                        return True
            except:
                pass
        return False

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        results: list[dict] = []

        if self.problem_type == 'classification' and self.target_col:
            X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
            y = self.df[self.target_col]
            if X.empty or len(y) < 10:
                return pd.DataFrame()
            try:
                strat = y if y.nunique() <= 20 else None
                X_train, X_test, y_train, y_test = train_test_split(
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
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state, max_features='sqrt'),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=random_state, learning_rate=0.05),
                'SVM': SVC(random_state=random_state, kernel='rbf', C=10, gamma='scale'),
            }
            for name, model in models.items():
                try:
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
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
                'Random Forest': RandomForestRegressor(n_estimators=200, random_state=random_state, max_features='sqrt'),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=random_state, learning_rate=0.05),
            }
            for name, model in reg_models.items():
                try:
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
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
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            for k in range(2, 9):
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
        self.insights.append(f"📊 **Dataset**: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        self.insights.append(f"🔢 **Numeric Features**: {len(self.numeric_cols)} | 📝 **Categorical**: {len(self.categorical_cols)}")
        if self.problem_type:
            self.insights.append(f"🎯 **Problem Type**: {self.problem_type.title()}")
        missing = int(self.df.isnull().sum().sum())
        self.insights.append(f"{'⚠️' if missing > 0 else '✅'} **Missing Values**: {missing}")
        dupes = int(self.df.duplicated().sum())
        self.insights.append(f"{'⚠️' if dupes > 0 else '✅'} **Duplicates**: {dupes}")
        for issue in self.detect_data_issues():
            self.insights.append(f"⚡ {issue}")
        if not self.detect_data_issues():
            self.insights.append("✅ **Data Quality**: No significant issues detected")
        return self.insights


def preprocess_data(df: pd.DataFrame, missing_strategy: str = 'auto', remove_outliers: bool = True, encode_method: str = 'label') -> 'DataAnalyzer':
    analyzer = DataAnalyzer(df)
    analyzer.analyze_types()
    analyzer.handle_missing_values(strategy=missing_strategy)
    analyzer.remove_duplicates()
    if remove_outliers:
        analyzer.handle_outliers(method='iqr')
    analyzer.encode_categorical(method=encode_method)
    return analyzer
