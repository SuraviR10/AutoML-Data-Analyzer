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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = []
        self.categorical_cols = []
        self.target_col = None
        self.problem_type = None
        self.recommendations = []
        self.insights = []
        self._feature_importances = None

    def analyze_types(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return self.numeric_cols, self.categorical_cols

    def handle_missing_values(self, strategy='auto'):
        missing_info = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {'count': missing_count, 'percentage': (missing_count / len(self.df)) * 100}
                if strategy == 'auto':
                    if col in self.numeric_cols:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
                elif strategy == 'mean' and col in self.numeric_cols:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and col in self.numeric_cols:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        return missing_info

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        return before - len(self.df)

    def handle_outliers(self, method='iqr'):
        outlier_info = {}
        for col in self.numeric_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)][col]
            if len(outliers) > 0:
                outlier_info[col] = {'count': len(outliers), 'lower_bound': lower, 'upper_bound': upper}
                if method == 'iqr':
                    self.df[col] = np.clip(self.df[col], lower, upper)
                elif method == 'remove':
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        return outlier_info

    def encode_categorical(self, method='label'):
        encoded_cols = {}
        for col in self.categorical_cols:
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                encoded_cols[col] = 'label'
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                encoded_cols[col] = 'onehot'
        return encoded_cols

    def compute_correlations(self):
        if len(self.numeric_cols) > 1:
            return self.df[self.numeric_cols].corr()
        return None

    def get_statistical_summary(self):
        """Extended statistical summary with skewness and kurtosis."""
        if not self.numeric_cols:
            return pd.DataFrame()
        stats = self.df[self.numeric_cols].describe().T
        stats['skewness'] = self.df[self.numeric_cols].skew()
        stats['kurtosis'] = self.df[self.numeric_cols].kurt()
        stats['missing_%'] = (self.df[self.numeric_cols].isnull().sum() / len(self.df) * 100).round(2)
        return stats.round(4)

    def compute_data_health_score(self):
        """Compute a unique 0-100 data health score."""
        score = 100.0
        penalties = {}

        # Missing values penalty (up to -30)
        missing_pct = self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100
        mp = min(missing_pct * 3, 30)
        score -= mp
        penalties['Missing Values'] = round(mp, 1)

        # Duplicates penalty (up to -20)
        dup_pct = self.df.duplicated().sum() / len(self.df) * 100
        dp = min(dup_pct * 2, 20)
        score -= dp
        penalties['Duplicates'] = round(dp, 1)

        # High skewness penalty (up to -20)
        skew_count = sum(1 for col in self.numeric_cols if abs(self.df[col].skew()) > 2)
        sp = min(skew_count * 4, 20)
        score -= sp
        penalties['High Skewness'] = round(sp, 1)

        # Class imbalance penalty (up to -15)
        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1:
                ratio = vc.max() / vc.min()
                ip = min((ratio - 1) * 1.5, 15)
                score -= ip
                penalties['Class Imbalance'] = round(ip, 1)

        # Low variance penalty (up to -15)
        low_var = sum(1 for col in self.numeric_cols if self.df[col].std() < 1e-6)
        lp = min(low_var * 5, 15)
        score -= lp
        penalties['Low Variance'] = round(lp, 1)

        return max(round(score, 1), 0), penalties

    def get_feature_engineering_tips(self):
        """Suggest feature engineering improvements."""
        tips = []
        for col in self.numeric_cols:
            skew = abs(self.df[col].skew())
            if skew > 2:
                tips.append(f"🔄 **{col}**: High skew ({skew:.2f}) — consider log/sqrt transform")
        if len(self.numeric_cols) >= 2:
            corr = self.df[self.numeric_cols].corr()
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.9:
                        tips.append(f"⚠️ **{corr.columns[i]}** & **{corr.columns[j]}**: Highly correlated ({corr.iloc[i,j]:.2f}) — consider dropping one")
        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1 and vc.max() / vc.min() > 5:
                tips.append("⚖️ **Target Column**: Class imbalance detected — consider SMOTE or class weighting")
        if not tips:
            tips.append("✅ No critical feature engineering issues detected!")
        return tips

    def get_feature_importance(self):
        """Get feature importances using Random Forest."""
        if self.target_col is None or self.target_col not in self.df.columns:
            return None
        X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
        y = self.df[self.target_col]
        if X.empty or len(X.columns) == 0:
            return None
        try:
            if self.problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            self._feature_importances = importance_df
            return importance_df
        except Exception:
            return None

    def detect_data_issues(self):
        issues = []
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        if missing_pct > 5:
            issues.append(f"High missing values: {missing_pct:.2f}%")
        if self.target_col and self.target_col in self.df.columns:
            vc = self.df[self.target_col].value_counts()
            if len(vc) > 1 and vc.max() / vc.min() > 10:
                issues.append(f"Class imbalance detected: ratio {vc.max()/vc.min():.2f}")
        for col in self.numeric_cols[:5]:
            skew = self.df[col].skew()
            if abs(skew) > 1:
                issues.append(f"High skewness in {col}: {skew:.2f}")
        return issues

    def detect_problem_type(self, target_col=None):
        self.target_col = target_col
        if target_col is None and self.numeric_cols:
            self.target_col = self.numeric_cols[-1]
        if self.target_col and self.target_col in self.df.columns:
            unique_values = self.df[self.target_col].nunique()
            dtype = self.df[self.target_col].dtype
            if unique_values <= 10 or dtype == 'object':
                self.problem_type = 'classification'
            elif unique_values > 10 and dtype in [np.int64, np.float64]:
                self.problem_type = 'regression'
            else:
                self.problem_type = 'classification'
        else:
            self.problem_type = 'clustering'
        return self.problem_type

    def recommend_algorithms(self):
        self.recommendations = []
        if self.problem_type == 'classification':
            self.recommendations = [
                {'algorithm': 'Logistic Regression', 'description': 'Best for binary classification with linearly separable data', 'pros': 'Fast, interpretable, works well with small datasets', 'cons': 'Assumes linear decision boundary', 'use_when': 'Data is linearly separable'},
                {'algorithm': 'Random Forest', 'description': 'Ensemble of decision trees for robust predictions', 'pros': 'Handles overfitting, works well with messy data, feature importance', 'cons': 'Less interpretable than single tree', 'use_when': 'General purpose, messy data'},
                {'algorithm': 'Gradient Boosting', 'description': 'Sequential ensemble that corrects previous errors', 'pros': 'High accuracy, handles mixed data types', 'cons': 'Slower to train, more hyperparameters', 'use_when': 'Need highest accuracy'},
                {'algorithm': 'SVM', 'description': 'Finds optimal hyperplane to separate classes', 'pros': 'Effective in high dimensions, robust to outliers', 'cons': 'Slow on large datasets', 'use_when': 'High-dimensional data'},
                {'algorithm': 'Decision Tree', 'description': 'Tree-based model splitting on feature values', 'pros': 'Easy to interpret, handles non-linear relationships', 'cons': 'Prone to overfitting', 'use_when': 'Need explainability'},
            ]
        elif self.problem_type == 'regression':
            self.recommendations = [
                {'algorithm': 'Linear Regression', 'description': 'Models linear relationships between features and target', 'pros': 'Simple, interpretable, fast', 'cons': 'Assumes linear relationships', 'use_when': 'Linear relationship exists'},
                {'algorithm': 'Ridge Regression', 'description': 'L2 regularized linear regression', 'pros': 'Prevents overfitting with correlated features', 'cons': 'May underfit with too much regularization', 'use_when': 'Multicollinearity present'},
                {'algorithm': 'Random Forest Regressor', 'description': 'Ensemble of decision trees for regression', 'pros': 'Robust, handles complexity, feature importance', 'cons': 'Less interpretable', 'use_when': 'Complex non-linear data'},
                {'algorithm': 'Gradient Boosting Regressor', 'description': 'Sequential boosting for regression', 'pros': 'High accuracy, handles outliers well', 'cons': 'Slower training', 'use_when': 'Need highest accuracy'},
                {'algorithm': 'SVR', 'description': 'Support Vector Regression', 'pros': 'Robust to outliers, works in high dimensions', 'cons': 'Slow on large datasets', 'use_when': 'Small-medium datasets'},
            ]
        elif self.problem_type == 'clustering':
            self.recommendations = [
                {'algorithm': 'K-Means', 'description': 'Partitions data into K clusters based on distance', 'pros': 'Simple, fast, scales well', 'cons': 'Assumes spherical clusters, need to specify K', 'use_when': 'Known number of clusters'},
                {'algorithm': 'DBSCAN', 'description': 'Density-based clustering algorithm', 'pros': 'Finds arbitrary shaped clusters, detects outliers', 'cons': 'Sensitive to epsilon and min_samples', 'use_when': 'Unknown clusters, noisy data'},
                {'algorithm': 'Hierarchical Clustering', 'description': 'Builds hierarchy of clusters (dendrogram)', 'pros': 'No need to specify K, visualizable', 'cons': 'Computationally expensive for large data', 'use_when': 'Small datasets, need hierarchy'},
            ]
        return self.recommendations

    def train_and_evaluate(self, test_size=0.2, random_state=42):
        results = []
        if self.problem_type == 'classification' and self.target_col:
            X = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number])
            y = self.df[self.target_col]
            if X.empty or len(y) < 10:
                return pd.DataFrame()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            models = {
                'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                'SVM': SVC(random_state=random_state, probability=True),
            }
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results.append({
                        'Model': name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
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
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            }
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results.append({
                        'Model': name,
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'R2-Score': r2_score(y_test, y_pred)
                    })
                except Exception:
                    pass

        elif self.problem_type == 'clustering':
            X = self.df.select_dtypes(include=[np.number])
            for k in range(2, 8):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    labels = kmeans.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        results.append({'K': k, 'Silhouette Score': score, 'Inertia': kmeans.inertia_})
                except Exception:
                    pass

        return pd.DataFrame(results)

    def generate_insights(self):
        self.insights = []
        self.insights.append(f"📊 **Dataset**: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        self.insights.append(f"🔢 **Numeric Features**: {len(self.numeric_cols)} | 📝 **Categorical**: {len(self.categorical_cols)}")
        if self.problem_type:
            self.insights.append(f"🎯 **Problem Type**: {self.problem_type.title()}")
        missing = self.df.isnull().sum().sum()
        self.insights.append(f"{'⚠️' if missing > 0 else '✅'} **Missing Values**: {missing}")
        dupes = self.df.duplicated().sum()
        self.insights.append(f"{'⚠️' if dupes > 0 else '✅'} **Duplicates**: {dupes}")
        issues = self.detect_data_issues()
        for issue in issues:
            self.insights.append(f"⚡ {issue}")
        if not issues:
            self.insights.append("✅ **Data Quality**: No significant issues detected")
        return self.insights


def preprocess_data(df, missing_strategy='auto', remove_outliers=True, encode_method='label'):
    analyzer = DataAnalyzer(df)
    analyzer.analyze_types()
    analyzer.handle_missing_values(strategy=missing_strategy)
    analyzer.remove_duplicates()
    if remove_outliers:
        analyzer.handle_outliers(method='iqr')
    analyzer.encode_categorical(method=encode_method)
    return analyzer
