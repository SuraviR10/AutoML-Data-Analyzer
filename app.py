"""
Auto Data Analyzer & ML Recommender - Streamlit UI
====================================================
Interactive web interface for the ML pipeline system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_analyzer import DataAnalyzer, preprocess_data
import io

# Page configuration
st.set_page_config(
    page_title="Auto Data Analyzer & ML Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced with beautiful color combinations
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e94560 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid #e94560;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-testid="stTab"] {
        background-color: transparent;
        color: #ffffff;
    }
    
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        border-radius: 10px 10px 0 0;
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e94560;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
    }
    
    div[data-testid="stMetric"] label {
        color: #a0a0a0 !important;
    }
    
    div[data-testid="stMetric"] div {
        color: #ffffff !important;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 2px solid #e94560;
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%);
        border: 2px dashed #e94560;
        border-radius: 15px;
        padding: 30px;
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background-color: #1a1a2e;
        border: 1px solid #e94560;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Checkboxes */
    .stCheckbox>label {
        color: #ffffff !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%);
        border-radius: 10px;
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Success */
    [data-testid="stSuccess"] {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        border-radius: 10px;
    }
    
    /* Warning */
    [data-testid="stWarning"] {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        border-radius: 10px;
    }
    
    /* Error */
    [data-testid="stError"] {
        background: linear-gradient(135deg, #d63031 0%, #e84393 100%);
        border-radius: 10px;
    }
    
    /* Info */
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, #0984e3 0%, #6c5ce7 100%);
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #e94560;
        border-width: 2px;
    }
    
    /* Text colors */
    .stMarkdown p {
        color: #dfe6e9;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 4px;
    }
    
    /* Plot containers */
    .plot-container {
        background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e94560;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.title("🤖 Auto Data Analyzer & ML Recommender")
    st.markdown("### Intelligent ML Pipeline Automation System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Upload Your Dataset")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your dataset in CSV or Excel format"
    )
    
    # Preprocessing options
    st.sidebar.markdown("### Preprocessing Options")
    missing_strategy = st.sidebar.selectbox(
        "Missing Value Strategy",
        ['auto', 'mean', 'median', 'mode'],
        index=0,
        help="Strategy to handle missing values"
    )
    
    remove_outliers = st.sidebar.checkbox(
        "Remove Outliers",
        value=True,
        help="Remove or cap outliers using IQR method"
    )
    
    encode_method = st.sidebar.selectbox(
        "Categorical Encoding",
        ['label', 'onehot'],
        index=0,
        help="Method to encode categorical features"
    )
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store original for comparison
            df_original = df.copy()
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Dataset Overview",
                "🧹 Preprocessing",
                "📈 EDA & Visualizations",
                "🤖 ML Recommendations",
                "⚡ Model Evaluation"
            ])
            
            # ==================== TAB 1: Dataset Overview ====================
            with tab1:
                st.header("📊 Dataset Overview")
                
                # Dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                st.markdown("### 🔍 First 10 Rows")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column info
                st.markdown("### 📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null': df.count().values,
                    'Null Values': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
                # Data types summary
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🔢 **Numeric Columns**: {len(numeric_cols)}")
                    if numeric_cols:
                        st.write(", ".join(numeric_cols[:5]) + ("..." if len(numeric_cols) > 5 else ""))
                with col2:
                    st.info(f"📝 **Categorical Columns**: {len(categorical_cols)}")
                    if categorical_cols:
                        st.write(", ".join(categorical_cols[:5]) + ("..." if len(categorical_cols) > 5 else ""))
            
            # ==================== TAB 2: Preprocessing ====================
            with tab2:
                st.header("🧹 Data Preprocessing")
                
                # Run preprocessing
                analyzer = preprocess_data(
                    df.copy(),
                    missing_strategy=missing_strategy,
                    remove_outliers=remove_outliers,
                    encode_method=encode_method
                )
                
                # Show preprocessing results
                st.markdown("### ✅ Preprocessing Results")
                
                # Missing values
                missing_before = df_original.isnull().sum().sum()
                missing_after = analyzer.df.isnull().sum().sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing Values (Before)", missing_before)
                with col2:
                    st.metric("Missing Values (After)", missing_after)
                with col3:
                    st.metric("Duplicates Removed", df_original.duplicated().sum())
                
                # Shape after preprocessing
                st.markdown("### 📐 Dataset Shape After Preprocessing")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", analyzer.df.shape[0])
                with col2:
                    st.metric("Columns", analyzer.df.shape[1])
                
                # Preview cleaned data
                st.markdown("### 👀 Cleaned Data Preview")
                st.dataframe(analyzer.df.head(10), use_container_width=True)
                
                # Download cleaned data
                csv = analyzer.df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            
            # ==================== TAB 3: EDA & Visualizations ====================
            with tab3:
                st.header("📈 Exploratory Data Analysis & Visualizations")
                
                # Re-analyze with cleaned data
                analyzer = DataAnalyzer(analyzer.df)
                analyzer.analyze_types()
                
                # Visualization options
                viz_type = st.selectbox(
                    "Select Visualization Type",
                    ["Histograms", "Box Plots", "Correlation Heatmap", "Count Plots"]
                )
                
                if viz_type == "Histograms":
                    st.markdown("### 📊 Distribution of Numeric Features")
                    cols = st.columns(3)
                    for i, col in enumerate(analyzer.numeric_cols[:9]):
                        with cols[i % 3]:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            analyzer.df[col].hist(bins=20, ax=ax, color='#e94560', edgecolor='white', alpha=0.8)
                            ax.set_title(f"{col}", fontsize=12, fontweight='bold', color='#ffffff')
                            ax.set_xlabel("Value", fontsize=10, color='#a0a0a0')
                            ax.set_ylabel("Frequency", fontsize=10, color='#a0a0a0')
                            ax.tick_params(colors='#a0a0a0')
                            ax.set_facecolor('#1a1a2e')
                            fig.patch.set_facecolor('#1a1a2e')
                            st.pyplot(fig)
                
                elif viz_type == "Box Plots":
                    st.markdown("### 📦 Box Plots for Outlier Detection")
                    cols = st.columns(2)
                    for i, col in enumerate(analyzer.numeric_cols[:6]):
                        with cols[i % 2]:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            bp = analyzer.df.boxplot(column=[col], ax=ax, patch_artist=True)
                            for patch in bp.patches:
                                patch.set_facecolor('#8b5cf6')
                                patch.set_alpha(0.7)
                            ax.set_title(f"{col}", fontsize=12, fontweight='bold', color='#ffffff')
                            ax.tick_params(colors='#a0a0a0')
                            ax.set_facecolor('#1a1a2e')
                            fig.patch.set_facecolor('#1a1a2e')
                            st.pyplot(fig)
                
                elif viz_type == "Correlation Heatmap":
                    st.markdown("### 🔥 Correlation Matrix")
                    if len(analyzer.numeric_cols) > 1:
                        corr_matrix = analyzer.df[analyzer.numeric_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax, 
                                   square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                                   annot_kws={'color': '#ffffff', 'fontsize': 10})
                        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', color='#ffffff')
                        ax.tick_params(colors='#a0a0a0')
                        for text in ax.get_xticklabels():
                            text.set_color('#ffffff')
                        for text in ax.get_yticklabels():
                            text.set_color('#ffffff')
                        fig.patch.set_facecolor('#1a1a2e')
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough numeric columns for correlation analysis")
                
                elif viz_type == "Count Plots":
                    st.markdown("### 📊 Categorical Feature Distributions")
                    if analyzer.categorical_cols:
                        cat_col = st.selectbox("Select Categorical Column", analyzer.categorical_cols)
                        if cat_col:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            counts = analyzer.df[cat_col].value_counts().head(10)
                            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
                            bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=1.5)
                            ax.set_xticks(range(len(counts)))
                            ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10, color='#ffffff')
                            ax.set_title(f"Distribution of {cat_col}", fontsize=14, fontweight='bold', color='#ffffff')
                            ax.set_xlabel(cat_col, fontsize=11, color='#a0a0a0')
                            ax.set_ylabel("Count", fontsize=11, color='#a0a0a0')
                            ax.tick_params(colors='#a0a0a0')
                            ax.grid(True, alpha=0.3, axis='y', color='#e94560')
                            ax.set_facecolor('#1a1a2e')
                            fig.patch.set_facecolor('#1a1a2e')
                            st.pyplot(fig)
                    else:
                        st.info("No categorical columns found")
                
                # Data issues
                st.markdown("### ⚡ Data Quality Issues")
                issues = analyzer.detect_data_issues()
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    st.success("No significant data quality issues detected!")
            
            # ==================== TAB 4: ML Recommendations ====================
            with tab4:
                st.header("🤖 ML Recommendation Engine")
                
                # Target column selection
                target_col = st.selectbox(
                    "Select Target Column",
                    options=['Auto-Detect'] + analyzer.numeric_cols,
                    index=0
                )
                
                if target_col == 'Auto-Detect':
                    target = None
                else:
                    target = target_col
                
                # Detect problem type
                problem_type = analyzer.detect_problem_type(target_col=target)
                
                st.markdown(f"### 🎯 Detected Problem Type: **{problem_type.title()}**")
                
                # Get recommendations
                recommendations = analyzer.recommend_algorithms()
                
                st.markdown("### 💡 Recommended Algorithms")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['algorithm']}", expanded=True):
                        st.markdown(f"**Description**: {rec['description']}")
                        st.markdown(f"**✅ Pros**: {rec['pros']}")
                        st.markdown(f"**❌ Cons**: {rec['cons']}")
                
                # Algorithm selection for evaluation
                st.markdown("### 🏃 Train & Compare Models")
                st.markdown("Select algorithms to train and evaluate:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    train_lr = st.checkbox("Logistic/Linear Regression", value=True)
                with col2:
                    train_dt = st.checkbox("Decision Tree", value=True)
                with col3:
                    train_rf = st.checkbox("Random Forest", value=True)
                
                evaluate_btn = st.button("🚀 Train & Evaluate Selected Models", type="primary")
                
                if evaluate_btn:
                    st.session_state['evaluate'] = True
                    st.session_state['train_models'] = {
                        'lr': train_lr,
                        'dt': train_dt,
                        'rf': train_rf
                    }
            
            # ==================== TAB 5: Model Evaluation ====================
            with tab5:
                st.header("⚡ Model Evaluation & Comparison")
                
                # Run evaluation
                results = analyzer.train_and_evaluate()
                
                if not results.empty:
                    st.markdown("### 📊 Performance Metrics")
                    
                    if analyzer.problem_type == 'classification':
                        # Format classification results
                        results_display = results.copy()
                        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                            results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}")
                        st.dataframe(results_display, use_container_width=True)
                        
                        # Best model
                        best_idx = results['Accuracy'].idxmax()
                        best_model = results.loc[best_idx, 'Model']
                        best_score = results.loc[best_idx, 'Accuracy']
                        
                        st.success(f"🏆 **Best Model**: {best_model} with {best_score:.4f} accuracy")
                        
                        # Visualization
                        st.markdown("### 📈 Model Comparison Chart")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        x = np.arange(len(results))
                        width = 0.2
                        colors = ['#e94560', '#8b5cf6', '#06b6d4', '#10b981']
                        
                        for i, metric in enumerate(metrics):
                            ax.bar(x + i*width, results[metric], width, label=metric, color=colors[i])
                        
                        ax.set_xlabel('Model', fontsize=12, color='#a0a0a0')
                        ax.set_ylabel('Score', fontsize=12, color='#a0a0a0')
                        ax.set_title('Classification Model Performance Comparison', fontsize=14, fontweight='bold', color='#ffffff')
                        ax.set_xticks(x + width * 1.5)
                        ax.set_xticklabels(results['Model'], rotation=45, color='#ffffff')
                        ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#e94560', labelcolor='#ffffff')
                        ax.set_ylim(0, 1)
                        ax.set_facecolor('#1a1a2e')
                        ax.tick_params(colors='#a0a0a0')
                        ax.grid(True, alpha=0.3, axis='y', color='#e94560')
                        fig.patch.set_facecolor('#1a1a2e')
                        st.pyplot(fig)
                    
                    elif analyzer.problem_type == 'regression':
                        # Format regression results
                        results_display = results.copy()
                        results_display['RMSE'] = results_display['RMSE'].apply(lambda x: f"{x:.4f}")
                        results_display['R2-Score'] = results_display['R2-Score'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(results_display, use_container_width=True)
                        
                        # Best model
                        best_idx = results['R2-Score'].idxmax()
                        best_model = results.loc[best_idx, 'Model']
                        best_score = results.loc[best_idx, 'R2-Score']
                        
                        st.success(f"🏆 **Best Model**: {best_model} with R² = {best_score:.4f}")
                        
                        # Visualization
                        st.markdown("### 📈 Model Comparison Chart")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        x = np.arange(len(results))
                        width = 0.35
                        colors1 = ['#10b981', '#06b6d4', '#f59e0b']
                        colors2 = ['#e94560', '#8b5cf6', '#ec4899']
                        
                        ax.bar(x - width/2, results['R2-Score'], width, label='R² Score', color=colors1)
                        ax.bar(x + width/2, results['RMSE'] / results['RMSE'].max(), width, label='RMSE (normalized)', color=colors2)
                        
                        ax.set_xlabel('Model', fontsize=12, color='#a0a0a0')
                        ax.set_ylabel('Score', fontsize=12, color='#a0a0a0')
                        ax.set_title('Regression Model Performance Comparison', fontsize=14, fontweight='bold', color='#ffffff')
                        ax.set_xticks(x)
                        ax.set_xticklabels(results['Model'], rotation=45, color='#ffffff')
                        ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#e94560', labelcolor='#ffffff')
                        ax.set_facecolor('#1a1a2e')
                        ax.tick_params(colors='#a0a0a0')
                        ax.grid(True, alpha=0.3, axis='y', color='#e94560')
                        fig.patch.set_facecolor('#1a1a2e')
                        st.pyplot(fig)
                    
                    elif analyzer.problem_type == 'clustering':
                        st.dataframe(results, use_container_width=True)
                        
                        # Best K
                        best_idx = results['Silhouette Score'].idxmax()
                        best_k = results.loc[best_idx, 'K']
                        best_score = results.loc[best_idx, 'Silhouette Score']
                        
                        st.success(f"🏆 **Optimal Number of Clusters**: K = {best_k} with silhouette score = {best_score:.4f}")
                        
                        # Visualization
                        st.markdown("### 📈 Silhouette Score by K")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(results['K'], results['Silhouette Score'], marker='o', linewidth=3, markersize=12, color='#8b5cf6')
                        ax.set_xlabel('Number of Clusters (K)', fontsize=12, color='#a0a0a0')
                        ax.set_ylabel('Silhouette Score', fontsize=12, color='#a0a0a0')
                        ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold', color='#ffffff')
                        ax.grid(True, alpha=0.3, color='#e94560')
                        ax.set_facecolor('#1a1a2e')
                        ax.tick_params(colors='#a0a0a0')
                        fig.patch.set_facecolor('#1a1a2e')
                        st.pyplot(fig)
                else:
                    st.info("👆 Please go to ML Recommendations tab and select a target column to enable model evaluation.")
            
            # ==================== Summary Section ====================
            st.markdown("---")
            st.markdown("### 📋 Analysis Summary")
            
            # Generate insights
            insights = analyzer.generate_insights()
            
            for insight in insights:
                if "⚠️" in insight:
                    st.warning(insight)
                elif "✅" in insight:
                    st.success(insight)
                elif "📊" in insight:
                    st.info(insight)
                else:
                    st.write(insight)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("""
            ### 💡 Troubleshooting
            - Make sure your file is in CSV or Excel format
            - Check that the file is not corrupted
            - Ensure the file has proper headers
            """)
    
    else:
        # Welcome screen when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 30px;'>
            <h1 style='font-size: 48px; margin-bottom: 20px;'>🤖</h1>
            <h2 style='color: #e94560; font-size: 36px; margin-bottom: 10px;'>Auto Data Analyzer</h2>
            <h3 style='color: #a0a0a0; font-size: 20px;'>Intelligent ML Pipeline Automation System</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #e94560; text-align: center;'>
                <h3 style='color: #e94560;'>📥</h3>
                <h4 style='color: #ffffff;'>Load Data</h4>
                <p style='color: #a0a0a0;'>CSV & Excel files</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #8b5cf6; text-align: center;'>
                <h3 style='color: #8b5cf6;'>🧹</h3>
                <h4 style='color: #ffffff;'>Preprocess</h4>
                <p style='color: #a0a0a0;'>Clean & Transform</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #06b6d4; text-align: center;'>
                <h3 style='color: #06b6d4;'>📊</h3>
                <h4 style='color: #ffffff;'>Analyze</h4>
                <p style='color: #a0a0a0;'>EDA & Insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #10b981; text-align: center;'>
                <h3 style='color: #10b981;'>📈</h3>
                <h4 style='color: #ffffff;'>Visualize</h4>
                <p style='color: #a0a0a0;'>Charts & Graphs</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #f59e0b; text-align: center;'>
                <h3 style='color: #f59e0b;'>🤖</h3>
                <h4 style='color: #ffffff;'>ML Recommend</h4>
                <p style='color: #a0a0a0;'>Smart Suggestions</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #1a1a2e 100%); padding: 20px; border-radius: 15px; border: 1px solid #ec4899; text-align: center;'>
                <h3 style='color: #ec4899;'>⚡</h3>
                <h4 style='color: #ffffff;'>Evaluate</h4>
                <p style='color: #a0a0a0;'>Train & Compare</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started", unsafe_allow_html=True)
        
        st.markdown("""
        <ol style='color: #dfe6e9; font-size: 16px; line-height: 2;'>
            <li>Upload your dataset using the sidebar</li>
            <li>Configure preprocessing options</li>
            <li>Explore different tabs for insights</li>
            <li>Get ML recommendations</li>
            <li>Evaluate and compare models</li>
        </ol>
        """, unsafe_allow_html=True)
        
        # Sample dataset info
        st.markdown("### 📝 Sample Dataset Format", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color: #a0a0a0;'>
            <li>Have clear column headers</li>
            <li>Include both numeric and categorical features (if available)</li>
            <li>Have a target column for supervised learning (optional)</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Show example
        example_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Salary': [50000, 60000, 75000, 80000, 90000],
            'Department': ['IT', 'HR', 'Sales', 'IT', 'Marketing'],
            'Target': [1, 0, 1, 0, 1]
        })
        st.dataframe(example_data, use_container_width=True)


if __name__ == "__main__":
    main()