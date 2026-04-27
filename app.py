"""
AutoML - Data Analyzer & ML Recommender
Modern Dashboard Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_analyzer import DataAnalyzer, preprocess_data

# Page Config
st.set_page_config(
    page_title="AutoML - Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif !important; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    h1, h2, h3 { color: #ffffff !important; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.4);
    }
    
    .feature-card {
        background: rgba(30, 41, 59, 0.6);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin: 10px 0;
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Plot Config
PLOT_BG = "#0f172a"
PLOT_FG = "#e2e8f0"
PALETTE = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4"]

def style_plot(ax, title=""):
    ax.set_facecolor(PLOT_BG)
    ax.figure.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_FG)
    if title:
        ax.set_title(title, color=PLOT_FG, fontsize=12, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.grid(True, alpha=0.2, color='#334155')

def main():
    # Header
    st.markdown("<h1 style='text-align:center;'>🤖 AutoML</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8;'>Data Analyzer & ML Recommender</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Upload Dataset")
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        missing_strategy = st.selectbox("Missing Values", ['auto', 'mean', 'median', 'mode'])
        remove_outliers = st.checkbox("Remove Outliers", value=True)
        encode_method = st.selectbox("Encoding", ['label', 'onehot'])
    
    if not uploaded_file:
        render_welcome()
        return
    
    # Load Data
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_original = df.copy()
        
        # Preprocess
        cache_key = f"analyzer_{uploaded_file.name}_{missing_strategy}_{remove_outliers}_{encode_method}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = preprocess_data(df.copy(), missing_strategy, remove_outliers, encode_method)
        analyzer = st.session_state[cache_key]
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "⚙️ Preprocessing", "📈 EDA & Visualizations", 
            "🤖 ML Recommendations", "🏆 Model Evaluation", "📥 Export Results"
        ])
        
        with tab1:
            render_overview(df, df_original, analyzer, uploaded_file)
        
        with tab2:
            render_preprocessing(df_original, analyzer)
        
        with tab3:
            render_eda(analyzer)
        
        with tab4:
            render_ml_recommendations(df, analyzer)
        
        with tab5:
            render_model_evaluation(analyzer)
        
        with tab6:
            render_export(analyzer)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

def render_welcome():
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <h2>Welcome to AutoML Dashboard</h2>
        <p style='color:#94a3b8;font-size:18px;'>Upload a dataset to begin automated analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align:center;color:#ffffff;margin-top:40px;'>Key Features</h3>", unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    features = [
        ("📥", "Dataset Overview", "Comprehensive metrics & quality score"),
        ("🧹", "Data Preprocessing", "Handle missing values, duplicates, outliers"),
        ("📊", "EDA & Visualizations", "Interactive plots for insights"),
        ("🤖", "ML Recommendation", "Smart algorithm suggestions"),
        ("🏆", "Model Evaluation", "Automated training & comparison"),
        ("⚡", "Export Results", "Download cleaned data & reports")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='feature-card'>
                <div style='font-size:40px;'>{icon}</div>
                <h4 style='color:#6366f1;margin:12px 0;'>{title}</h4>
                <p style='color:#94a3b8;margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def render_overview(df, df_original, analyzer, uploaded_file):
    st.markdown("### 📊 Dataset Overview")
    
    # Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))
    c5.metric("File Size", f"{uploaded_file.size/1024:.1f} KB")
    
    # Detect problem type
    analyzer_temp = DataAnalyzer(analyzer.df)
    analyzer_temp.analyze_types()
    problem_type = analyzer_temp.detect_problem_type()
    c6.metric("Problem Type", problem_type.title() if problem_type else "Unknown")
    
    st.markdown("---")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 Feature Types Distribution")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [len(numeric_cols), len(categorical_cols)]
        labels = [f'Numeric\n{len(numeric_cols)}', f'Categorical\n{len(categorical_cols)}']
        colors = [PALETTE[0], PALETTE[2]]
        
        pie_result = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, textprops={'color': PLOT_FG, 'fontsize': 11})
        if len(pie_result) == 3:
            wedges, texts, autotexts = pie_result
        else:
            wedges, texts = pie_result
            autotexts = []
        ax.set_facecolor(PLOT_BG)
        fig.patch.set_facecolor(PLOT_BG)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### 📊 Data Quality Score")
        
        # Calculate quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
        dup_pct = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
        quality_score = max(0, 100 - missing_pct - dup_pct)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([quality_score, 100-quality_score], colors=[PALETTE[4], '#1e293b'], 
               startangle=90, counterclock=False)
        from matplotlib.patches import Circle
        circle = Circle((0, 0), 0.7, color=PLOT_BG)
        ax.add_artist(circle)
        ax.text(0, 0, f'{quality_score:.0f}\n/100', ha='center', va='center', 
            fontsize=24, color=PLOT_FG, fontweight='bold')
        ax.text(0, -0.3, 'Excellent' if quality_score > 90 else 'Good' if quality_score > 70 else 'Fair',
            ha='center', va='center', fontsize=12, color=PALETTE[4])
        ax.set_facecolor(PLOT_BG)
        fig.patch.set_facecolor(PLOT_BG)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Dataset Preview
    st.markdown("#### 👀 Dataset Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                    ax=ax, cbar_kws={'label': 'Correlation'}, 
                    annot_kws={'color': PLOT_FG, 'fontsize': 9})
        style_plot(ax, "Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close()

def render_preprocessing(df_original, analyzer):
    st.markdown("### ⚙️ Data Preprocessing")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    missing_before = int(df_original.isnull().sum().sum())
    missing_after = int(analyzer.df.isnull().sum().sum())
    dupes = int(df_original.duplicated().sum())
    
    c1.metric("Missing (Before)", f"{missing_before:,}")
    c2.metric("Missing (After)", f"{missing_after:,}", delta=f"-{missing_before-missing_after}")
    c3.metric("Duplicates Removed", f"{dupes:,}")
    c4.metric("Rows After Cleaning", f"{analyzer.df.shape[0]:,}")
    
    st.markdown("---")
    
    # Cleaning Steps
    st.markdown("#### ✅ Cleaning Steps Applied")
    for log in analyzer.preprocessing_log:
        st.success(log)
    
    st.markdown("---")
    
    # Cleaned Preview
    st.markdown("#### 👀 Cleaned Data Preview")
    st.dataframe(analyzer.df.head(10), use_container_width=True)

def render_eda(analyzer):
    st.markdown("### 📈 Exploratory Data Analysis")
    
    eda_analyzer = DataAnalyzer(analyzer.df)
    eda_analyzer.analyze_types()
    
    st.markdown("#### Interactive Visualizations")
    
    col_viz_type, col_viz_col = st.columns([1, 2])
    
    with col_viz_type:
        viz_type = st.selectbox("📊 Select Visualization Type", 
                                ["Distribution Plot", "Box Plot", "Count Plot", "Scatter Plot"])
    
    if viz_type == "Distribution Plot":
        if eda_analyzer.numeric_cols:
            with col_viz_col:
                selected_col = st.selectbox("Select Numeric Column", eda_analyzer.numeric_cols, key="dist_plot_col")
            if selected_col:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(eda_analyzer.df[selected_col].dropna(), bins=20, kde=True, 
                             color=PALETTE[0], edgecolor='white', ax=ax)
                style_plot(ax, f"Distribution of {selected_col}")
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No numeric columns for distribution plots.")
    
    elif viz_type == "Box Plot":
        if eda_analyzer.numeric_cols:
            with col_viz_col:
                selected_col = st.selectbox("Select Numeric Column", eda_analyzer.numeric_cols, key="box_plot_col")
            if selected_col:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(y=eda_analyzer.df[selected_col].dropna(), color=PALETTE[1], ax=ax)
                style_plot(ax, f"Box Plot of {selected_col}")
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No numeric columns for box plots.")
    
    elif viz_type == "Count Plot":
        if eda_analyzer.categorical_cols:
            with col_viz_col:
                selected_col = st.selectbox("Select Categorical Column", eda_analyzer.categorical_cols, key="count_plot_col")
            if selected_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=eda_analyzer.df[selected_col], order=eda_analyzer.df[selected_col].value_counts().index[:10], 
                              palette=PALETTE, ax=ax)
                style_plot(ax, f"Count Plot of {selected_col}")
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No categorical columns for count plots.")
            
    elif viz_type == "Scatter Plot":
        if len(eda_analyzer.numeric_cols) >= 2:
            with col_viz_col:
                col_x, col_y = st.columns(2)
                with col_x:
                    x_axis = st.selectbox("X-axis", eda_analyzer.numeric_cols, key="scatter_x")
                with col_y:
                    y_axis = st.selectbox("Y-axis", eda_analyzer.numeric_cols, key="scatter_y")
            
            if x_axis and y_axis:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=eda_analyzer.df[x_axis], y=eda_analyzer.df[y_axis], 
                                color=PALETTE[2], alpha=0.7, ax=ax)
                style_plot(ax, f"Scatter Plot: {x_axis} vs {y_axis}")
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Need at least two numeric columns for scatter plots.")

def render_ml_recommendations(df, analyzer):
    st.markdown("### 🤖 ML Recommendation Engine")
    
    # Target selection
    target_col = st.selectbox("🎯 Select Target Column", 
                              ['Auto-Detect'] + df.columns.tolist(),
                              index=0 if st.session_state.get('selected_target') is None else (df.columns.tolist().index(st.session_state['selected_target']) + 1 if st.session_state['selected_target'] in df.columns else 0)
                              )
    target = None if target_col == 'Auto-Detect' else target_col
    st.session_state['selected_target'] = target
    
    # Detect problem type
    rec_analyzer = DataAnalyzer(analyzer.df)
    rec_analyzer.analyze_types()
    problem_type = rec_analyzer.detect_problem_type(target_col=target)
    
    # Display problem type
    colors = {'classification': '#10b981', 'regression': '#6366f1', 'clustering': '#8b5cf6'}
    color = colors.get(problem_type, '#94a3b8')
    
    st.markdown(f"""
    <div style='background:rgba(30,41,59,0.6);border-left:4px solid {color};
                padding:16px;border-radius:8px;margin:16px 0;'>
        <strong style='color:{color};font-size:18px;'>🎯 {problem_type.title() if problem_type else 'Unknown'}</strong>
        <span style='color:#94a3b8;margin-left:10px;'>Problem Detected</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    recommendations = rec_analyzer.recommend_algorithms()
    
    st.markdown("#### 💡 Recommended Algorithms")
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['algorithm']}", expanded=(i==1)):
            st.markdown(f"**📋 Description:** {rec['description']}")
            st.markdown(f"**✅ Pros:** {rec['pros']}")
            st.markdown(f"**❌ Cons:** {rec['cons']}")
            st.markdown(f"**🎯 Use When:** {rec['use_when']}")

def render_model_evaluation(analyzer):
    st.markdown("### 🏆 Model Evaluation")
    
    target = st.session_state.get('selected_target')
    
    eval_analyzer = DataAnalyzer(analyzer.df)
    eval_analyzer.analyze_types()
    eval_analyzer.detect_problem_type(target_col=target)
    
    if not eval_analyzer.target_col:
        st.info("👆 Select a target column in ML Recommendations tab first")
        return
    
    st.markdown(f"**Target:** {eval_analyzer.target_col} | **Type:** {eval_analyzer.problem_type}")
    
    with st.spinner("Training models..."):
        results = eval_analyzer.train_and_evaluate()
    
    if results.empty:
        st.warning("Not enough data to train models")
        return
    
    st.markdown("#### 📊 Performance Metrics")
    st.dataframe(results, use_container_width=True)
    
    # Best model
    if eval_analyzer.problem_type == 'classification':
        best_idx = results['Accuracy'].idxmax()
        best_model = results.loc[best_idx, 'Model']
        best_score = results.loc[best_idx, 'Accuracy']
        st.success(f"🏆 Best Model: **{best_model}** (Accuracy: {best_score:.4f})")
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(results))
        ax.bar(x, results['Accuracy'], color=PALETTE[0], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(results['Model'], rotation=30, ha='right')
        style_plot(ax, "Model Comparison")
        st.pyplot(fig)
        plt.close()
    
    elif eval_analyzer.problem_type == 'regression':
        best_idx = results['R2-Score'].idxmax()
        best_model = results.loc[best_idx, 'Model']
        best_score = results.loc[best_idx, 'R2-Score']
        st.success(f"🏆 Best Model: **{best_model}** (R² Score: {best_score:.4f})")

def render_export(analyzer):
    st.markdown("### 📥 Export Results")
    
    st.markdown("#### Download Cleaned Data")
    csv = analyzer.df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "cleaned_data.csv", "text/csv")
    
    st.markdown("#### Download Report")
    report = f"""
# Data Analysis Report

## Dataset Summary
- Rows: {analyzer.df.shape[0]:,}
- Columns: {analyzer.df.shape[1]}
- Numeric Features: {len(analyzer.numeric_cols)}
- Categorical Features: {len(analyzer.categorical_cols)}

## Preprocessing Steps
{chr(10).join(f'- {log}' for log in analyzer.preprocessing_log)}

## Insights
{chr(10).join(f'- {insight}' for insight in analyzer.generate_insights())}
"""
    st.download_button("📥 Download Report", report, "analysis_report.md", "text/markdown")

if __name__ == "__main__":
    main()
