"""
Auto Data Analyzer & ML Recommender - Streamlit UI
Modern, Professional Data Science Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.axes
import seaborn as sns
from data_analyzer import DataAnalyzer, preprocess_data

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Data Analyzer & ML Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        min-height: 100vh;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        color: #f1f5f9 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e7ff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #c7d2fe !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 6px;
        gap: 6px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8 !important;
        font-weight: 600;
        padding: 10px 20px;
        background: transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.6);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(6, 182, 212, 0.5);
        border-radius: 16px;
        padding: 20px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        color: #f1f5f9 !important;
        font-weight: 600;
        padding: 12px 16px;
    }
    
    /* Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.2);
        margin: 24px 0;
    }
    
    /* Markdown Text */
    .stMarkdown p {
        color: #cbd5e1;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Plot Configuration ────────────────────────────────────────────────────
PLOT_BG = "transparent"
PLOT_FG = "#f1f5f9"
GRID_CLR = "rgba(148, 163, 184, 0.2)"
PALETTE = [
    "#06b6d4",  # Cyan
    "#8b5cf6",  # Purple
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#3b82f6",  # Blue
    "#ec4899",  # Pink
    "#14b8a6",  # Teal
]


def _create_gradient_cmap(colors: list, n_bins: int = 256):
    """Create a gradient colormap from a list of colors."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("gradient", colors, N=n_bins)


def _style_ax(ax: matplotlib.axes.Axes, title: str = ""):
    ax.set_facecolor(PLOT_BG)
    fig = ax.get_figure()
    if fig is not None:
        fig.patch.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_FG)
    ax.xaxis.label.set_color(PLOT_FG)
    ax.yaxis.label.set_color(PLOT_FG)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', color=PLOT_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, alpha=0.4, color=GRID_CLR, linestyle='--')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(PLOT_FG)


# ── Main Application ───────────────────────────────────────────────────────
def main():
    # Title
    st.title("📊 Auto Data Analyzer & ML Recommender")
    st.markdown("##### Intelligent ML Pipeline Automation System")
    st.markdown("---")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # File Upload Section
    st.sidebar.markdown("### 📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your dataset in CSV or Excel format"
    )
    st.sidebar.markdown("---")
    
    # Preprocessing Options
    st.sidebar.markdown("### 🔧 Preprocessing Options")
    
    missing_strategy = st.sidebar.selectbox(
        "Missing Value Strategy",
        ['auto', 'mean', 'median', 'mode'],
        index=0,
        help="Strategy to handle missing values"
    )
    
    remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR)", value=True)
    
    encode_method = st.sidebar.selectbox(
        "Categorical Encoding",
        ['label', 'onehot'],
        index=0,
        help="Method to encode categorical features"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Upload a dataset to begin automated analysis and ML recommendations.")

    # ── Welcome Screen (No File) ─────────────────────────────────────────────
    if uploaded_file is None:
        _render_welcome_screen()
        return

    # ── File loaded ──────────────────────────────────────────────────────────
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df_original = df.copy()

        # Run preprocessing once and cache in session state
        proc_key = f"analyzer_{uploaded_file.name}_{missing_strategy}_{remove_outliers}_{encode_method}"
        if proc_key not in st.session_state:
            st.session_state[proc_key] = preprocess_data(
                df.copy(),
                missing_strategy=missing_strategy,
                remove_outliers=remove_outliers,
                encode_method=encode_method
            )
        analyzer: DataAnalyzer = st.session_state[proc_key]

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dataset Overview",
            "🧹 Preprocessing",
            "📈 EDA & Visualizations",
            "🤖 ML Recommendations",
            "⚡ Model Evaluation",
        ])

        # ── TAB 1: Dataset Overview ──────────────────────────────────────────
        with tab1:
            st.markdown("### 📊 Dataset Overview")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            st.markdown("#### 🔍 First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("#### 📋 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null Values': df.isnull().sum().values,
                'Unique Values': [df[c].nunique() for c in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"🔢 **Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}")
            with c2:
                st.info(f"📝 **Categorical Columns ({len(categorical_cols)}):** {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}")

        # ── TAB 2: Preprocessing ─────────────────────────────────────────────
        with tab2:
            st.markdown("### 🧹 Data Preprocessing")

            missing_before = int(df_original.isnull().sum().sum())
            missing_after = int(analyzer.df.isnull().sum().sum())
            dupes_removed = int(df_original.duplicated().sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Missing Values (Before)", missing_before)
            c2.metric("Missing Values (After)", missing_after, delta=missing_after - missing_before)
            c3.metric("Duplicates Removed", dupes_removed)

            c1, c2 = st.columns(2)
            c1.metric("Rows After Cleaning", f"{analyzer.df.shape[0]:,}")
            c2.metric("Columns After Cleaning", analyzer.df.shape[1])

            st.markdown("#### 👀 Cleaned Data Preview")
            st.dataframe(analyzer.df.head(10), use_container_width=True)

            csv = analyzer.df.to_csv(index=False)
            st.download_button("📥 Download Cleaned Dataset", data=csv,
                               file_name="cleaned_data.csv", mime="text/csv")

        # ── TAB 3: EDA & Visualizations ──────────────────────────────────────
        with tab3:
            st.markdown("### 📈 Exploratory Data Analysis")

            eda_analyzer = DataAnalyzer(analyzer.df)
            eda_analyzer.analyze_types()

            viz_type = st.selectbox(
                "Select Visualization",
                ["Histograms", "Box Plots", "Correlation Heatmap", "Count Plots"]
            )

            if viz_type == "Histograms":
                st.markdown("#### Distribution of Numeric Features")
                cols_grid = st.columns(3)
                for i, col in enumerate(eda_analyzer.numeric_cols[:9]):
                    with cols_grid[i % 3]:
                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        eda_analyzer.df[col].hist(bins=25, ax=ax, color=PALETTE[i % len(PALETTE)],
                                                   edgecolor='white', alpha=0.85)
                        _style_ax(ax, col)
                        ax.set_xlabel("Value", fontsize=9)
                        ax.set_ylabel("Frequency", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            elif viz_type == "Box Plots":
                st.markdown("#### Box Plots for Outlier Detection")
                cols_grid = st.columns(2)
                for i, col in enumerate(eda_analyzer.numeric_cols[:6]):
                    with cols_grid[i % 2]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        bp = ax.boxplot(eda_analyzer.df[col].dropna(), patch_artist=True,
                                        medianprops=dict(color='#ef4444', linewidth=2))
                        for patch in bp['boxes']:
                            patch.set_facecolor(PALETTE[i % len(PALETTE)])
                            patch.set_alpha(0.7)
                        _style_ax(ax, col)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            elif viz_type == "Correlation Heatmap":
                st.markdown("#### Correlation Matrix")
                if len(eda_analyzer.numeric_cols) > 1:
                    corr_matrix = eda_analyzer.df[eda_analyzer.numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(max(8, len(eda_analyzer.numeric_cols)), max(6, len(eda_analyzer.numeric_cols) - 1)))
                    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax,
                                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                                annot_kws={'fontsize': 9, 'color': '#1e293b'}, fmt='.2f')
                    _style_ax(ax, "Feature Correlation Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis.")

            elif viz_type == "Count Plots":
                st.markdown("#### Categorical Feature Distributions")
                if eda_analyzer.categorical_cols:
                    cat_col = st.selectbox("Select Categorical Column", eda_analyzer.categorical_cols)
                    if cat_col:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        counts = eda_analyzer.df[cat_col].value_counts().head(10)
                        cmap = mplcm.get_cmap('viridis')
                        colors = [cmap(v) for v in np.linspace(0.2, 0.8, len(counts))]
                        bar_vals = np.array(counts.values, dtype=float)
                        ax.bar(range(len(counts)), bar_vals,
                               color=colors, edgecolor='white', linewidth=1.2)
                        ax.set_xticks(range(len(counts)))
                        ax.set_xticklabels(counts.index.tolist(), rotation=40, ha='right', fontsize=10)
                        ax.set_xlabel(cat_col, fontsize=11)
                        ax.set_ylabel("Count", fontsize=11)
                        _style_ax(ax, f"Distribution of {cat_col}")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.info("No categorical columns found in the cleaned dataset.")

            st.markdown("#### ⚡ Data Quality Issues")
            issues = eda_analyzer.detect_data_issues()
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ No significant data quality issues detected!")

        # ── TAB 4: ML Recommendations ────────────────────────────────────────
        with tab4:
            st.markdown("### 🤖 ML Recommendation Engine")

            all_cols = df.columns.tolist()
            target_col = st.selectbox(
                "🎯 Select Target Column",
                options=['Auto-Detect'] + all_cols,
                index=0,
                help="Select the column you want to predict"
            )
            target = None if target_col == 'Auto-Detect' else target_col

            # Save target selection for Tab 5
            st.session_state['selected_target'] = target

            rec_analyzer = DataAnalyzer(analyzer.df)
            rec_analyzer.analyze_types()
            problem_type = rec_analyzer.detect_problem_type(target_col=target)

            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.8);border-left:4px solid #8b5cf6;
                        padding:16px 20px;border-radius:12px;margin:16px 0;'>
                <strong style='color:#8b5cf6;font-size:18px;'>🎯 Detected Problem Type:</strong>
                <span style='color:#f1f5f9;font-size:16px;'> {problem_type.title()}</span>
            </div>""", unsafe_allow_html=True)

            recommendations = rec_analyzer.recommend_algorithms()
            st.markdown("#### 💡 Recommended Algorithms")

            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['algorithm']}", expanded=(i == 1)):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**📋 Description:** {rec['description']}")
                        st.markdown(f"**✅ Pros:** {rec['pros']}")
                    with c2:
                        st.markdown(f"**❌ Cons:** {rec['cons']}")
                        st.markdown(f"**🕐 Use When:** {rec['use_when']}")

        # ── TAB 5: Model Evaluation ──────────────────────────────────────────
        with tab5:
            st.markdown("### ⚡ Model Evaluation & Comparison")

            saved_target = st.session_state.get('selected_target', None)

            eval_analyzer = DataAnalyzer(analyzer.df)
            eval_analyzer.analyze_types()
            eval_analyzer.detect_problem_type(target_col=saved_target)

            if eval_analyzer.target_col is None:
                st.info("👆 Go to **ML Recommendations** tab and select a target column first.")
            else:
                st.markdown(f"""
                <div style='background:#f0fdf4;border-left:4px solid #10b981;
                            padding:10px 16px;border-radius:8px;margin-bottom:12px;'>
                    <strong>Target:</strong> {eval_analyzer.target_col} &nbsp;|&nbsp;
                    <strong>Problem Type:</strong> {(eval_analyzer.problem_type or '').title()}
                </div>""", unsafe_allow_html=True)

                with st.spinner("Training models..."):
                    results = eval_analyzer.train_and_evaluate()

                if results.empty:
                    st.warning("Not enough data to train models (need at least 10 rows with numeric features).")
                else:
                    st.markdown("#### 📊 Performance Metrics")

                    if eval_analyzer.problem_type == 'classification':
                        display = results.copy()
                        for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                            display[m] = display[m].apply(lambda x: f"{x:.4f}")
                        st.dataframe(display, use_container_width=True)

                        best_idx = int(results['Accuracy'].idxmax())
                        best_model = str(results.loc[best_idx, 'Model'])
                        best_score = float(str(results.loc[best_idx, 'Accuracy']))
                        st.success(f"🏆 **Best Model:** {best_model} — Accuracy: {best_score:.4f} ({best_score*100:.1f}%)")

                        st.markdown("#### 📈 Model Comparison Chart")
                        fig, ax = plt.subplots(figsize=(11, 5))
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        x = np.arange(len(results))
                        width = 0.18
                        for i, metric in enumerate(metrics):
                            metric_vals = np.array(results[metric].astype(float))
                            ax.bar(x + i * width, metric_vals, width,
                                   label=metric, color=PALETTE[i], alpha=0.88)
                        ax.set_xticks(x + width * 1.5)
                        ax.set_xticklabels(results['Model'].tolist(), rotation=30, ha='right')
                        ax.set_ylabel("Score")
                        ax.set_ylim(0, 1.05)
                        ax.legend(loc='lower right', framealpha=0.9)
                        _style_ax(ax, "Classification Model Performance")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    elif eval_analyzer.problem_type == 'regression':
                        display = results.copy()
                        display['MAE'] = display['MAE'].apply(lambda x: f"{x:.4f}")
                        display['RMSE'] = display['RMSE'].apply(lambda x: f"{x:.4f}")
                        display['R2-Score'] = display['R2-Score'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(display, use_container_width=True)

                        best_idx = int(results['R2-Score'].idxmax())
                        best_model = str(results.loc[best_idx, 'Model'])
                        best_score = float(str(results.loc[best_idx, 'R2-Score']))
                        st.success(f"🏆 **Best Model:** {best_model} — R² Score: {best_score:.4f}")

                        st.markdown("#### 📈 Model Comparison Chart")
                        fig, ax = plt.subplots(figsize=(11, 5))
                        x = np.arange(len(results))
                        width = 0.35
                        rmse_vals = np.array(results['RMSE'].astype(float))
                        rmse_norm = rmse_vals / rmse_vals.max() if rmse_vals.max() > 0 else rmse_vals
                        r2_vals = np.array(results['R2-Score'].astype(float))
                        ax.bar(x - width / 2, r2_vals, width,
                               label='R² Score', color=PALETTE[0], alpha=0.88)
                        ax.bar(x + width / 2, rmse_norm, width,
                               label='RMSE (normalized)', color=PALETTE[4], alpha=0.88)
                        ax.set_xticks(x)
                        ax.set_xticklabels(results['Model'].tolist(), rotation=30, ha='right')
                        ax.set_ylabel("Score")
                        ax.legend(loc='upper right', framealpha=0.9)
                        _style_ax(ax, "Regression Model Performance")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    elif eval_analyzer.problem_type == 'clustering':
                        st.dataframe(results, use_container_width=True)

                        best_idx = int(results['Silhouette Score'].idxmax())
                        best_k = int(str(results.loc[best_idx, 'K']))
                        best_score = float(str(results.loc[best_idx, 'Silhouette Score']))
                        st.success(f"🏆 **Optimal Clusters:** K = {best_k} — Silhouette Score: {best_score:.4f}")

                        st.markdown("#### 📈 Silhouette Score vs K")
                        fig, ax = plt.subplots(figsize=(9, 4))
                        k_vals = np.array(results['K'].astype(int))
                        sil_vals = np.array(results['Silhouette Score'].astype(float))
                        ax.plot(k_vals, sil_vals,
                                marker='o', linewidth=2.5, markersize=10, color=PALETTE[0])
                        ax.fill_between(k_vals, sil_vals,
                                        alpha=0.15, color=PALETTE[0])
                        ax.set_xlabel("Number of Clusters (K)")
                        ax.set_ylabel("Silhouette Score")
                        _style_ax(ax, "Silhouette Score vs Number of Clusters")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        # ── Summary ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Analysis Summary")
        summary_analyzer = DataAnalyzer(analyzer.df)
        summary_analyzer.analyze_types()
        summary_analyzer.detect_problem_type(target_col=st.session_state.get('selected_target'))
        for insight in summary_analyzer.generate_insights():
            if "⚠️" in insight:
                st.warning(insight)
            elif "✅" in insight:
                st.success(insight)
            else:
                st.info(insight)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.markdown("""
        **Troubleshooting:**
        - Ensure the file is a valid CSV or Excel format
        - Check that the file has proper column headers
        - Make sure the file is not corrupted
        """)


# ── Helper Functions ─────────────────────────────────────────────────────

def _render_welcome_screen():
    """Render the welcome screen when no file is uploaded."""
    st.markdown("""
    <div style='text-align:center;padding:60px 0 40px;'>
        <span style='font-size:80px;'>📊</span>
        <h2 style='color:#f1f5f9;margin:20px 0 8px;'>Welcome to Auto Data Analyzer</h2>
        <p style='color:#94a3b8;font-size:18px;'>Upload a dataset from the sidebar to get started</p>
    </div>""", unsafe_allow_html=True)

    # Feature Cards
    card_data = [
        ("#06b6d4", "📥", "Load Data", "CSV & Excel files supported"),
        ("#8b5cf6", "🧹", "Preprocess", "Clean & Transform data"),
        ("#10b981", "📊", "Analyze", "EDA & Insights"),
        ("#f59e0b", "📈", "Visualize", "Interactive Charts"),
        ("#ec4899", "🤖", "ML Recommend", "Smart Suggestions"),
        ("#ef4444", "⚡", "Evaluate", "Train & Compare Models"),
    ]
    
    row1 = st.columns(3)
    row2 = st.columns(3)
    
    for idx, (color, icon, title, desc) in enumerate(card_data):
        col = row1[idx] if idx < 3 else row2[idx - 3]
        with col:
            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.8);padding:28px;border-radius:16px;
                        border-top:4px solid {color};text-align:center;
                        box-shadow:0 8px 32px rgba(0,0,0,0.3);margin:10px 0;
                        transition:transform 0.3s ease;'>
                <div style='font-size:40px;'>{icon}</div>
                <h4 style='color:{color};margin:12px 0 6px;'>{title}</h4>
                <p style='color:#94a3b8;margin:0;font-size:14px;'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Upload** your dataset (CSV or Excel) using the sidebar  
    2. **Configure** preprocessing options in the sidebar  
    3. **Explore** the tabs: Overview → Preprocessing → EDA → ML Recommendations → Model Evaluation  
    """)
    
    # Example data format
    example_data = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50, 28, 33, 38, 42],
        'Salary': [50000, 60000, 75000, 80000, 90000, 95000, 55000, 65000, 70000, 85000],
        'Department': ['IT', 'HR', 'Sales', 'IT', 'Marketing', 'Finance', 'IT', 'HR', 'Sales', 'Marketing'],
        'Performance': ['Good', 'Excellent', 'Good', 'Average', 'Excellent', 'Good', 'Average', 'Good', 'Excellent', 'Good'],
        'Target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })
    st.markdown("**Example dataset format:**")
    st.dataframe(example_data, use_container_width=True)


def _render_dataset_overview(df: pd.DataFrame, df_original: pd.DataFrame, uploaded_file):
    """Render the Dataset Overview tab."""
    st.markdown("### 📊 Dataset Overview")
    
    # Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Column Information
    st.markdown("#### 🔍 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes.values],
        'Non-Null': df.count().values,
        'Null Values': df.isnull().sum().values,
        'Unique Values': [df[c].nunique() for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

    # Column Types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔢 Numeric Columns")
        if numeric_cols:
            for col in numeric_cols:
                st.markdown(f"- **{col}**")
        else:
            st.info("No numeric columns found")
    
    with c2:
        st.markdown("#### 📝 Categorical Columns")
        if categorical_cols:
            for col in categorical_cols:
                st.markdown(f"- **{col}**")
        else:
            st.info("No categorical columns found")

    # Data Preview
    st.markdown("#### 👀 Data Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)


def _render_preprocessing(df_original: pd.DataFrame, analyzer: DataAnalyzer):
    """Render the Preprocessing tab."""
    st.markdown("### 🧹 Data Preprocessing")
    
    # Preprocessing Metrics
    missing_before = int(df_original.isnull().sum().sum())
    missing_after = int(analyzer.df.isnull().sum().sum())
    dupes_removed = int(df_original.duplicated().sum())
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Values (Before)", f"{missing_before:,}")
    c2.metric("Missing Values (After)", f"{missing_after:,}", 
              delta=f"-{missing_before - missing_after}" if missing_before > missing_after else "0")
    c3.metric("Duplicates Removed", f"{dupes_removed:,}")
    c4.metric("Rows After Cleaning", f"{analyzer.df.shape[0]:,}")

    # Cleaned Data Preview
    st.markdown("#### 👀 Cleaned Data Preview")
    st.dataframe(analyzer.df.head(10), use_container_width=True)

    # Download Button
    csv = analyzer.df.to_csv(index=False)
    st.download_button(
        "📥 Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )


def _render_eda(analyzer: DataAnalyzer):
    """Render the EDA tab with visualizations."""
    st.markdown("### 📈 Exploratory Data Analysis")
    
    eda_analyzer = DataAnalyzer(analyzer.df)
    eda_analyzer.analyze_types()
    
    # Visualization Selector
    viz_type = st.selectbox(
        "Select Visualization",
        ["Histograms", "Box Plots", "Correlation Heatmap", "Count Plots", "Pair Plot", "Violin Plot"]
    )
    
    if viz_type == "Histograms":
        _render_histograms(eda_analyzer)
    elif viz_type == "Box Plots":
        _render_boxplots(eda_analyzer)
    elif viz_type == "Correlation Heatmap":
        _render_correlation_heatmap(eda_analyzer)
    elif viz_type == "Count Plots":
        _render_countplots(eda_analyzer)
    elif viz_type == "Pair Plot":
        _render_pairplot(eda_analyzer)
    elif viz_type == "Violin Plot":
        _render_violinplot(eda_analyzer)

    # Data Quality Issues
    st.markdown("#### ⚡ Data Quality Issues")
    issues = eda_analyzer.detect_data_issues()
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ No significant data quality issues detected!")


def _render_histograms(eda_analyzer: DataAnalyzer):
    """Render histogram distributions."""
    st.markdown("#### Distribution of Numeric Features")
    
    if not eda_analyzer.numeric_cols:
        st.info("No numeric columns available for histogram visualization.")
        return
    
    cols_grid = st.columns(3)
    max_cols = min(9, len(eda_analyzer.numeric_cols))
    
    for i in range(max_cols):
        col = eda_analyzer.numeric_cols[i]
        with cols_grid[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 4))
            n, bins, patches = ax.hist(
                eda_analyzer.df[col].dropna(),
                bins=25,
                color=PALETTE[i % len(PALETTE)],
                edgecolor='white',
                alpha=0.85
            )
            _style_ax(ax, col)
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def _render_boxplots(eda_analyzer: DataAnalyzer):
    """Render box plots for outlier detection."""
    st.markdown("#### Box Plots for Outlier Detection")
    
    if not eda_analyzer.numeric_cols:
        st.info("No numeric columns available for box plot visualization.")
        return
    
    cols_grid = st.columns(2)
    max_cols = min(6, len(eda_analyzer.numeric_cols))
    
    for i in range(max_cols):
        col = eda_analyzer.numeric_cols[i]
        with cols_grid[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 5))
            bp = ax.boxplot(
                eda_analyzer.df[col].dropna(),
                patch_artist=True,
                medianprops=dict(color='#ef4444', linewidth=2.5),
                flierprops=dict(marker='o', markerfacecolor='#f59e0b', markersize=6, alpha=0.6)
            )
            for patch in bp['boxes']:
                gradient_cmap = _create_gradient_cmap([PALETTE[i % len(PALETTE)], PALETTE[(i+1) % len(PALETTE)]])
                patch.set_facecolor(gradient_cmap(0.5))
                patch.set_alpha(0.8)
            _style_ax(ax, col)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def _render_correlation_heatmap(eda_analyzer: DataAnalyzer):
    """Render correlation heatmap."""
    st.markdown("#### Correlation Matrix")
    
    if len(eda_analyzer.numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    corr_matrix = eda_analyzer.df[eda_analyzer.numeric_cols].corr()
    fig_size = max(10, len(eda_analyzer.numeric_cols))
    
    fig, ax = plt.subplots(figsize=(fig_size * 0.8, fig_size * 0.6))
    
    # Create custom colormap
    cmap = _create_gradient_cmap(["#ef4444", "#f59e0b", "#10b981", "#06b6d4", "#8b5cf6"])
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        annot_kws={'fontsize': 9, 'color': '#f1f5f9'},
        fmt='.2f',
        vmin=-1,
        vmax=1
    )
    _style_ax(ax, "Feature Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_countplots(eda_analyzer: DataAnalyzer):
    """Render count plots for categorical features."""
    st.markdown("#### Categorical Feature Distributions")
    
    if not eda_analyzer.categorical_cols:
        st.info("No categorical columns found in the cleaned dataset.")
        return
    
    cat_col = st.selectbox("Select Categorical Column", eda_analyzer.categorical_cols)
    
    if cat_col:
        fig, ax = plt.subplots(figsize=(12, 6))
        counts = eda_analyzer.df[cat_col].value_counts().head(15)
        
        # Create gradient bars
        gradient_cmap = _create_gradient_cmap(PALETTE[:len(counts)])
        colors = [gradient_cmap(i / len(counts)) for i in range(len(counts))]
        
        bars = ax.bar(range(len(counts)), np.array(counts.values, dtype=float), color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.tolist(), rotation=45, ha='right', fontsize=11)
        ax.set_xlabel(cat_col, fontsize=12, color=PLOT_FG)
        ax.set_ylabel("Count", fontsize=12, color=PLOT_FG)
        
        # Add value labels on bars
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{int(val)}', ha='center', va='bottom', fontsize=10, color=PLOT_FG)
        
        _style_ax(ax, f"Distribution of {cat_col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_pairplot(eda_analyzer: DataAnalyzer):
    """Render pair plot for selected numeric columns."""
    st.markdown("#### Pair Plot (Sample)")
    
    if len(eda_analyzer.numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for pair plot.")
        return
    
    # Select up to 4 columns for pair plot
    selected_cols = st.multiselect(
        "Select columns for pair plot",
        eda_analyzer.numeric_cols,
        default=eda_analyzer.numeric_cols[:min(4, len(eda_analyzer.numeric_cols))]
    )
    
    if len(selected_cols) >= 2:
        sample_df = eda_analyzer.df[selected_cols].sample(min(100, len(eda_analyzer.df)))
        fig = sns.pairplot(sample_df, diag_kind="kde", palette=PALETTE[:len(selected_cols)])
        fig.fig.patch.set_facecolor(PLOT_BG)
        plt.tight_layout()
        st.pyplot(fig.fig)
        plt.close('all')
    else:
        st.info("Please select at least 2 columns.")


def _render_violinplot(eda_analyzer: DataAnalyzer):
    """Render violin plots for numeric features."""
    st.markdown("#### Violin Plots")
    
    if not eda_analyzer.numeric_cols:
        st.info("No numeric columns available.")
        return
    
    selected_col = st.selectbox("Select Column", eda_analyzer.numeric_cols)
    
    if selected_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot - convert to numpy array
        data = np.array(eda_analyzer.df[selected_col].dropna().values, dtype=float)
        
        # Use violinplot with showlines to avoid collection issues
        ax.violinplot(data, positions=[1], showmeans=True, showmedians=True, showextrema=True)
        
        _style_ax(ax, f"Violin Plot: {selected_col}")
        ax.set_ylabel("Value", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_ml_recommendations(df: pd.DataFrame, analyzer: DataAnalyzer):
    """Render ML Recommendations tab."""
    st.markdown("### 🤖 ML Recommendation Engine")
    
    # Target Column Selection
    all_cols = df.columns.tolist()
    target_col = st.selectbox(
        "🎯 Select Target Column",
        options=['Auto-Detect'] + all_cols,
        index=0,
        help="Select the column you want to predict"
    )
    
    target = None if target_col == 'Auto-Detect' else target_col
    st.session_state['selected_target'] = target
    
    # Analyze and detect problem type
    rec_analyzer = DataAnalyzer(analyzer.df)
    rec_analyzer.analyze_types()
    problem_type = rec_analyzer.detect_problem_type(target_col=target)
    
    # Problem Type Display
    problem_type_display = {
        'classification': ('🎯 Classification', '#10b981'),
        'regression': ('📈 Regression', '#06b6d4'),
        'clustering': ('🔮 Clustering', '#8b5cf6')
    }
    
    pt_display, pt_color = problem_type_display.get(problem_type, ('❓ Unknown', '#94a3b8'))
    
    st.markdown(f"""
    <div style='background:rgba(30,41,59,0.8);border-left:4px solid {pt_color};
                padding:16px 20px;border-radius:12px;margin:16px 0;'>
        <strong style='color:{pt_color};font-size:18px;'>{pt_display}</strong>
        <span style='color:#94a3b8;margin-left:10px;'>- {problem_type.title()} Problem Detected</span>
    </div>""", unsafe_allow_html=True)
    
    # Get Recommendations
    recommendations = rec_analyzer.recommend_algorithms()
    
    st.markdown("#### 💡 Recommended Algorithms")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['algorithm']}", expanded=(i == 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**📋 Description:** {rec['description']}")
                st.markdown(f"**✅ Pros:** {rec['pros']}")
            with c2:
                st.markdown(f"**❌ Cons:** {rec['cons']}")
                st.markdown(f"**🕐 Use When:** {rec['use_when']}")


def _render_model_evaluation(analyzer: DataAnalyzer):
    """Render Model Evaluation tab."""
    st.markdown("### ⚡ Model Evaluation & Comparison")
    
    saved_target = st.session_state.get('selected_target', None)
    
    eval_analyzer = DataAnalyzer(analyzer.df)
    eval_analyzer.analyze_types()
    eval_analyzer.detect_problem_type(target_col=saved_target)
    
    if eval_analyzer.target_col is None:
        st.info("👆 Go to **ML Recommendations** tab and select a target column first.")
        return
    
    # Target Info
    st.markdown(f"""
    <div style='background:rgba(16,185,129,0.15);border-left:4px solid #10b981;
                padding:12px 16px;border-radius:8px;margin-bottom:16px;'>
        <strong>Target:</strong> {eval_analyzer.target_col} &nbsp;|&nbsp;
        <strong>Problem Type:</strong> {(eval_analyzer.problem_type or 'Unknown').title()}
    </div>""", unsafe_allow_html=True)
    
    with st.spinner("Training models... This may take a moment."):
        results = eval_analyzer.train_and_evaluate()
    
    if results.empty:
        st.warning("Not enough data to train models (need at least 10 rows with numeric features).")
        return
    
    # Display Results
    st.markdown("#### 📊 Performance Metrics")
    
    if eval_analyzer.problem_type == 'classification':
        _render_classification_results(results)
    elif eval_analyzer.problem_type == 'regression':
        _render_regression_results(results)
    elif eval_analyzer.problem_type == 'clustering':
        _render_clustering_results(results)


def _render_classification_results(results: pd.DataFrame):
    """Render classification model results."""
    display = results.copy()
    for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        display[m] = display[m].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display, use_container_width=True)
    
    # Best Model
    best_idx = int(results['Accuracy'].idxmax())
    best_model = str(results.iloc[best_idx]['Model'])
    best_score = float(results.iloc[best_idx]['Accuracy'])
    
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(6,182,212,0.2));
                border-radius:16px;padding:20px;text-align:center;margin:16px 0;'>
        <h3 style='color:#10b981;margin:0;'>🏆 Best Model: {best_model}</h3>
        <p style='color:#f1f5f9;font-size:24px;margin:8px 0 0;'>
            Accuracy: <strong>{best_score:.4f}</strong> ({best_score*100:.1f}%)
        </p>
    </div>""", unsafe_allow_html=True)
    
    # Comparison Chart
    st.markdown("#### 📈 Model Comparison Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        metric_vals = np.array(results[metric].astype(float).values, dtype=float)
        ax.bar(x + i * width, metric_vals, width, label=metric, color=PALETTE[i], alpha=0.9)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results['Model'].tolist(), rotation=30, ha='right')
    ax.set_ylabel("Score", color=PLOT_FG)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', framealpha=0.9, facecolor='#1e293b')
    _style_ax(ax, "Classification Model Performance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_regression_results(results: pd.DataFrame):
    """Render regression model results."""
    display = results.copy()
    display['MAE'] = display['MAE'].apply(lambda x: f"{x:.4f}")
    display['RMSE'] = display['RMSE'].apply(lambda x: f"{x:.4f}")
    display['R2-Score'] = display['R2-Score'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display, use_container_width=True)
    
    # Best Model
    best_idx = int(results['R2-Score'].idxmax())
    best_model = str(results.iloc[best_idx]['Model'])
    best_score = float(results.iloc[best_idx]['R2-Score'])
    
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(6,182,212,0.2),rgba(139,92,246,0.2));
                border-radius:16px;padding:20px;text-align:center;margin:16px 0;'>
        <h3 style='color:#06b6d4;margin:0;'>🏆 Best Model: {best_model}</h3>
        <p style='color:#f1f5f9;font-size:24px;margin:8px 0 0;'>
            R² Score: <strong>{best_score:.4f}</strong>
        </p>
    </div>""", unsafe_allow_html=True)
    
    # Comparison Chart
    st.markdown("#### 📈 Model Comparison Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.35
    
    r2_vals = np.array(results['R2-Score'].astype(float).values, dtype=float)
    rmse_vals = np.array(results['RMSE'].astype(float).values, dtype=float)
    rmse_max = rmse_vals.max() if hasattr(rmse_vals, 'max') else float(np.max(rmse_vals))
    rmse_norm = rmse_vals / rmse_max if rmse_max > 0 else rmse_vals
    
    ax.bar(x - width/2, r2_vals, width, label='R² Score', color=PALETTE[0], alpha=0.9)
    ax.bar(x + width/2, rmse_norm, width, label='RMSE (normalized)', color=PALETTE[3], alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'].tolist(), rotation=30, ha='right')
    ax.set_ylabel("Score", color=PLOT_FG)
    ax.legend(loc='upper right', framealpha=0.9, facecolor='#1e293b')
    _style_ax(ax, "Regression Model Performance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_clustering_results(results: pd.DataFrame):
    """Render clustering model results."""
    st.dataframe(results, use_container_width=True)
    
    # Best K
    best_idx = int(results['Silhouette Score'].idxmax())
    best_k = int(results.iloc[best_idx]['K'])
    best_score = float(results.iloc[best_idx]['Silhouette Score'])
    
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(139,92,246,0.2),rgba(236,72,153,0.2));
                border-radius:16px;padding:20px;text-align:center;margin:16px 0;'>
        <h3 style='color:#8b5cf6;margin:0;'>🔮 Optimal Clusters: K = {best_k}</h3>
        <p style='color:#f1f5f9;font-size:24px;margin:8px 0 0;'>
            Silhouette Score: <strong>{best_score:.4f}</strong>
        </p>
    </div>""", unsafe_allow_html=True)
    
    # Silhouette Score Chart
    st.markdown("#### 📈 Silhouette Score vs K")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    k_vals = np.array(results['K'].astype(int).values, dtype=int)
    sil_vals = np.array(results['Silhouette Score'].astype(float).values, dtype=float)
    
    ax.plot(k_vals, sil_vals, marker='o', linewidth=2.5, markersize=12, color=PALETTE[1])
    ax.fill_between(k_vals, sil_vals, alpha=0.2, color=PALETTE[1])
    
    ax.set_xlabel("Number of Clusters (K)", color=PLOT_FG)
    ax.set_ylabel("Silhouette Score", color=PLOT_FG)
    _style_ax(ax, "Silhouette Score vs Number of Clusters")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_summary(analyzer: DataAnalyzer):
    """Render the analysis summary."""
    st.markdown("---")
    st.markdown("### 📋 Analysis Summary")
    
    summary_analyzer = DataAnalyzer(analyzer.df)
    summary_analyzer.analyze_types()
    summary_analyzer.detect_problem_type(target_col=st.session_state.get('selected_target'))
    
    for insight in summary_analyzer.generate_insights():
        if "⚠️" in insight:
            st.warning(insight)
        elif "✅" in insight:
            st.success(insight)
        else:
            st.info(insight)


if __name__ == "__main__":
    main()
