# 🚀 Quick Start Guide

## Running the Application

1. **Open Terminal/Command Prompt** in the project folder

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **The browser will automatically open** at `http://localhost:8501`

## First Time Setup

If you haven't installed dependencies yet:

```bash
pip install -r requirements.txt
```

## Testing with Sample Data

1. Run the application
2. Click "Browse files" in the sidebar
3. Upload `sample_data.csv` (included in the project)
4. Explore all tabs to see the features

## Navigation

### Sidebar
- **Upload Dataset**: Click to upload CSV or Excel files
- **Settings**: Configure preprocessing options
  - Missing Values: How to handle missing data
  - Remove Outliers: Toggle outlier removal
  - Encoding: Choose encoding method

### Main Tabs

1. **📊 Overview**
   - View dataset statistics
   - Check data quality
   - See feature correlations

2. **⚙️ Preprocessing**
   - Review cleaning steps
   - Preview cleaned data

3. **📈 EDA & Visualizations**
   - Select chart type
   - Explore data visually

4. **🤖 ML Recommendations**
   - Select target column
   - View problem type
   - Read algorithm suggestions

5. **🏆 Model Evaluation**
   - Train models automatically
   - Compare performance
   - Identify best model

6. **📥 Export Results**
   - Download cleaned data
   - Download analysis report

## Tips

- **Auto-Detect**: Let the system automatically detect the target column
- **Quality Score**: Aim for 90+ for best results
- **Model Training**: May take 10-30 seconds depending on dataset size
- **Large Files**: Files >10MB may take longer to process

## Troubleshooting

**Application won't start?**
- Check Python version: `python --version` (need 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt`

**Can't upload file?**
- Ensure file is CSV or Excel format
- Check file isn't corrupted
- Try a smaller file first

**Models not training?**
- Need at least 10 rows of data
- Ensure target column is selected
- Check for sufficient numeric features

## Example Workflow

1. Upload `sample_data.csv`
2. Keep default settings
3. Go to **Overview** → See 25 rows, 6 columns
4. Go to **ML Recommendations** → Select "Promoted" as target
5. See "Classification" problem detected
6. Go to **Model Evaluation** → Train models
7. See Random Forest achieves ~80% accuracy
8. Go to **Export** → Download results

## Next Steps

- Try your own datasets
- Experiment with different settings
- Compare different encoding methods
- Explore various visualizations

---

**Need Help?** Check README.md for detailed documentation
