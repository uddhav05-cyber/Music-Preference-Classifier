# 🎵 Music Preference Classifier

A machine learning project that predicts music genre preferences based on demographic factors using Decision Tree classification.

## 🎯 Project Highlights

- **Accuracy**: 85%+ prediction accuracy
- **Dataset**: 500 samples with age/gender correlations
- **Model**: Decision Tree Classifier with optimal max_depth tuning
- **Deployment**: Flask web application with REST API

## 📊 Key Findings

- Age groups 10-25: Strong preference for Pop/Hip-Hop
- Age groups 40+: Classical and Jazz dominate
- Gender correlations identified in music taste patterns

## 🛠️ Technologies

- Python 3.8+
- scikit-learn
- Flask
- Pandas, NumPy
- Matplotlib/Seaborn for visualizations

## 🚀 Installation
```bash
pip install -r requirements.txt
python music_classifier.py  # Train model
python app.py  # Run web app
```

## 📈 Future Enhancements

- Add more features (location, income level)
- Implement Random Forest for comparison
- Deploy on cloud platform
```

### Step 5: Create Jupyter Notebook Analysis

Create `analysis.ipynb` with:
- Data exploration and visualization
- Feature correlation heatmaps
- Model performance metrics
- Decision tree visualization
- Confusion matrix

## Phase 3: Deployment Options

### Option A: Deploy to Heroku

1. Create `Procfile`:
```
web: gunicorn app:app
