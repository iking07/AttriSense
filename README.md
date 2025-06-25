# AttriSense - Employee Attrition Prediction System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.38%25-brightgreen.svg)]()

A machine learning-powered web application that predicts employee attrition using XGBoost with advanced feature engineering and provides actionable insights for HR teams.

## 🎯 Overview

AttriSense leverages XGBoost machine learning to help organizations predict and prevent employee turnover. By analyzing various employee factors with sophisticated feature engineering, the system provides accurate predictions with 88.38% accuracy, enabling proactive retention strategies.

## 📊 Model Performance

### Algorithm: XGBoost Classifier
- **Test Accuracy:** 88.38%
- **Cross-Validation Accuracy:** 86.14%
- **Model Type:** Gradient Boosting Classification

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| 0 (Stay) | 0.95 | 0.87 | 0.91 | 247 |
| 1 (Leave) | 0.78 | 0.91 | 0.84 | 123 |
| **Accuracy** | | | **0.88** | **370** |
| **Macro Avg** | 0.86 | 0.89 | 0.87 | 370 |
| **Weighted Avg** | 0.89 | 0.88 | 0.89 | 370 |

### Optimal Hyperparameters
```python
{
    'learning_rate': 0.07,
    'max_depth': 5, 
    'n_estimators': 120,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}
```

### Key Performance Metrics
- **Excellent Generalization:** Small gap between CV (86.14%) and test accuracy (88.38%)
- **Balanced Performance:** High performance on both majority and minority classes
- **Strong Recall for Attrition:** 91% recall for identifying employees likely to leave
- **High Precision for Retention:** 95% precision for predicting employee retention

## 🔍 Feature Importance Analysis

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | OverTime | 20.22% | Whether employee works overtime |
| 2 | StockOptionLevel | 11.15% | Employee stock option level (0-3) |
| 3 | JobLevel | 8.71% | Hierarchical job level (1-5) |
| 4 | MonthlyIncome | 7.84% | Employee monthly salary |
| 5 | YearsAtCompany | 7.54% | Tenure at current company |
| 6 | EnvironmentSatisfaction | 6.83% | Work environment satisfaction (1-4) |
| 7 | JobInvolvement | 6.34% | Level of job involvement (1-4) |
| 8 | Experience_Ratio | 6.07% | Tenure ratio vs total experience |
| 9 | JobSatisfaction | 5.99% | Overall job satisfaction (1-4) |
| 10 | Age_Group | 5.34% | Employee age category |

## 🏗️ Project Structure

```
AttriSense/
│
├── static/
│   └── style.css                 # Frontend styling
│
├── templates/
│   ├── index.html               # Main input form
│   └── result.html              # Prediction results page
│
├── attrition.csv                # Training dataset
├── model.py                     # XGBoost model training and evaluation
├── app.py                       # Flask web application
├── model.pkl                    # Trained XGBoost model (serialized)
├── model_columns.pkl            # Feature columns (serialized)
├── scaler.pkl                   # StandardScaler (serialized)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Features

- **High Accuracy Predictions:** 88.38% accuracy with XGBoost algorithm
- **Advanced Feature Engineering:** Automatically creates predictive features from raw data
- **Balanced Performance:** Excellent detection of both retention and attrition cases
- **Web-Based Interface:** User-friendly form for inputting employee data
- **Real-Time Results:** Instant prediction results with probability scores
- **Feature Importance:** Insights into which factors drive attrition decisions
- **Responsive Design:** Works across desktop and mobile devices
- **Scalable Architecture:** Built with Flask for easy deployment and scaling

## How Does it Work?

![Screenshot 2025-06-25 014130](https://github.com/user-attachments/assets/b968ea2c-d93a-44cb-8196-518b79666ef4)

![Screenshot 2025-06-25 014143](https://github.com/user-attachments/assets/bb31fa60-3120-4d4e-a25e-34b7bb7570bd)



## 📈 Model Architecture

### Feature Engineering Pipeline
1. **Categorical Encoding:** OverTime, Attrition mapped to binary values
2. **Age Grouping:** Age binned into meaningful categories (≤30, 31-40, 41-50, >50)
3. **Distance Categorization:** Distance from home grouped into ranges
4. **Experience Ratio:** Calculated as YearsAtCompany / TotalWorkingYears
5. **Income-Age Ratio:** Monthly income normalized by age
6. **Feature Scaling:** StandardScaler applied to all numerical features

### XGBoost Configuration
- **Regularization:** L1 (α=0.5) and L2 (λ=0.5) penalties prevent overfitting
- **Class Balancing:** `scale_pos_weight=2` handles minority class imbalance
- **Tree Constraints:** `max_depth=5`, `min_child_weight=5` for generalization
- **Sampling:** `subsample=0.7`, `colsample_bytree=0.7` for robustness
- **Grid Search:** Automated hyperparameter optimization with 3-fold CV

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AttriSense.git
   cd AttriSense
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.7.0
flask>=2.0.0
pickle-mixin>=1.0.2
numpy>=1.21.0
```

## 🏃‍♂️ Running the Application

1. **Train the model (if needed):**
   ```bash
   python model.py
   ```

2. **Start the Flask application:**
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## 💻 Usage

1. **Access the Web Interface:** Navigate to the home page
2. **Fill Employee Details:** Enter relevant employee information in the form
3. **Get Prediction:** Click submit to receive attrition prediction with confidence scores
4. **View Results:** See probability scores and actionable recommendations

## 🔧 Model Training Details

### Data Preprocessing
- **Class Imbalance Handling:** Upsampled minority class from ~16% to ~33%
- **Feature Selection:** Automatic inclusion of available predictive features
- **Missing Value Treatment:** Forward fill and zero imputation strategies
- **Train-Test Split:** 80-20 stratified split maintaining class distribution

### Performance Optimization
- **Grid Search Parameters:** Systematic hyperparameter tuning
- **Cross-Validation:** 5-fold CV for robust performance estimation
- **Early Stopping:** Prevents overfitting during training
- **Feature Importance:** Built-in XGBoost feature ranking

## 📊 Business Impact

### Key Insights from Model
1. **Overtime is the #1 Predictor:** 20% importance - work-life balance critical
2. **Financial Incentives Matter:** Stock options significantly reduce attrition risk
3. **Career Progression:** Higher job levels correlate with retention
4. **Satisfaction Factors:** Environment and job involvement are key drivers
5. **Experience Utilization:** Employees want their experience valued appropriately

### Actionable Recommendations
- **Reduce Mandatory Overtime:** Implement flexible work arrangements
- **Enhance Stock Programs:** Improve equity compensation packages
- **Create Career Paths:** Clear promotion and development opportunities
- **Monitor Satisfaction:** Regular surveys on environment and job involvement
- **Competitive Compensation:** Ensure salaries match experience levels

## 📈 Future Enhancements

### Planned Features
- **Model Explainability:** SHAP integration for individual prediction explanations
- **Database Integration:** PostgreSQL/MySQL for storing predictions and feedback
- **Cloud Deployment:** AWS/GCP deployment with auto-scaling
- **Advanced Analytics:** Trend analysis and predictive insights dashboard
- **API Endpoints:** RESTful API for integration with HR systems

### Technical Improvements
- **Model Ensemble:** Combine XGBoost with other algorithms
- **Real-time Learning:** Online learning capabilities for model updates
- **A/B Testing:** Compare different feature sets and algorithms
- **Automated Pipelines:** MLOps integration with automated retraining

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**AttriSense** - Predicting the future of your workforce with XGBoost precision 🚀