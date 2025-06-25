import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle

df = pd.read_csv("attrition.csv")

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

if 'Age' in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=[1, 2, 3, 4])
    df['Age_Group'] = df['Age_Group'].astype(int)

if 'DistanceFromHome' in df.columns:
    df['Distance_Category'] = pd.cut(df['DistanceFromHome'], bins=[0, 5, 15, 30], labels=[1, 2, 3])
    df['Distance_Category'] = df['Distance_Category'].astype(int)

if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
    df['Experience_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
    df['Income_Per_Age'] = df['MonthlyIncome'] / df['Age']

features = ["JobSatisfaction", "MonthlyIncome", "YearsAtCompany", "OverTime", "WorkLifeBalance"]

additional_features = []
for col in ['Age_Group', 'Distance_Category', 'Experience_Ratio', 'Income_Per_Age', 
            'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'StockOptionLevel']:
    if col in df.columns:
        additional_features.append(col)

features.extend(additional_features)
features = [f for f in features if f in df.columns]

target = "Attrition"

X = df[features].fillna(0)
y = df[target]

minority_class = df[df[target] == 1]
majority_class = df[df[target] == 0]

minority_upsampled = resample(minority_class, 
                             replace=True,
                             n_samples=len(majority_class)//2,
                             random_state=42)

df_balanced = pd.concat([majority_class, minority_upsampled])
X_balanced = df_balanced[features].fillna(0)
y_balanced = df_balanced[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=1,
    min_child_weight=5,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=2
)

param_grid = {
    'n_estimators': [80, 100, 120],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.07],
    'reg_alpha': [0.5, 1, 1.5],
    'reg_lambda': [0.5, 1, 1.5]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
report = classification_report(y_test, y_pred)

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(features, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print("Classification Report:")
print(report)
