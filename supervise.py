import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('cleansing_water_data.csv')

factor_cols = [col for col in df.columns if col.startswith('factor_')]

imputer_rf = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=42
)

imputed_data = imputer_rf.fit_transform(df[factor_cols])

df[factor_cols] = pd.DataFrame(imputed_data, columns=factor_cols, index=df.index)

df.to_csv('cleansing_water_data_supervise_fill.csv')

for col in factor_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[factor_cols] = df[factor_cols].fillna(df[factor_cols].mean())
df['is_kiyora'] = (df['brand_primary'] == 'Kiyora').astype(int)

X = df[factor_cols]
y = df['is_kiyora']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

feature_importances = pd.DataFrame(
    {
        'Factor': factor_cols,
        'Importance_Score': rf_model.feature_importances_
    }
).sort_values(by='Importance_Score', ascending=False)

print("Why Kiyora (Top 5):")
print(feature_importances.head(5).to_string(index=False))
print("\n")

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

print(metrics)
names = list(metrics.keys())
values = list(metrics.values())

plt.bar(names, values)
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xlabel('Metric')

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.show()
