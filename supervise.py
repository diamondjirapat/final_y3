import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cleansing_water_data.csv')

df_clean = df[df['use_cleansing_water'] == 'ใช้'].copy()
# df_clean.to_csv('clean_cleansing_water_data.csv', index=False)

df_clean['uses_kiyora'] = df_clean['brands_used'].apply(
    lambda x: 1 if isinstance(x, str) and 'Kiyora' in x else 0
)

factor_cols = [c for c in df.columns if c.startswith('factor_')]
# for col in factor_cols:
#     df_clean[col] = df_clean[col].fillna(df_clean[col].median())

def get_dummies_multiselect(series, prefix):
    return series.str.get_dummies(sep=',').add_prefix(f"{prefix}_")

concerns_df = get_dummies_multiselect(df_clean['concerns'], 'concern')

# concerns_df.to_csv('concerns_multiselect.csv', index=False)
# df_clean.to_csv('clean_cleansing_water_data.csv', index=False)

skin_type_encoded = pd.get_dummies(df_clean['skin_type'], prefix='skin', drop_first=False)
le = LabelEncoder()
df_clean['age_encoded'] = le.fit_transform(df_clean['age'])
df_clean['income_encoded'] = le.fit_transform(df_clean['monthly_income'])


X = pd.concat([df_clean[factor_cols], concerns_df, skin_type_encoded, df_clean[['age_encoded', 'income_encoded']]], axis=1)
y = df_clean['uses_kiyora']
# X.to_csv('x.csv', index=False)

class corr:
    corr = X.corrwith(y).sort_values(ascending=False)
    print("--- ปัจจัยที่มีผลเชิงบวกต่อการเลือกใช้ Kiyora มากที่สุด ---")
    print(corr.head(5))

    plt.rcParams['font.family'] = 'Tahoma'
    plt.figure(figsize=(10, 8))
    top_features = corr.abs().sort_values(ascending=False).head(10).index
    sns.heatmap(X[top_features].join(y).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Top 10 Features vs Uses Kiyora)')
    plt.tight_layout()
    plt.show()

class train:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    print("\n--- Model Accuracy ---")
    print(results_df)

    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    import joblib
    joblib.dump(model, 'dt.pkl')

    #plot
    results_df.set_index('Model', inplace=True)
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

class plot:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(y='age', data=df_clean, order=df_clean['age'].value_counts().index)
    plt.title('อายุคนที่ใช้ Cleansing Water)')

    plt.subplot(1, 2, 2)
    sns.countplot(y='skin_type', data=df_clean, order=df_clean['skin_type'].value_counts().index)
    plt.title('สภาพผิวของคนใช้ Cleansing Water')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    df_factors = df_clean[factor_cols].rename(columns=lambda x: x.replace('factor_', ''))
    sns.boxplot(data=df_factors, orient='h')
    plt.title('คะแนนปัจจัยในการเลือกซื้อ')
    plt.tight_layout()
    plt.show()
