import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

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

corr = X.corrwith(y).sort_values(ascending=False)
print("--- ปัจจัยที่มีผลเชิงบวกต่อการเลือกใช้ Kiyora มากที่สุด ---")
print(corr.head(5))

plt.rcParams['font.family'] = 'Tahoma'
plt.figure(figsize=(30, 30))
top_features = corr.sort_values().index
sns.heatmap(X[top_features].join(y).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Top 10 Features vs Uses Kiyora)')
plt.tight_layout()
plt.show()

class plot:
    top_brands = df_clean['brand_primary'].value_counts().head(5).index.tolist()
    df_top_brands = df_clean[df_clean['brand_primary'].isin(top_brands)].copy()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(y='age', hue='brand_primary', data=df_top_brands, order=df_top_brands['age'].value_counts().index)
    plt.title('อายุคนที่ใช้แต่ละแบรนด์เป็นหลัก')

    plt.subplot(1, 2, 2)
    sns.countplot(y='skin_type', hue='brand_primary', data=df_top_brands, order=df_top_brands['skin_type'].value_counts().index)
    plt.title('สภาพผิวของคนที่ใช้แต่ละแบรนด์เป็นหลัก')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    df_factors = df_top_brands[['brand_primary'] + factor_cols].rename(columns=lambda x: x.replace('factor_', ''))
    df_melted = df_factors.melt(id_vars='brand_primary', var_name='Factor', value_name='Score')

    sns.boxplot(x='Score', y='Factor', hue='brand_primary', data=df_melted, orient='h')
    plt.title('คะแนนปัจจัยในการเลือกซื้อแต่ละแบรนด์เป็นหลัก')
    plt.xlabel('คะแนน (1-5)')
    plt.tight_layout()
    plt.show()

class plotComparison:
    top_brands = df_clean['brand_primary'].value_counts().head(5).index.tolist()
    df_top_brands = df_clean[df_clean['brand_primary'].isin(top_brands)]

    brand_core_values = df_top_brands.groupby('brand_primary')[factor_cols].mean()
    brand_core_values.columns = [c.replace('factor_', '') for c in brand_core_values.columns]

    print("Core Values Top 5 Brand")
    print(brand_core_values.T.round(2))

    plt.figure(figsize=(10, 8))
    sns.heatmap(brand_core_values.T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Brand Core Values Comparison')
    plt.xlabel('Brands')
    plt.ylabel('Factors')
    plt.tight_layout()
    plt.show()


class train:
    corr_values = X.corrwith(y)
    selected_features = corr_values[corr_values.abs() >= 0.10].index.tolist()
    X = X[selected_features]
    y = df_clean['uses_kiyora']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    results = []
    best_overall_model = None
    best_overall_score = 0

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_train_pred = best_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred, zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Train Accuracy': train_acc,
            'Train Precision': train_prec,
            'Train Recall': train_rec,
            'Train F1-score': train_f1,
            'Test Accuracy': acc,
            'Test Precision': precision_score(y_test, y_pred, zero_division=0),
            'Test Recall': recall_score(y_test, y_pred, zero_division=0),
            'Test F1-score': f1_score(y_test, y_pred, zero_division=0)
        })

        if acc > best_overall_score:
            best_overall_score = acc
            best_overall_model = best_model

    results_df = pd.DataFrame(results)
    
    print("\n--- Best Parameters ---")
    for res in results:
        print(f"{res['Model']}: {res['Best Params']}")

    print("\n--- Model Accuracy ---")
    display_df = results_df.drop('Best Params', axis=1)
    print(display_df)

    import joblib
    joblib.dump(best_overall_model, 'best_model.pkl')
    print(f"\n best_model.pkl (Accuracy: {best_overall_score:.4f})")

    #plot
    display_df.set_index('Model', inplace=True)
    
    train_cols = [c for c in display_df.columns if 'Train' in c]
    test_cols = [c for c in display_df.columns if 'Test' in c]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    display_df[train_cols].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Training Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='lower right')
    
    display_df[test_cols].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Testing Performance Comparison')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

class train_advanced:
    print("\n" + "="*50)
    print("Running Advanced Feature Engineering Model Pipeline")
    print("="*50)
    
    # 1. Advanced Preprocessing: One-hot encode age and income
    age_dummies = pd.get_dummies(df_clean['age'], prefix='age', drop_first=False)
    income_dummies = pd.get_dummies(df_clean['monthly_income'], prefix='income', drop_first=False)
    
    # Re-build X with new dummy columns (bypassing the LabelEncoders used in original class)
    X_adv = pd.concat([df_clean[factor_cols], concerns_df, skin_type_encoded, age_dummies, income_dummies], axis=1)
    y_adv = df_clean['uses_kiyora']
    
    # 2. Algorithmic Feature Selection (Random Forest feature importances)
    print(f"Original Feature Count (After One-Hot): {X_adv.shape[1]}")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_selector.fit(X_adv, y_adv)
    
    # Select features that have an importance >= median importance
    selector = SelectFromModel(rf_selector, prefit=True, threshold='median')
    selected_features = X_adv.columns[selector.get_support()]
    print(f"Selected Feature Count (using RF Importance): {len(selected_features)}")
    
    X_adv_selected = X_adv[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_adv_selected, y_adv, test_size=0.3, random_state=42, stratify=y_adv)

    # 3. Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Models Configuration
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    results = []
    best_overall_model = None
    best_overall_score = 0

    for name, model in models.items():
        print(f"Training Advanced {name} with GridSearchCV...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Calculate training metrics
        y_train_pred = best_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred, zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Train Accuracy': train_acc,
            'Train Precision': train_prec,
            'Train Recall': train_rec,
            'Train F1-score': train_f1,
            'Test Accuracy': acc,
            'Test Precision': precision_score(y_test, y_pred, zero_division=0),
            'Test Recall': recall_score(y_test, y_pred, zero_division=0),
            'Test F1-score': f1_score(y_test, y_pred, zero_division=0)
        })

        if acc > best_overall_score:
            best_overall_score = acc
            best_overall_model = best_model

    results_df = pd.DataFrame(results)
    
    print("\n--- Advanced Model Accuracy ---")
    display_df = results_df.drop('Best Params', axis=1)
    print(display_df)

    import joblib
    joblib.dump(best_overall_model, 'best_model_advanced.pkl')
    print(f"\nSaved best advanced overall model to 'best_model_advanced.pkl' (Accuracy: {best_overall_score:.4f})")

    #plot
    display_df.set_index('Model', inplace=True)
    
    train_cols = [c for c in display_df.columns if 'Train' in c]
    test_cols = [c for c in display_df.columns if 'Test' in c]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    display_df[train_cols].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Advanced - Training Performance')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='lower right')
    
    display_df[test_cols].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Advanced - Testing Performance')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
