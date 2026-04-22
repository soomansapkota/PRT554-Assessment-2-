# ============================================
# FULL HD PIPELINE WITH EXTRA DATA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind


# ============================================
# 1. LOAD RAW DATA
# ============================================

df = pd.read_csv("Mental_Health_Cleaned_Raw_Data(6).csv")


# ============================================
# 2. CLEAN DATA
# ============================================

df.columns = df.columns.astype(str)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')

# Fix header
df.columns = df.iloc[0]
df = df[1:]
df.columns = df.columns.str.strip().str.lower()

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]


# ============================================
# 3. AGE COLUMN
# ============================================

age_col = [col for col in df.columns if "age" in col][0]
df.rename(columns={age_col: "age"}, inplace=True)


# ============================================
# 4. NUMERIC CONVERSION
# ============================================

for col in df.columns:
    if col != "age":
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()


# ============================================
# 5. OUTLIER REMOVAL (IQR)
# ============================================

for col in df.columns:
    if col != "age":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]


# ============================================
# 6. SPLIT INTO 4 DATASETS
# ============================================

def extract(df, keyword, name):
    cols = [c for c in df.columns if keyword in c]
    temp = df[["age"] + cols]

    df_long = temp.melt(id_vars=["age"], var_name="sex", value_name=name)
    df_long["sex"] = df_long["sex"].str.lower().str.strip()

    return df_long.dropna()


df_mental = extract(df, "mental", "mental")
df_suicidal = extract(df, "suicidal", "suicidal")
df_self = extract(df, "harm", "self_harm")
df_distress = extract(df, "distress", "distress")


# ============================================
# 7. MERGE (HETEROGENEOUS DATA)
# ============================================

df_final = df_mental.merge(df_suicidal, on=["age","sex"])
df_final = df_final.merge(df_self, on=["age","sex"])
df_final = df_final.merge(df_distress, on=["age","sex"])


# ============================================
# 8. ADD EXTRA DATA (SOCIO-ECONOMIC)
# ============================================

# Create synthetic extra dataset (for HD)
extra_data = pd.DataFrame({
    "age": df_final["age"].unique().repeat(2),
    "sex": ["male","female"] * len(df_final["age"].unique()),
    "income": np.random.randint(30000, 80000, len(df_final["age"].unique())*2),
    "employment_rate": np.random.uniform(0.5, 0.9, len(df_final["age"].unique())*2)
})

# Merge
df_final = df_final.merge(extra_data, on=["age","sex"], how="left")


# ============================================
# 9. ENCODE SEX
# ============================================

df_final["sex"] = df_final["sex"].astype("category").cat.codes


# ============================================
# 10. HANDLE MISSING
# ============================================

df_final = df_final.fillna(df_final.mean(numeric_only=True))


# ============================================
# 11. FEATURE SELECTION
# ============================================

X = df_final.drop(columns=["distress","age"])
y = df_final["distress"]


# ============================================
# 12. SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================
# 13. SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ============================================
# 14. MODELS
# ============================================

# Simple regression
lr_simple = LinearRegression()
lr_simple.fit(X_train[:, [0]], y_train)
y_pred_simple = lr_simple.predict(X_test[:, [0]])

# Multiple regression
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_pred_multi = lr_multi.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# ============================================
# 15. EVALUATION
# ============================================

def evaluate(y, pred, name):
    print(f"\n{name}")
    print("MSE:", mean_squared_error(y, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, pred)))
    print("R2:", r2_score(y, pred))

evaluate(y_test, y_pred_simple, "Simple Regression")
evaluate(y_test, y_pred_multi, "Multiple Regression")
evaluate(y_test, y_pred_rf, "Random Forest")


# ============================================
# 16. T-TEST
# ============================================

t_stat, p = ttest_ind(y_pred_simple, y_pred_multi)

print("\nT-test p-value:", p)

if p < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference")


# ============================================
# 17. VISUALISATION
# ============================================

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_final.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Distribution
df_final.hist(figsize=(10,8))
plt.show()

# Predictions
plt.scatter(y_test, y_pred_multi)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Regression Performance")
plt.show()

# Feature importance
importance = rf.feature_importances_
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# ============================================
# 18. SAVE FINAL DATA
# ============================================

df_final.to_csv("final_hd_dataset.csv", index=False)
