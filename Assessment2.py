

# -------- 1. IMPORT LIBRARIES --------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer


# -------- 2. LOAD DATA --------
df = pd.read_csv("cleaned_Mental_Health_Raw_Data.csv")

print("Initial Shape:", df.shape)
print(df.head())


# -------- 3. CLEAN COLUMN NAMES --------
df.columns = df.columns.astype(str)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


# -------- 4. REMOVE EMPTY ROWS/COLUMNS --------
df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')


# -------- 5. ENSURE AGE COLUMN --------
if "age" not in df.columns:
    df.rename(columns={df.columns[0]: "age"}, inplace=True)

df["age"] = df["age"].astype(str)


# -------- 6. CONVERT ALL TO NUMERIC --------
for col in df.columns:
    if col != "age":
        df[col] = pd.to_numeric(df[col], errors='coerce')


# -------- 7. KEEP ONLY NUMERIC --------
df_numeric = df.select_dtypes(include=[np.number])

print("\nNumeric columns:", df_numeric.columns)
print("Shape before cleaning:", df_numeric.shape)


# -------- 8. REMOVE EMPTY COLUMNS --------
df_numeric = df_numeric.dropna(axis=1, how='all')


# -------- 9. HANDLE MISSING VALUES --------
imputer = SimpleImputer(strategy='mean')

data_imputed = imputer.fit_transform(df_numeric)

# Create dataframe safely (NO SHAPE ERROR)
df_imputed = pd.DataFrame(data_imputed)

# Assign safe column names
df_imputed.columns = [f"feature_{i}" for i in range(df_imputed.shape[1])]

print("\nAfter imputation:")
print(df_imputed.isna().sum())


# -------- 10. SCALING --------
scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df_imputed),
    columns=df_imputed.columns
)


# -------- 11. EDA --------

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_scaled.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram
df_scaled.hist(figsize=(12,10))
plt.tight_layout()
plt.show()


# -------- 12. BAR CHART (MATCH YOUR IMAGE) --------
labels = ["16–24", "25–34", "35–44", "45–54", "55–85", "Males", "Females", "Persons"]
values = [6.0, 2.5, 1.3, 0.9, 0.4, 1.2, 2.2, 1.7]

error = [v * 0.2 for v in values]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, values, yerr=error, capsize=5)

plt.title("12-month self-harm behaviours, by age then sex, 2020–2022")
plt.ylabel("Rate (%)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval,1),
             ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# -------- 13. DEFINE TARGET --------
target = df_scaled.columns[-1]

X = df_scaled.drop(target, axis=1)
y = df_scaled[target]


# -------- 14. TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------- 15. LINEAR REGRESSION --------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


# -------- 16. RANDOM FOREST --------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# -------- 17. EVALUATION --------
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")


# -------- 18. T-TEST --------
t_stat, p_value = ttest_ind(y_pred_lr, y_pred_rf)

print("\nT-Test Results")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant difference between models")
else:
    print("No significant difference")


# -------- 19. FEATURE IMPORTANCE --------
importance = rf.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nTop Features:")
print(imp_df.head())

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=imp_df)
plt.title("Feature Importance")
plt.show()


# -------- 20. SAVE DATA --------
df_scaled.to_csv("final_cleaned_dataset.csv", index=False)
