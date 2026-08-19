import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Local Dataset
filename = "data_classification.csv"
df = pd.read_csv(filename)

print("==================================================")
print(f"📊 Dataset Loaded: {filename}")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
print("==================================================\n")

# 2. Drop non-predictive metadata (ID & Date columns)
cols_to_drop = ["OrderID", "Date", "CustomerID"]
df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3. Convert Continuous Target ('TotalPrice') into Discrete Categories
# Splits prices into 3 classes: Low, Medium, High
target_col = df_clean.columns[-1]  # TotalPrice
df_clean[target_col] = pd.qcut(df_clean[target_col], q=3, labels=["Low", "Medium", "High"])

# 4. Separate Features (X) and Target (y)
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# 5. Encode Categorical Features into Numbers
X = pd.get_dummies(X, drop_first=True)

# 6. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Supervised Classification Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Training Complete!")
print(f"🎯 Model Accuracy Score: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))