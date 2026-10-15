import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load feature dataset
df = pd.read_csv("attendance_dataset_features.csv")

# Select features and target
features = ['day_of_week', 'is_weekend', 'cum_absences']
X = df[features]
y = df['status_bin']

# Train logistic regression model on entire dataset (no train/test split)
model = LogisticRegression()
model.fit(X, y)

print("Model trained on entire data.")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
