import pandas as pd

# Assuming you have already loaded attendance_df as shown in Step 1
# For example, you can load from CSV produced in Step 1 or re-run extraction

attendance_df = pd.read_csv("attendance_dataset.csv")

# Convert 'date' column to datetime if not already
attendance_df['date'] = pd.to_datetime(attendance_df['date'])

# Add day of week (Monday=0, Sunday=6)
attendance_df['day_of_week'] = attendance_df['date'].dt.dayofweek

# Binary feature: is_weekend (1 if Saturday or Sunday)
attendance_df['is_weekend'] = attendance_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Create cumulative absence count per employee up to previous records (lag feature)
attendance_df = attendance_df.sort_values(['employee_id', 'date'])
attendance_df['cum_absences'] = attendance_df.groupby('employee_id')['status_bin'].cumsum() - attendance_df['status_bin']

print(attendance_df.head())

# Save the feature-enriched dataset for machine learning use
attendance_df.to_csv("attendance_dataset_features.csv", index=False)
