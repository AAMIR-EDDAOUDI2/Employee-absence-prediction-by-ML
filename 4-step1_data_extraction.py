from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
db = client['enterprise_db']

# Query all attendance records
attendance_docs = list(db.attendance.find())

# Convert MongoDB documents to pandas DataFrame
attendance_df = pd.DataFrame(attendance_docs)

# Optionally, convert date from string to datetime
attendance_df['date'] = pd.to_datetime(attendance_df['date'])

# Example: Map status to binary (present=0, absent=1) for ML purposes
attendance_df['status_bin'] = attendance_df['status'].apply(lambda x: 1 if x == 'absent' else 0)

print(attendance_df.head())

# Save to CSV file if needed for inspection or other tools
attendance_df.to_csv("attendance_dataset.csv", index=False)
