from pymongo import MongoClient
from collections import Counter

client = MongoClient("mongodb://localhost:27017/")
db = client['enterprise_db']
attendance_collection = db['attendance']

# Load all attendance records
records = list(attendance_collection.find())

# Count total attendance records
total_records = len(records)
print(f"Total attendance records: {total_records}")

# Count absences grouped by employee
absences = [r for r in records if r['status'] == 'absent']
absences_count = Counter(r['employee_id'] for r in absences)
print("Absences count per employee:")
for emp_id, count in absences_count.items():
    print(f"Employee {emp_id}: {count} absences")

# Detect employees with more than a threshold of absences (e.g. 1)
threshold = 1
frequent_absentees = [emp_id for emp_id, count in absences_count.items() if count > threshold]
print(f"Employees with absences more than {threshold}: {frequent_absentees}")

# Count attendance status distribution
status_distribution = Counter(r['status'] for r in records)
print("Attendance status distribution:")
for status, count in status_distribution.items():
    print(f"{status}: {count}")
