from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['enterprise_db']
attendance_collection = db['attendance']

print("All attendance records:")
for doc in attendance_collection.find():
    print(doc)

print("\nAttendance where status is absent:")
for doc in attendance_collection.find({"status": "absent"}):
    print(doc)
