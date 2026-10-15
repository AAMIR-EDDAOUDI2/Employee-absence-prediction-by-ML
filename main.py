from pymongo import MongoClient
import pandas as pd
from sklearn.linear_model import LogisticRegression
import menu_nav

client = MongoClient("mongodb://localhost:27017/")
db = client['enterprise_db']

def insert_initial_data():
    employees = [
        {"employee_id": 1, "name": "Alice"},
        {"employee_id": 2, "name": "Bob"}
    ]
    db.employees.insert_many(employees)
    print("Inserted initial employee data.")

def query_attendance():
    attendance = list(db.attendance.find())
    for record in attendance:
        print(record)

def analyze_attendance():
    records = list(db.attendance.find())
    df = pd.DataFrame(records)
    if df.empty:
        print("No attendance data found.")
        return
    df['status_bin'] = df['status'].apply(lambda x: 1 if x == 'absent' else 0)
    absences_count = df[df['status_bin'] == 1]['employee_id'].value_counts()
    print("Absences per employee:")
    print(absences_count)

def feature_engineering():
    records = list(db.attendance.find())
    df = pd.DataFrame(records)
    if df.empty:
        print("No attendance data found.")
        return
    df['date'] = pd.to_datetime(df['date'])
    df['status_bin'] = df['status'].apply(lambda x: 1 if x == 'absent' else 0)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df = df.sort_values(['employee_id', 'date'])
    df['cum_absences'] = df.groupby('employee_id')['status_bin'].cumsum() - df['status_bin']
    df.to_csv("attendance_features.csv", index=False)
    print("Feature engineering done, saved to attendance_features.csv")

def train_model():
    df = pd.read_csv("attendance_features.csv")
    X = df[['day_of_week', 'is_weekend', 'cum_absences']]
    y = df['status_bin']
    model = LogisticRegression()
    model.fit(X, y)
    print("Model trained on entire data.")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def main():
    while True:
        menu_nav.display_menu()
        choice = menu_nav.get_user_choice()

        if choice == '1':
            insert_initial_data()
        elif choice == '2':
            query_attendance()
        elif choice == '3':
            analyze_attendance()
        elif choice == '4':
            feature_engineering()
        elif choice == '5':
            train_model()
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
