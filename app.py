import streamlit as st
from pymongo import MongoClient
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import io
import datetime as dt
import os
import base64
from pathlib import Path


st.set_page_config(page_title="Attendance Insight & Predict", layout="wide")

# ---------- Background image (local path) ----------
BG_PATH = r"C:\Users\user\Downloads\STAGE\bck.jpg"

def set_bg_local(image_path: str, add_overlay: bool = True):
    p = Path(image_path)
    if not p.exists():
        st.warning(f"Background image not found: {image_path}")
        return
    with open(p, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{encoded}) no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    """
    if add_overlay:
        css += """
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.25);
        z-index: -1;
    }
    """
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)

set_bg_local(BG_PATH, add_overlay=True)

st.title("Attendance Insight & Predict")
st.caption("End-to-end demo: data seeding, CSV import, feature engineering, model training, prediction, and visualization.")

# ---------- Safe secrets/env fallback for Mongo URI ----------
def get_mongo_uri():
    try:
        if "MONGODB_URI" in st.secrets:
            return st.secrets["MONGODB_URI"]
    except Exception:
        pass
    return os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

MONGO_URI = get_mongo_uri()
client = MongoClient(MONGO_URI)
db = client['enterprise_db']

# ---------- Utilities ----------
def ensure_indexes():
    db.attendance.create_index([("employee_id", 1), ("date", 1)])
    db.employees.create_index("employee_id", unique=True)

# ---------- Simple seed of 20 employees ----------
def insert_initial_data_simple():
    employees = [
        {"employee_id": 1, "name": "Alice"},
        {"employee_id": 2, "name": "Bob"},
        {"employee_id": 3, "name": "Charlie"},
        {"employee_id": 4, "name": "Diana"},
        {"employee_id": 5, "name": "Ethan"},
        {"employee_id": 6, "name": "Fatima"},
        {"employee_id": 7, "name": "George"},
        {"employee_id": 8, "name": "Hana"},
        {"employee_id": 9, "name": "Ivan"},
        {"employee_id": 10, "name": "Julia"},
        {"employee_id": 11, "name": "Karl"},
        {"employee_id": 12, "name": "Layla"},
        {"employee_id": 13, "name": "Marco"},
        {"employee_id": 14, "name": "Nina"},
        {"employee_id": 15, "name": "Omar"},
        {"employee_id": 16, "name": "Priya"},
        {"employee_id": 17, "name": "Quentin"},
        {"employee_id": 18, "name": "Rosa"},
        {"employee_id": 19, "name": "Samir"},
        {"employee_id": 20, "name": "Tara"},
    ]
    db.employees.delete_many({})
    db.employees.insert_many(employees)
    st.success("Inserted initial employee data (20 employees).")

# ---------- Synthetic attendance seeding ----------
def seed_synthetic_attendance(n_employees=15, start_date="2025-08-01", days=45):
    employees = [{"employee_id": i, "name": f"Emp_{i}"} for i in range(1, n_employees + 1)]
    db.employees.delete_many({})
    db.employees.insert_many(employees)
    db.attendance.delete_many({})
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []
    for d in range(days):
        day_dt = start + timedelta(days=d)
        date = day_dt.strftime("%Y-%m-%d")
        weekday = day_dt.weekday()
        weekend = 1 if weekday >= 5 else 0
        for e in employees:
            shift_type = "day" if random.random() < 0.6 else "night"
            p = 0.07
            if shift_type == "night":
                p += 0.05
            if weekend:
                p += 0.04
            status = "absent" if random.random() < p else "present"
            rows.append({
                "employee_id": e["employee_id"],
                "date": date,
                "status": status,
                "shift_id": 1,
                "shift_type": shift_type
            })
    if rows:
        db.attendance.insert_many(rows)
    ensure_indexes()
    st.success(f"Seeded {len(rows)} attendance rows for {n_employees} employees across {days} days.")

# ---------- Simple browsing/analysis ----------
def query_attendance():
    attendance = list(db.attendance.find())
    if not attendance:
        st.info("No attendance data found.")
        return
    st.dataframe(pd.DataFrame(attendance))

def analyze_attendance():
    records = list(db.attendance.find())
    if not records:
        st.info("No attendance data found.")
        return
    df = pd.DataFrame(records)
    df['status_bin'] = df['status'].eq('absent').astype(int)
    counts = df[df['status_bin'] == 1]['employee_id'].value_counts()
    st.write("Absences per employee:")
    st.dataframe(counts.rename_axis('employee_id').reset_index(name='count'))

# ---------- Feature engineering ----------
def feature_engineering():
    records = list(db.attendance.find())
    if not records:
        st.info("No attendance data found.")
        return
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df['status_bin'] = df['status'].eq('absent').astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['shift_type_bin'] = df.get('shift_type', pd.Series(['day'] * len(df))).eq('night').astype(int)
    df = df.sort_values(['employee_id', 'date'])
    df['cum_absences'] = df.groupby('employee_id')['status_bin'].cumsum() - df['status_bin']
    df['roll7_absences'] = (df.groupby('employee_id')['status_bin']
                              .rolling(window=7, min_periods=1).sum()
                              .shift(1)
                              .reset_index(0, drop=True))
    df['roll7_absences'] = df['roll7_absences'].fillna(0)
    cols = ['employee_id', 'date', 'status_bin', 'day_of_week', 'is_weekend',
            'month', 'shift_type_bin', 'cum_absences', 'roll7_absences']
    df_feat = df[cols].copy()
    df_feat.to_csv("attendance_features.csv", index=False)
    st.success("Feature engineering done, saved to attendance_features.csv")
    with st.expander("Preview engineered features"):
        st.dataframe(df_feat.head(20))

# ---------- Training and evaluation ----------
def train_and_evaluate(test_size=0.25, random_state=42, balance_weight=1.0):
    try:
        df = pd.read_csv("attendance_features.csv")
    except FileNotFoundError:
        st.error("attendance_features.csv not found. Run Feature engineering first.")
        return
    required_columns = ['day_of_week','is_weekend','month','shift_type_bin','cum_absences','roll7_absences','status_bin']
    if not all(c in df.columns for c in required_columns):
        st.error("Required feature columns missing. Please re-run Feature engineering.")
        return
    X = df[['day_of_week','is_weekend','month','shift_type_bin','cum_absences','roll7_absences']]
    y = df['status_bin']
    st.write("Dataset shape:", X.shape)
    st.write("Class balance:", y.value_counts(dropna=False))
    if y.nunique() < 2:
        st.warning("Only one class in target. Seed more varied data and re-run.")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = LogisticRegression(max_iter=500, class_weight={1: balance_weight, 0: 1}, solver='liblinear')
    model.fit(X_train, y_train)
    threshold = 0.3
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    st.success("Model trained (train/validation split).")
    st.caption("Coefficients (class 1 vs 0)")
    st.dataframe(pd.DataFrame(model.coef_, columns=X.columns))
    st.caption("Intercept")
    st.write(model.intercept_)
    with st.expander("Validation classification report"):
        st.text(classification_report(y_test, y_pred, digits=3))
    st.subheader("Validation confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=(0,1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (Validation)")
    st.pyplot(fig)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax2.plot([0,1],[0,1],'k--',alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC curve (Validation)")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# ---------- Heatmap visualization ----------
def plot_monthly_absence_heatmap():
    records = list(db.attendance.find())
    if not records:
        st.info("No attendance data found.")
        return
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df['status_bin'] = df['status'].eq('absent').astype(int)
    df['month'] = df['date'].dt.month
    absences_by_month = df.groupby(['employee_id', 'month'])['status_bin'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(absences_by_month, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_title("Monthly Absences Heatmap")
    st.pyplot(fig)

# ---------- Export ----------
def export_attendance_csv():
    records = list(db.attendance.find())
    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No data to export.")
        return
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("Download CSV", csv_buffer.getvalue(), "attendance.csv", "text/csv")

# ---------- Prediction UI ----------
def predict_future_absences_ui():
    st.subheader("Predict future absences")
    start_date = st.date_input("Start date", dt.date.today())
    end_date = st.date_input("End date", dt.date.today() + dt.timedelta(days=7))
    threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return
    employees = list(db.employees.find())
    if not employees:
        st.error("No employees found. Please import/seed them first.")
        return
    if st.button("Run prediction"):
        try:
            df_hist = pd.read_csv("attendance_features.csv")
            X_train = df_hist[['day_of_week','is_weekend','month','shift_type_bin','cum_absences','roll7_absences']]
            y_train = df_hist['status_bin']
            model = LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
            model.fit(X_train, y_train)
        except Exception as e:
            st.error("Error loading/training model. Run Feature engineering and Train & evaluate first.")
            st.error(str(e))
            return
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist_sorted = df_hist.sort_values(['employee_id', 'date'])
        rows = []
        for emp in employees:
            emp_id = emp['employee_id']
            emp_hist = df_hist_sorted[df_hist_sorted['employee_id'] == emp_id]
            current = start_date
            while current <= end_date:
                day_of_week = current.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                month = current.month
                current_ts = pd.Timestamp(current)
                start_rolling = current - dt.timedelta(days=7)
                start_rolling_ts = pd.Timestamp(start_rolling)
                cum_abs = emp_hist[emp_hist['date'] < current_ts]['status_bin'].sum()
                roll7 = emp_hist[(emp_hist['date'] >= start_rolling_ts) & (emp_hist['date'] < current_ts)]['status_bin'].sum()
                features = np.array([[day_of_week, is_weekend, month, 0, cum_abs, roll7]])
                prob_absent = model.predict_proba(features)[0][1]
                if prob_absent >= threshold:
                    rows.append({
                        "employee_id": emp_id,
                        "name": emp.get('name', f"Emp_{emp_id}"),
                        "date": current.strftime("%Y-%m-%d"),
                        "absence_probability": prob_absent
                    })
                current += dt.timedelta(days=1)
        if not rows:
            st.info(f"No absences predicted with threshold >= {threshold}")
            return
        df_pred = pd.DataFrame(rows)
        df_pred_sorted = df_pred.sort_values(by='absence_probability', ascending=False)
        df_pred_unique = df_pred_sorted.drop_duplicates(subset=['employee_id', 'date'])
        st.write(f"Predicted absences between {start_date} and {end_date} with probability >= {threshold}:")
        st.dataframe(df_pred_unique.reset_index(drop=True))

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Navigation")
    action = st.selectbox("Choose action", [
        "Seed synthetic data",
        "Insert initial data (simple)",
        "Import employees CSV",
        "Query attendance",
        "Analyze attendance",
        "Feature engineering",
        "Train & evaluate (split)",
        "Predict future absences",
        "Dashboard summary",
        "Heatmap: monthly absences",
        "Export attendance CSV",
    ])

# ---------- Routing ----------
if action == "Seed synthetic data":
    n_emp = st.number_input("Number of employees", min_value=5, max_value=500, value=20, step=1)
    days = st.number_input("Number of days", min_value=7, max_value=365, value=45, step=1)
    start = st.text_input("Start date (YYYY-MM-DD)", value="2025-08-01")
    if st.button("Run"):
        seed_synthetic_attendance(n_employees=int(n_emp), start_date=start, days=int(days))

elif action == "Insert initial data (simple)":
    if st.button("Run"):
        insert_initial_data_simple()

elif action == "Import employees CSV":
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            # TODO: Map columns and upsert to db.employees as needed.
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

elif action == "Query attendance":
    query_attendance()

elif action == "Analyze attendance":
    analyze_attendance()

elif action == "Feature engineering":
    feature_engineering()

elif action == "Train & evaluate (split)":
    test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)
    balance_weight = st.slider("Class 1 weight", 0.5, 20.0, 1.0, 0.1)
    if st.button("Run"):
        train_and_evaluate(test_size=float(test_size), random_state=int(random_state), balance_weight=float(balance_weight))

elif action == "Predict future absences":
    predict_future_absences_ui()

elif action == "Dashboard summary":
    records = list(db.attendance.find())
    df = pd.DataFrame(records)
    if df.empty:
        st.info("No attendance data found.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df['status_bin'] = df['status'].eq('absent').astype(int)
        st.metric("Total absences", df['status_bin'].sum())
        st.line_chart(df.groupby('date')['status_bin'].sum())

elif action == "Heatmap: monthly absences":
    plot_monthly_absence_heatmap()

elif action == "Export attendance CSV":
    export_attendance_csv()
