<<<<<<< HEAD
# 📊 EduAnalytics Pro — AI-Powered Instructor & Revenue Analytics Dashboard

An interactive Business Intelligence dashboard built with **Streamlit** for analyzing an online education platform's courses, instructors, and transaction data — featuring automated business insights, instructor leaderboards, revenue forecasting, and machine learning predictions.

---

## 📁 Dataset

The app uses three CSV files:

| File | Description |
|---|---|
| `Courses.csv` | Course catalog — category, type, level, price, duration, rating |
| `Teachers.csv` | Instructor profiles — age, gender, expertise, experience, rating |
| `Transactions.csv` | Purchase records — user, course, teacher, amount, payment method, date |

---

## 🎯 Objectives

- Provide an executive-level, automatically generated business summary
- Track instructor and course performance with leaderboards and drill-downs
- Forecast future revenue using regression modeling
- Predict teacher ratings and course revenue using machine learning
- Present a polished, recruiter-ready BI dashboard experience

---

## ✨ Features

- **📈 AI Business Summary** — auto-generated executive insights (revenue trend, top category, payment method, rating-vs-enrollment correlation)
- **🧠 Business Insights Panel** — top instructor, weakest category, best month, top payment method
- **👨‍🏫 Instructor Leaderboard** — ranked with 🥇🥈🥉 highlights and individual teacher profile cards
- **📚 Course Explorer** — category → course → transaction drill-down navigation
- **💰 Revenue & Forecast** — advanced KPIs (avg price, revenue/teacher, repeat purchase rate) and a 6-month revenue forecast with confidence interval
- **🤖 ML Insights** — Random Forest models predicting Teacher Rating or Course Revenue, with feature importance and live prediction inputs
- **📊 KPI Cards with Growth Indicators** — ↑ / ↓ arrows vs. previous month
- **🔗 Correlation Heatmap** — price, duration, rating, revenue, enrollments
- **🔍 Search & Filter** — search by teacher/course name, custom date range filter
- **📤 Export Reports** — download data as CSV, Excel, or PDF
- **🌗 Dark / Light Theme** — consistent, transparent-background Plotly charts with purple accents
- **🔐 Simple Login Page** — basic email/password gate (not a production auth system)

---

## 🛠️ Technologies Used

- Python
- Pandas / NumPy
- Plotly
- Streamlit
- Scikit-learn
- XlsxWriter (Excel export)
- fpdf2 (PDF export)

---

## 🚀 Setup & Installation

1. **Clone or download this project folder.**

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure the data files are in the project root** (same folder as `app.py`):
   ```
   EduPro_Instructor_Analysis/
   ├── app.py
   ├── requirements.txt
   ├── Courses.csv
   ├── Teachers.csv
   └── Transactions.csv
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. Open the link shown in the terminal (typically `http://localhost:8501`) in your browser.

---

## 📂 Project Structure

```
EduPro_Instructor_Analysis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md            # Project documentation
├── Courses.csv          # Course dataset
├── Teachers.csv         # Instructor dataset
└── Transactions.csv     # Transaction dataset
```

---

## 📌 Notes

- The login page is a simple UI gate, not real authentication — any email/password (or "Continue as Guest") will grant access.
- The revenue forecast uses linear regression on monthly totals; results are illustrative since the dataset covers roughly a year of data.
- PDF export requires `fpdf2` to be installed; if it's missing, the PDF button is disabled and CSV/Excel export still work.

---

## 👤 Author
**Ratika Ahuja**
=======
# EduAnalytics-Pro
A data-driven framework for evaluating instructor and course quality on EduPro, shifting focus from learners to teaching effectiveness.
>>>>>>> 28e07335f733ac830c34373c7ef42b29e6f97d22
