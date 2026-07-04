import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="EduAnalytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
PURPLE = "#a855f7"
PURPLE_LIGHT = "#c084fc"
GREEN = "#22c55e"
RED = "#ef4444"
def inject_css(theme: str):
    if theme == "Dark":
        bg, card, text, subtext, border = "#0e0e16", "#171723", "#f5f5f7", "#9ca3af", "#2a2a3c"
    else:
        bg, card, text, subtext, border = "#f5f5f9", "#ffffff", "#111111", "#555555", "#e2e2ea"
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    section[data-testid="stSidebar"] {{ background-color: {card}; border-right: 1px solid {border}; }}
    div[data-testid="stMetric"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 14px 18px;
    .kpi-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }}
    .insight-card {{
        background-color: {card};
        border: 1px solid {border};
        border-left: 4px solid {PURPLE};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .leader-1 {{ background: linear-gradient(90deg, #facc1530, transparent); border-left:4px solid #facc15; }}
    .leader-2 {{ background: linear-gradient(90deg, #94a3b830, transparent); border-left:4px solid #94a3b8; }}
    .leader-3 {{ background: linear-gradient(90deg, #b4530930, transparent); border-left:4px solid #b45309; }}
    h1, h2, h3 {{ color: {text}; }}
    .subtext {{ color: {subtext}; font-size: 0.85rem; }}
    .footer {{ text-align:center; color:{subtext}; padding: 30px 0 10px 0; font-size: 0.85rem; }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

def style_fig(fig):
    """Apply consistent dark/light theme styling to every Plotly chart."""
    font_color = "#f5f5f7" if st.session_state.theme == "Dark" else "#111111"
    grid_color = "#2a2a3c" if st.session_state.theme == "Dark" else "#e2e2ea"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        colorway=[PURPLE, PURPLE_LIGHT, "#7c3aed", "#d8b4fe", "#6d28d9", "#f0abfc"],
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return fig
@st.cache_data
def load_data():
    courses = pd.read_csv("Courses.csv")
    teachers = pd.read_csv("Teachers.csv")
    transactions = pd.read_csv("Transactions.csv")

    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"], format="%d/%m/%Y")
    transactions["Month"] = transactions["TransactionDate"].dt.to_period("M").astype(str)
    transactions["MonthName"] = transactions["TransactionDate"].dt.strftime("%B %Y")

    df = transactions.merge(courses, on="CourseID", how="left")
    df = df.merge(teachers, on="TeacherID", how="left")
    return courses, teachers, transactions, df

courses, teachers, transactions, df = load_data()
with st.sidebar:
    st.markdown("## 📊 EduAnalytics Pro")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "👨‍🏫 Instructors", "📐 Quality Analysis", "📚 Courses", "💰 Revenue & Forecast",
         "🤖 ML Insights", "⚙ Settings", "ℹ About"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("### 🔍 Search")
    search_teacher = st.text_input("Search Teacher", placeholder="e.g. John")
    search_course = st.text_input("Search Course", placeholder="e.g. Python")

    st.divider()
    st.markdown("### 📅 Date Filter")
    min_date, max_date = transactions["TransactionDate"].min(), transactions["TransactionDate"].max()
    date_range = st.date_input("Custom Date Range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)

    st.divider()
    st.markdown("### 🎯 Filters")
    expertise_options = sorted(teachers["Expertise"].dropna().unique())
    selected_expertise = st.multiselect("Instructor Expertise", expertise_options, default=expertise_options)

    category_options = sorted(courses["CourseCategory"].dropna().unique())
    selected_categories = st.multiselect("Course Category", category_options, default=category_options)

    level_options = sorted(courses["CourseLevel"].dropna().unique())
    selected_levels = st.multiselect("Course Level", level_options, default=level_options)

    rating_min, rating_max = float(teachers["TeacherRating"].min()), float(teachers["TeacherRating"].max())
    rating_range = st.slider("Instructor Rating Range", min_value=round(rating_min, 1), max_value=round(rating_max, 1),
                              value=(round(rating_min, 1), round(rating_max, 1)), step=0.1)

    st.divider()
    theme_choice = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    new_theme = "Dark" if theme_choice else "Light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# apply date filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    dff = df[(df["TransactionDate"] >= start) & (df["TransactionDate"] <= end)].copy()
else:
    dff = df.copy()

# apply expertise / category / level / rating filters
dff = dff[
    dff["Expertise"].isin(selected_expertise)
    & dff["CourseCategory"].isin(selected_categories)
    & dff["CourseLevel"].isin(selected_levels)
    & dff["TeacherRating"].between(rating_range[0], rating_range[1])
]

teachers_f = teachers[
    teachers["Expertise"].isin(selected_expertise)
    & teachers["TeacherRating"].between(rating_range[0], rating_range[1])
].copy()

courses_f = courses[
    courses["CourseCategory"].isin(selected_categories)
    & courses["CourseLevel"].isin(selected_levels)
].copy()

# apply search filters
if search_teacher:
    dff = dff[dff["TeacherName"].str.contains(search_teacher, case=False, na=False)]
    teachers_f = teachers_f[teachers_f["TeacherName"].str.contains(search_teacher, case=False, na=False)]
if search_course:
    dff = dff[dff["CourseName"].str.contains(search_course, case=False, na=False)]
    courses_f = courses_f[courses_f["CourseName"].str.contains(search_course, case=False, na=False)]
def month_over_month(data, value_col="Amount"):
    monthly = data.groupby("Month")[value_col].sum().sort_index()
    if len(monthly) < 2:
        return monthly.iloc[-1] if len(monthly) else 0, 0.0
    current, previous = monthly.iloc[-1], monthly.iloc[-2]
    pct = ((current - previous) / previous * 100) if previous else 0
    return current, pct

def kpi_card(label, value, delta=None, icon=""):
    arrow = ""
    color = GREEN
    if delta is not None:
        if delta < 0:
            arrow, color = "↓", RED
        elif delta > 0:
            arrow, color = "↑", GREEN
        else:
            arrow, color = "→", "#9ca3af"
    delta_html = f"<span style='color:{color}; font-weight:600;'>{arrow} {abs(delta):.1f}%</span> <span class='subtext'>vs last month</span>" if delta is not None else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="subtext">{icon} {label}</div>
        <div style="font-size:1.7rem; font-weight:700;">{value}</div>
        <div>{delta_html}</div>
    </div>
    """, unsafe_allow_html=True)

def to_excel_bytes(d: pd.DataFrame):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        d.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()

def export_buttons(data: pd.DataFrame, name: str):
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ CSV", data.to_csv(index=False).encode(), f"{name}.csv", "text/csv", use_container_width=True)
    c2.download_button("⬇ Excel", to_excel_bytes(data), f"{name}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    try:
        from fpdf import FPDF

        def clean(text):
            # Strip emojis/unicode that the base PDF font can't render,
            # and hard-wrap very long unbroken strings.
            text = str(text).encode("latin-1", "ignore").decode("latin-1")
            return text if text.strip() else "-"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 10, clean(f"{name} Report - {datetime.now().strftime('%Y-%m-%d')}"), ln=True)
        pdf.ln(2)

        cols = list(data.columns)[:6]
        col_width = (pdf.w - 20) / max(len(cols), 1)

        pdf.set_font("Helvetica", "B", 8)
        for col in cols:
            pdf.cell(col_width, 7, clean(col)[:18], border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=8)
        for _, row in data.head(40)[cols].iterrows():
            for v in row.values:
                pdf.cell(col_width, 6, clean(v)[:18], border=1)
            pdf.ln()

        pdf_bytes = bytes(pdf.output(dest="S"))
        c3.download_button("⬇ PDF", pdf_bytes, f"{name}.pdf", "application/pdf", use_container_width=True)
    except ImportError:
        c3.button("⬇ PDF (install fpdf2)", disabled=True, use_container_width=True)
    except Exception:
        c3.button("⬇ PDF (error)", disabled=True, use_container_width=True)
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")

    total_revenue = dff["Amount"].sum()
    total_students = dff["UserID"].nunique()
    total_teachers = teachers["TeacherID"].nunique()
    total_courses = courses["CourseID"].nunique()
    avg_rating = teachers["TeacherRating"].mean()

    rev_now, rev_pct = month_over_month(dff, "Amount")
    cnt_monthly = dff.groupby("Month")["TransactionID"].count().sort_index()
    cnt_pct = ((cnt_monthly.iloc[-1] - cnt_monthly.iloc[-2]) / cnt_monthly.iloc[-2] * 100) if len(cnt_monthly) > 1 and cnt_monthly.iloc[-2] else 0

    top_category = dff.groupby("CourseCategory")["Amount"].sum().idxmax() if len(dff) else "N/A"
    weakest_category = dff.groupby("CourseCategory")["Amount"].sum().idxmin() if len(dff) else "N/A"
    top_payment = dff["PaymentMethod"].mode()[0] if len(dff) else "N/A"
    best_month = dff.groupby("MonthName")["Amount"].sum().idxmax() if len(dff) else "N/A"

    high_rated = teachers[teachers["TeacherRating"] > 4.5]["TeacherID"]
    high_rated_enroll = dff[dff["TeacherID"].isin(high_rated)].groupby("TeacherID")["TransactionID"].count().mean()
    other_enroll = dff[~dff["TeacherID"].isin(high_rated)].groupby("TeacherID")["TransactionID"].count().mean()
    enroll_uplift = ((high_rated_enroll - other_enroll) / other_enroll * 100) if other_enroll else 0
    st.markdown(f"""
    <div class="insight-card">
        <h3>📈 AI Business Summary</h3>
        <ul>
            <li>Revenue is <b>{'up' if rev_pct >= 0 else 'down'} {abs(rev_pct):.1f}%</b> compared to the previous month.</li>
            <li><b>{top_category}</b> courses generated the highest revenue this period.</li>
            <li>Teachers rated above 4.5 see <b>{enroll_uplift:+.0f}%</b> {'higher' if enroll_uplift >= 0 else 'lower'} average enrollments than others.</li>
            <li><b>{top_payment}</b> is the most used payment method ({(dff['PaymentMethod']==top_payment).mean()*100:.0f}% of transactions).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("Revenue", f"${total_revenue/1000:,.1f}K", rev_pct, "💰")
    with k2: kpi_card("Transactions", f"{len(dff):,}", cnt_pct, "🧾")
    with k3: kpi_card("Students", f"{total_students:,}", icon="🎓")
    with k4: kpi_card("Teachers", f"{total_teachers}", icon="👨‍🏫")
    with k5: kpi_card("Avg Rating", f"{avg_rating:.2f} ⭐", icon="📊")

    st.markdown("####")

    # Business Insights Panel
    st.subheader("🧠 Business Insights")
    top_teacher_row = dff.groupby(["TeacherID", "TeacherName"]).agg(Revenue=("Amount", "sum")).reset_index()
    top_teacher_row = top_teacher_row.merge(teachers[["TeacherID", "TeacherRating"]], on="TeacherID").sort_values("Revenue", ascending=False)
    ic1, ic2, ic3, ic4 = st.columns(4)
    if len(top_teacher_row):
        t = top_teacher_row.iloc[0]
        ic1.markdown(f"""<div class="insight-card"><b>🏆 Top Instructor</b><br>{t['TeacherName']}<br>
        Rating: {t['TeacherRating']:.2f} ⭐<br>Revenue: ${t['Revenue']:,.0f}</div>""", unsafe_allow_html=True)
    ic2.markdown(f"""<div class="insight-card"><b>📉 Weakest Category</b><br>{weakest_category}</div>""", unsafe_allow_html=True)
    ic3.markdown(f"""<div class="insight-card"><b>📅 Best Month</b><br>{best_month}</div>""", unsafe_allow_html=True)
    ic4.markdown(f"""<div class="insight-card"><b>💳 Top Payment Method</b><br>{top_payment}</div>""", unsafe_allow_html=True)

    st.markdown("####")
    c1, c2 = st.columns(2)
    with c1:
        rev_by_month = dff.groupby("Month")["Amount"].sum().reset_index()
        fig = px.area(rev_by_month, x="Month", y="Amount", title="Revenue Trend", markers=True)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        rev_by_cat = dff.groupby("CourseCategory")["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
        fig = px.bar(rev_by_cat, x="CourseCategory", y="Amount", title="Revenue by Category")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        pay = dff["PaymentMethod"].value_counts().reset_index()
        pay.columns = ["PaymentMethod", "Count"]
        fig = px.pie(pay, names="PaymentMethod", values="Count", hole=0.55, title="Payment Method Split")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c4:
        st.markdown("**Correlation Heatmap**")
        merged_for_corr = courses.merge(
            transactions.groupby("CourseID").agg(Revenue=("Amount", "sum"), Enrollments=("TransactionID", "count")).reset_index(),
            on="CourseID", how="left"
        ).fillna(0)
        corr_cols = merged_for_corr[["CoursePrice", "CourseDuration", "CourseRating", "Revenue", "Enrollments"]]
        corr = corr_cols.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples", title="Course Metric Correlations")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("📤 Export This View")
    export_buttons(dff, "dashboard_data")
elif page == "👨‍🏫 Instructors":
    st.title("👨‍🏫 Instructor Leaderboard")

    rev_agg = dff.groupby("TeacherID").agg(
        Revenue=("Amount", "sum"),
        Students=("UserID", "nunique"),
        Transactions=("TransactionID", "count"),
    ).reset_index()

    perf = teachers_f.merge(rev_agg, on="TeacherID", how="left").fillna(
        {"Revenue": 0, "Students": 0, "Transactions": 0}
    )
    perf = perf.sort_values("Revenue", ascending=False).reset_index(drop=True)
    perf.index += 1

    if perf.empty:
        st.warning("No instructors match the current filters. Try adjusting the filters in the sidebar.")
    else:
        medals = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(perf))):
            row = perf.iloc[i]
            st.markdown(f"""
            <div class="insight-card leader-{i+1}">
                <span style="font-size:1.4rem;">{medals[i]} <b>{row['TeacherName']}</b></span><br>
                <span class="subtext">{row['Expertise']} • {row['YearsOfExperience']} yrs experience</span><br>
                Rating: <b>{row['TeacherRating']:.2f} ⭐</b> &nbsp;|&nbsp;
                Revenue: <b>${row['Revenue']:,.0f}</b> &nbsp;|&nbsp;
                Students: <b>{int(row['Students'])}</b>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Full Leaderboard")
        show = perf.copy()
        show["Revenue"] = show["Revenue"].apply(lambda x: f"${x:,.0f}")
        show["TeacherRating"] = show["TeacherRating"].round(2)
        st.dataframe(
            show[["TeacherName", "Expertise", "TeacherRating", "Revenue", "Students", "YearsOfExperience"]],
            use_container_width=True, height=400
        )

        st.subheader("📊 Experience vs Rating")
        fig = px.scatter(perf, x="YearsOfExperience", y="TeacherRating", size="Revenue", color="Expertise",
                          hover_name="TeacherName", title="Years of Experience vs Teacher Rating (bubble = revenue)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(perf, x="TeacherRating", y="Revenue", size="Students", color="Expertise",
                              hover_name="TeacherName", title="Rating vs Revenue (bubble = students)")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            exp_rev = perf.groupby("Expertise")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
            fig = px.bar(exp_rev, x="Expertise", y="Revenue", title="Revenue by Expertise Area")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("🧮 Expertise-wise Performance Comparison")
        exp_compare = perf.groupby("Expertise").agg(
            AvgRating=("TeacherRating", "mean"),
            AvgExperience=("YearsOfExperience", "mean"),
            TotalRevenue=("Revenue", "sum"),
            TotalStudents=("Students", "sum"),
            Instructors=("TeacherID", "count"),
        ).reset_index().sort_values("TotalRevenue", ascending=False)
        exp_compare_display = exp_compare.copy()
        exp_compare_display["AvgRating"] = exp_compare_display["AvgRating"].round(2)
        exp_compare_display["AvgExperience"] = exp_compare_display["AvgExperience"].round(1)
        exp_compare_display["TotalRevenue"] = exp_compare_display["TotalRevenue"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(exp_compare_display, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(exp_compare, x="Expertise", y="AvgRating", title="Average Rating by Expertise",
                         color="AvgRating", color_continuous_scale="Purples")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c4:
            fig = px.bar(exp_compare, x="Expertise", y="Instructors", title="Number of Instructors by Expertise")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("🌡️ Course Quality Heatmap")
        quality = courses_f.groupby(["CourseCategory", "CourseLevel"])["CourseRating"].mean().reset_index()
        if not quality.empty:
            pivot = quality.pivot(index="CourseCategory", columns="CourseLevel", values="CourseRating")
            fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Purples", aspect="auto",
                             title="Average Course Rating by Category & Level")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("No courses match the current category/level filters.")

        st.subheader("🔎 Teacher Profile")
        chosen = st.selectbox("Select a teacher", perf["TeacherName"].tolist())
        if chosen:
            row = perf[perf["TeacherName"] == chosen].iloc[0]
            full = teachers[teachers["TeacherName"] == chosen].iloc[0]
            tcourses = dff[dff["TeacherName"] == chosen]
            best_cat = tcourses.groupby("CourseCategory")["Amount"].sum().idxmax() if len(tcourses) else "N/A"
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Age", int(full["Age"]))
            p2.metric("Experience", f"{full['YearsOfExperience']} yrs")
            p3.metric("Rating", f"{full['TeacherRating']:.2f} ⭐")
            p4.metric("Revenue", f"${row['Revenue']:,.0f}")
            st.markdown(f"**Best Category:** {best_cat}  •  **Gender:** {full['Gender']}  •  **Students Taught:** {int(row['Students'])}")

        export_buttons(perf, "instructor_leaderboard")
elif page == "📐 Quality Analysis":
    st.title("📐 Instructor & Course Quality Analysis")
    st.caption("Answering EduPro's core analytical questions: rating distributions, experience effects, "
               "instructor impact on course success, and expertise-based quality patterns.")

    # Build teacher-level course-quality view from filtered data
    tc_pairs = dff.drop_duplicates(subset=["TeacherID", "CourseID"])[
        ["TeacherID", "TeacherName", "CourseID", "CourseRating", "CourseCategory", "CourseLevel", "Gender", "Expertise"]
    ]
    avg_course_rating_by_teacher = tc_pairs.groupby("TeacherID")["CourseRating"].mean().rename("AvgCourseRating")
    quality_df = teachers_f.merge(avg_course_rating_by_teacher, on="TeacherID", how="left")
    enroll_by_teacher = dff.groupby("TeacherID")["TransactionID"].count().rename("Enrollments")
    quality_df = quality_df.merge(enroll_by_teacher, on="TeacherID", how="left").fillna({"Enrollments": 0})
    st.subheader("📌 Key Performance Indicators")
    avg_teacher_rating = teachers_f["TeacherRating"].mean() if len(teachers_f) else 0
    avg_course_rating = courses_f["CourseRating"].mean() if len(courses_f) else 0
    rating_consistency = 1 / teachers_f["TeacherRating"].std() if len(teachers_f) > 1 and teachers_f["TeacherRating"].std() else 0
    exp_corr = teachers_f[["YearsOfExperience", "TeacherRating"]].corr().iloc[0, 1] if len(teachers_f) > 1 else 0

    quality_df["Tier"] = pd.cut(quality_df["TeacherRating"], bins=[0, 3, 4, 5], labels=["Low", "Mid", "High"])
    tier_enroll = quality_df.groupby("Tier", observed=True)["Enrollments"].mean()
    influence_ratio = (tier_enroll.get("High", 0) / tier_enroll.get("Low", 1)) if tier_enroll.get("Low", 0) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Teacher Rating", f"{avg_teacher_rating:.2f} ⭐")
    k2.metric("Avg Course Rating", f"{avg_course_rating:.2f} ⭐")
    k3.metric("Rating Consistency Index", f"{rating_consistency:.2f}", help="1 / std-dev of teacher ratings — higher = more consistent quality across instructors")
    k4.metric("Experience Impact Score", f"{exp_corr:+.2f}", help="Correlation between YearsOfExperience and TeacherRating")
    k5.metric("Enrollment Influence Ratio", f"{influence_ratio:.1f}x", help="Avg enrollments for High-rated (>4) instructors vs Low-rated (<3) instructors")

    st.divider()
    st.subheader("📊 Distribution of Instructor Ratings")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(teachers_f, x="TeacherRating", nbins=15, title="Instructor Rating Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.box(teachers_f, y="TeacherRating", points="all", title="Instructor Rating Spread (Box Plot)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("👥 Instructor Demographic Profile")
    c3, c4, c5 = st.columns(3)
    with c3:
        fig = px.histogram(teachers_f, x="Age", nbins=12, title="Age Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c4:
        fig = px.histogram(teachers_f, x="YearsOfExperience", nbins=12, title="Experience Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c5:
        exp_counts = teachers_f["Expertise"].value_counts().reset_index()
        exp_counts.columns = ["Expertise", "Count"]
        fig = px.bar(exp_counts, x="Expertise", y="Count", title="Expertise Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.subheader("📈 Experience vs Performance")
    c6, c7 = st.columns(2)
    with c6:
        fig = px.scatter(teachers_f, x="YearsOfExperience", y="TeacherRating", trendline="ols",
                          color="Expertise", title=f"Experience vs Teacher Rating (corr = {exp_corr:+.2f})")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c7:
        exp_course_corr = quality_df[["YearsOfExperience", "AvgCourseRating"]].corr().iloc[0, 1] if quality_df["AvgCourseRating"].notna().sum() > 1 else 0
        fig = px.scatter(quality_df, x="YearsOfExperience", y="AvgCourseRating", trendline="ols",
                          color="Expertise", title=f"Experience vs Course Rating (corr = {exp_course_corr:+.2f})")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("A flattening or declining trend line at higher experience levels indicates diminishing returns — "
               "i.e. additional tenure stops translating into higher ratings beyond a certain threshold.")

    # ── Instructor Rating vs Course Rating ──────────────────────────────
    st.subheader("🔗 Teacher Rating vs. Course Rating")
    tr_cr_corr = quality_df[["TeacherRating", "AvgCourseRating"]].corr().iloc[0, 1] if quality_df["AvgCourseRating"].notna().sum() > 1 else 0
    fig = px.scatter(quality_df, x="TeacherRating", y="AvgCourseRating", size="Enrollments", color="Expertise",
                      hover_name="TeacherName", trendline="ols",
                      title=f"Teacher Rating vs Average Rating of Their Courses (corr = {tr_cr_corr:+.2f})")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    if tr_cr_corr < 0:
        st.warning(f"⚠️ A negative correlation ({tr_cr_corr:+.2f}) suggests instructor popularity/rating and the "
                   "quality rating of the courses they teach are not well aligned in this dataset — worth deeper "
                   "investigation rather than assuming high instructor ratings guarantee high course ratings.")

    st.divider()
    st.subheader("🎓 Course Quality by Category & Level")
    c8, c9 = st.columns(2)
    with c8:
        cat_q = courses_f.groupby("CourseCategory")["CourseRating"].mean().reset_index().sort_values("CourseRating", ascending=False)
        fig = px.bar(cat_q, x="CourseRating", y="CourseCategory", orientation="h", title="Avg Course Rating by Category")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c9:
        level_q = courses_f.groupby("CourseLevel")["CourseRating"].mean().reset_index().sort_values("CourseRating", ascending=False)
        fig = px.bar(level_q, x="CourseLevel", y="CourseRating", title="Avg Course Rating by Level")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("⚧ Gender vs Course Level Comparison")
    gender_level = tc_pairs.groupby(["Gender", "CourseLevel"])["CourseRating"].mean().reset_index()
    if not gender_level.empty:
        fig = px.bar(gender_level, x="CourseLevel", y="CourseRating", color="Gender", barmode="group",
                     title="Average Course Rating by Instructor Gender & Course Level")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.subheader("🏆 Instructor Rating Tier vs Course Success")
    tier_summary = quality_df.groupby("Tier", observed=True).agg(
        AvgCourseRating=("AvgCourseRating", "mean"),
        AvgEnrollments=("Enrollments", "mean"),
        Instructors=("TeacherID", "count"),
    ).reset_index()
    tier_summary["Tier"] = pd.Categorical(tier_summary["Tier"], categories=["Low", "Mid", "High"], ordered=True)
    tier_summary = tier_summary.sort_values("Tier")
    c10, c11 = st.columns(2)
    with c10:
        fig = px.bar(tier_summary, x="Tier", y="AvgCourseRating", color="Tier",
                     title="Avg Course Rating by Instructor Rating Tier (Low <3, Mid 3–4, High >4)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c11:
        fig = px.bar(tier_summary, x="Tier", y="AvgEnrollments", color="Tier",
                     title="Avg Enrollments by Instructor Rating Tier")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.dataframe(tier_summary, use_container_width=True)

    st.divider()
    st.subheader("🧭 Expertise-Based Quality Insights")
    expertise_quality = quality_df.groupby("Expertise").agg(
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgCourseRating=("AvgCourseRating", "mean"),
        Instructors=("TeacherID", "count"),
    ).reset_index().sort_values("AvgCourseRating", ascending=False)

    fig = px.bar(expertise_quality, x="Expertise", y=["AvgTeacherRating", "AvgCourseRating"], barmode="group",
                 title="Teacher Rating vs Course Rating by Expertise Area")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    if len(expertise_quality):
        strongest = expertise_quality.iloc[0]
        weakest = expertise_quality.iloc[-1]
        st.markdown(f"""
        <div class="insight-card">
            <b>🟢 Strongest Quality Domain:</b> {strongest['Expertise']} (avg course rating {strongest['AvgCourseRating']:.2f})<br>
            <b>🔴 Expertise Gap / Needs Attention:</b> {weakest['Expertise']} (avg course rating {weakest['AvgCourseRating']:.2f})<br>
            <span class="subtext">Domains with a large gap between teacher rating and course rating may indicate
            content/design issues rather than instructor delivery issues, and vice versa.</span>
        </div>
        """, unsafe_allow_html=True)

    export_buttons(quality_df, "instructor_quality_analysis")
elif page == "📚 Courses":
    st.title("📚 Course Explorer")
    cat_perf = courses.merge(
        transactions.groupby("CourseID").agg(Revenue=("Amount", "sum"), Enrollments=("TransactionID", "count")).reset_index(),
        on="CourseID", how="left"
    ).fillna(0)

    st.subheader("Step 1 — Pick a Category")
    categories = sorted(cat_perf["CourseCategory"].unique())
    chosen_cat = st.selectbox("Category", ["All"] + categories)
    sub = cat_perf if chosen_cat == "All" else cat_perf[cat_perf["CourseCategory"] == chosen_cat]

    fig = px.bar(sub.sort_values("Revenue", ascending=False), x="CourseName", y="Revenue",
                 color="CourseLevel", title=f"Course Revenue — {chosen_cat}")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Step 2 — Pick a Course")
    course_name = st.selectbox("Course", sub["CourseName"].tolist())
    if course_name:
        crow = sub[sub["CourseName"] == course_name].iloc[0]
        ctx = dff[dff["CourseName"] == course_name]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Revenue", f"${crow['Revenue']:,.0f}")
        c2.metric("Enrollments", int(crow["Enrollments"]))
        c3.metric("Rating", f"{crow['CourseRating']:.2f} ⭐")
        c4.metric("Price", f"${crow['CoursePrice']:.0f}")
        completion = np.clip(crow["CourseRating"] / 5 * 100, 0, 100)
        c5.metric("Est. Completion", f"{completion:.0f}%")

        st.markdown("**Step 3 — Transactions for this course**")
        st.dataframe(ctx[["TransactionID", "UserID", "TeacherName", "TransactionDate", "Amount", "PaymentMethod"]],
                     use_container_width=True, height=300)
        export_buttons(ctx, f"{course_name.replace(' ', '_')}_transactions")

    st.divider()
    st.subheader("🏆 Top 10 Courses by Revenue")
    top10 = cat_perf.sort_values("Revenue", ascending=False).head(10)
    fig = px.bar(top10, x="Revenue", y="CourseName", orientation="h", title="Top 10 Courses")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig), use_container_width=True)
elif page == "💰 Revenue & Forecast":
    st.title("💰 Revenue & Forecasting")

    avg_price = courses["CoursePrice"].mean()
    avg_rev_per_teacher = dff.groupby("TeacherID")["Amount"].sum().mean()
    highest_selling = dff.groupby("CourseName")["Amount"].sum().idxmax()
    avg_txn_value = dff["Amount"].mean()
    repeat_users = dff.groupby("UserID")["TransactionID"].count()
    repeat_rate = (repeat_users > 1).mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg Course Price", f"${avg_price:,.0f}")
    m2.metric("Avg Revenue / Teacher", f"${avg_rev_per_teacher:,.0f}")
    m3.metric("Highest Selling Course", highest_selling)
    m4.metric("Avg Transaction Value", f"${avg_txn_value:,.2f}")
    m5.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%")

    st.divider()
    # Forecasting using Linear Regression on monthly revenue
    monthly = transactions.groupby("Month")["Amount"].sum().reset_index().sort_values("Month")
    monthly["t"] = range(len(monthly))

    if len(monthly) >= 3:
        X = monthly[["t"]]
        y = monthly["Amount"]
        model = LinearRegression().fit(X, y)
        residual_std = np.std(y - model.predict(X))

        future_t = np.arange(len(monthly), len(monthly) + 6).reshape(-1, 1)
        future_pred = model.predict(future_t)
        future_months = pd.date_range(
            pd.Period(monthly["Month"].iloc[-1]).to_timestamp() + pd.offsets.MonthBegin(1),
            periods=6, freq="MS"
        ).strftime("%Y-%m")

        forecast_df = pd.DataFrame({
            "Month": future_months,
            "Forecast": future_pred,
            "Lower": future_pred - 1.96 * residual_std,
            "Upper": future_pred + 1.96 * residual_std,
        })

        st.subheader("📈 Next 6 Months Revenue Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Amount"], name="Actual Revenue",
                                  mode="lines+markers", line=dict(color=PURPLE)))
        fig.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["Forecast"], name="Forecast",
                                  mode="lines+markers", line=dict(color=PURPLE_LIGHT, dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(forecast_df["Month"]) + list(forecast_df["Month"][::-1]),
            y=list(forecast_df["Upper"]) + list(forecast_df["Lower"][::-1]),
            fill="toself", fillcolor="rgba(168,85,247,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval", showlegend=True
        ))
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.dataframe(forecast_df.style.format({"Forecast": "${:,.0f}", "Lower": "${:,.0f}", "Upper": "${:,.0f}"}),
                     use_container_width=True)
    else:
        st.info("Not enough monthly history to build a reliable forecast yet.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        rev_method = dff.groupby("PaymentMethod")["Amount"].sum().reset_index()
        fig = px.bar(rev_method, x="PaymentMethod", y="Amount", title="Revenue by Payment Method")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        rev_level = dff.groupby("CourseLevel")["Amount"].sum().reset_index()
        fig = px.pie(rev_level, names="CourseLevel", values="Amount", title="Revenue by Course Level")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    export_buttons(dff, "revenue_data")
elif page == "🤖 ML Insights":
    st.title("🤖 Machine Learning Insights")

    target = st.radio("Choose prediction target", ["Predict Teacher Rating", "Predict Course Revenue"], horizontal=True)

    if target == "Predict Teacher Rating":
        ml_df = teachers.copy()
        le_gender = LabelEncoder()
        le_expertise = LabelEncoder()
        ml_df["GenderEnc"] = le_gender.fit_transform(ml_df["Gender"])
        ml_df["ExpertiseEnc"] = le_expertise.fit_transform(ml_df["Expertise"])
        rev_by_teacher = transactions.groupby("TeacherID")["Amount"].sum().rename("Revenue")
        ml_df = ml_df.merge(rev_by_teacher, on="TeacherID", how="left").fillna(0)

        features = ["Age", "YearsOfExperience", "GenderEnc", "ExpertiseEnc", "Revenue"]
        X = ml_df[features]
        y = ml_df["TeacherRating"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        c1, c2 = st.columns(2)
        c1.metric("Model Accuracy (R²)", f"{acc:.2f}")
        c2.metric("Mean Absolute Error", f"{mae:.2f}")

        importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance — Teacher Rating")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("🔮 Try a Prediction")
        c1, c2, c3 = st.columns(3)
        age_in = c1.slider("Age", 22, 65, 35)
        exp_in = c2.slider("Years of Experience", 0, 25, 5)
        rev_in = c3.number_input("Revenue Generated", 0, 200000, 10000, step=1000)
        c4, c5 = st.columns(2)
        gender_in = c4.selectbox("Gender", le_gender.classes_)
        expertise_in = c5.selectbox("Expertise", le_expertise.classes_)
        if st.button("Predict Rating"):
            x_new = pd.DataFrame([[age_in, exp_in, le_gender.transform([gender_in])[0],
                                    le_expertise.transform([expertise_in])[0], rev_in]], columns=features)
            pred = model.predict(x_new)[0]
            st.success(f"Predicted Teacher Rating: **{pred:.2f} ⭐**")

    else:  # Predict Course Revenue
        ml_df = courses.merge(
            transactions.groupby("CourseID").agg(Revenue=("Amount", "sum")).reset_index(),
            on="CourseID", how="left"
        ).fillna(0)
        le_cat = LabelEncoder()
        le_type = LabelEncoder()
        le_level = LabelEncoder()
        ml_df["CatEnc"] = le_cat.fit_transform(ml_df["CourseCategory"])
        ml_df["TypeEnc"] = le_type.fit_transform(ml_df["CourseType"])
        ml_df["LevelEnc"] = le_level.fit_transform(ml_df["CourseLevel"])

        features = ["CoursePrice", "CourseDuration", "CourseRating", "CatEnc", "TypeEnc", "LevelEnc"]
        X = ml_df[features]
        y = ml_df["Revenue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        c1, c2 = st.columns(2)
        c1.metric("Model Accuracy (R²)", f"{acc:.2f}")
        c2.metric("Mean Absolute Error", f"${mae:,.0f}")

        importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance — Course Revenue")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("🔮 Try a Prediction")
        c1, c2, c3 = st.columns(3)
        price_in = c1.number_input("Course Price", 0, 1000, 200)
        dur_in = c2.number_input("Course Duration (hrs)", 1, 100, 20)
        rating_in = c3.slider("Course Rating", 1.0, 5.0, 4.0)
        c4, c5, c6 = st.columns(3)
        cat_in = c4.selectbox("Category", le_cat.classes_)
        type_in = c5.selectbox("Type", le_type.classes_)
        level_in = c6.selectbox("Level", le_level.classes_)
        if st.button("Predict Revenue"):
            x_new = pd.DataFrame([[price_in, dur_in, rating_in, le_cat.transform([cat_in])[0],
                                    le_type.transform([type_in])[0], le_level.transform([level_in])[0]]], columns=features)
            pred = model.predict(x_new)[0]
            st.success(f"Predicted Revenue: **${pred:,.0f}**")
elif page == "⚙ Settings":
    st.title("⚙ Settings")
    st.write("Theme, account, and display preferences.")
    st.toggle("Dark Mode", value=(st.session_state.theme == "Dark"), key="settings_theme_toggle")
    if st.session_state.settings_theme_toggle != (st.session_state.theme == "Dark"):
        st.session_state.theme = "Dark" if st.session_state.settings_theme_toggle else "Light"
        st.rerun()
elif page == "ℹ About":
    st.title("ℹ About This Project")
    st.markdown("""
    **Dataset:** Online education platform data — Courses, Teachers, and Transactions.

    **Objectives:**
    - Provide an executive-level, automatically generated business summary
    - Track instructor and course performance with leaderboards and drill-downs
    - Forecast future revenue using regression modeling
    - Predict teacher ratings and course revenue using machine learning

    **Technologies Used:**
    - Python
    - Pandas / NumPy
    - Plotly
    - Streamlit
    - Scikit-learn
    """)