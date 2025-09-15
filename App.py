import pandas as pd
import numpy as np
import datetime as datetime
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pyodbc
from sqlalchemy import create_engine
import streamlit as st
import base64
from itertools import combinations
from collections import Counter
import plotly.express as px
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter



# --------------------------
# Background image paths
# --------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MAIN_BG = os.path.join(ASSETS_DIR, "main_bg.jpg")
LOGIN_BG = os.path.join(ASSETS_DIR, "login_bg.webp")

# --------------------------
# Helper: set background image for the app
# --------------------------
def set_bg(image_path, text_color="#000000"):
    """Set a full-page background image for Streamlit app.
    - image_path: absolute path to the image on disk
    - text_color: default text color to apply on top of the background (default black)"""
    if not os.path.exists(image_path):
        st.error(f"Background image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: {text_color};
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(255,255,255,0.12); /* tweak alpha to lighten/darken overlay */
            pointer-events: none;
            z-index: 0;
        }}

        /* Keep Streamlit content above the overlay */
        .css-1lcbmhc.e1fqkh3o2, .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* Make sure headings and text respect the chosen color */
        .stApp, .stApp * {{
            color: {text_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# kpi theme

def custom_kpi(label, value, delta=None, value_color="#000000", delta_color="#007A33"):
    st.markdown(f"""
        <div style="padding: 15px 20px; border-radius: 12px; background-color: rgba(255,255,255,0.85);
                    color: {value_color}; border: 1px solid rgba(0,0,0,0.08); margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: 500; color: #333333;">{label}</div>
            <div style="font-size: 24px; font-weight: bold; color: {value_color};">{value}</div>
            {f'<div style="font-size: 12px; color: {delta_color};">{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)

# --------------------------
# USER LOGIN CREDENTIALS
# --------------------------
Users = {
    "Bhawya": 'ATM',
}

# --------------------------
# LOAD DATA FROM SQL
# --------------------------
@st.cache_data
def load_data():
    # load csv from same folder as app
    base_path = os.path.dirname(__file__)
    fp = os.path.join(base_path, "ATM_Cash_management.csv")
    df_ATM = pd.read_csv(fp)


    return df_ATM
# --------------------------
# LOGIN PAGE
# --------------------------
def login_page():
    
    set_bg(LOGIN_BG, text_color="#000000")

    st.markdown("""
        <style>
        .stButton > button {
            background-color: rgba(255,255,255,0.9);
            color: #111;
            border-radius: 8px;
            padding: 0.5em 2em;
            transition: 0.3s;
            border: none;
        }

        .stButton > button:hover {
            background-color: rgba(255,255,255,1);
            color: #000;
        }

        /* Slight translucent input backgrounds for readability */
        .stTextInput>div>div>input, .stTextInput>div>div>textarea {
            background: rgba(255,255,255,0.88) !important;
            color: #111 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;   /* try 0rem if you want it almost flush to the edge */
            padding-right: 1rem !important;
            margin-left: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(
            "<h4 style='text-align:center; color:rgba(0,0,0,0.85);'>Welcome to the ATM & Cash Management Optimization Dashboard</h4>",
            unsafe_allow_html=True
        )
        st.title("🔐 Login")

        #  Assigning unique keys to all widgets
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        # Track if login was attempted
        if "login_attempted" not in st.session_state:
            st.session_state.login_attempted = False

        if st.button("Login", key="login_button"):
            st.session_state.login_attempted = True
            if username in Users and Users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "intro"
                st.success("Login Successful")
                st.rerun()

        # Show error only after login attempt
        if st.session_state.login_attempted and not st.session_state.logged_in:
            st.error("Invalid Username or Password", icon="🚫")


# --------------------------
# BUSINESS CONTEXT PAGE
# --------------------------
def business_context():
    set_bg(MAIN_BG, text_color="#000000")

    
    st.markdown("""
        <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 90% !important;
        }
        div.stButton > button {
            background-color: rgba(255,255,255,0.9);
            color: #111;
            font-size: 16px;
            padding: 10px 30px;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
            display: inline-flex;
            white-space: nowrap;
        }
        div.stButton > button:hover {
            background-color: rgba(0,122,204,0.95);
        }

        /* Business context card CSS - image friendly */
        .bc-section {
          background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.85));
          padding: 28px;
          border-radius: 10px;
          margin-top: 12px;
          margin-bottom: 18px;
        }
        .bc-title {
          text-align: center;
          font-size: 34px;
          font-weight: 700;
          color: #111111;
          margin-bottom: 6px;
        }
        .bc-paragraph {
          text-align: center;
          font-size: 16px;
          color: #333333;
          max-width: 980px;
          margin: 0 auto 18px auto;
          line-height: 1.5;
        }

        .cards-row {
          display:flex;
          gap: 18px;
          justify-content:center;
          align-items:flex-start;
          flex-wrap:wrap;
          margin-top: 12px;
        }

        .bc-card {
          background: rgba(255,255,255,0.95);
          border-radius: 10px;
          box-shadow: 0 6px 18px rgba(0,0,0,0.08);
          padding: 16px 18px;
          width: 320px;
          text-align: left;
          display:flex;
          gap: 12px;
          align-items:flex-start;
          border: 1px solid rgba(0,0,0,0.05);
        }

        .bc-card .icon {
          width:44px;
          height:44px;
          border-radius:8px;
          display:flex;
          align-items:center;
          justify-content:center;
          font-size:18px;
          background: rgba(0,0,0,0.03);
          border: 1px solid rgba(0,0,0,0.04);
        }

        .bc-card h4 {
          margin:0 0 6px 0;
          font-size:16px;
          color:#111111;
        }
        .bc-card p {
          margin:0;
          color:#333333;
          font-size:14px;
          line-height:1.4;
        }

        .center-badge {
          width:44px;
          height:44px;
          border-radius:10px;
          background: rgba(255,255,255,0.92);
          box-shadow: 0 6px 14px rgba(0,0,0,0.06);
          display:inline-flex;
          align-items:center;
          justify-content:center;
          margin: 0 6px;
          border: 1px solid rgba(0,0,0,0.04);
          color: #111111;
          font-size:18px;
        }

        @media (max-width:760px) {
          .bc-card { width: 100%; max-width: 420px; }
          .center-badge { display:none; } /* hide badges on very small screens */
        }
        </style>
    """, unsafe_allow_html=True)

    # Title & small image
    st.markdown("<div class='bc-title'>💼 BUSINESS CONTEXT</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='bc-paragraph'>
           Efficient ATM cash management is crucial for seamless customer experiences and operational efficiency.
           The challenge lies in balancing adequate cash stock with minimizing idle funds.
        </div>
    """, unsafe_allow_html=True)

    # Cards row
    st.markdown("""
      <div class="cards-row">
        <div class="bc-card">
          <div class="icon">💸</div>
          <div>
            <h4>The Dilemma:</h4>
            <p>Too much cash ties up capital and increases risk.</p>
          </div>
        </div>

      

        <div class="bc-card">
          <div class="icon">👥</div>
          <div>
            <h4>Customer Impact:</h4>
            <p>Understocking leads to cashouts and dissatisfaction.</p>
          </div>
        </div>

        <div class="bc-card">
          <div class="icon">🎯</div>
          <div>
            <h4>Our Goal:</h4>
            <p>Minimize both out-of-cash incidents and excess idle cash.</p>
          </div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # close section wrapper
    st.markdown("</div>", unsafe_allow_html=True)

    # Continue button centered
    btn_col1, btn_col2, btn_col3 = st.columns([4, 2, 4])
    with btn_col2:
        if st.button("👉 Continue to Dashboard", key="continue_to_dashboard_btn"):
            st.session_state.page = "dashboard"
            st.rerun()


# --------------------------
# DASHBOARD PAGE
# --------------------------

def dashboard():
    set_bg(MAIN_BG, text_color="#000000")

    df_ATM = load_data()

    
# ensure Date col is datetime
    df_ATM['Date'] = pd.to_datetime(df_ATM['Date'])
    df_ATM['Year'] = df_ATM['Date'].dt.year
    df_ATM['Quarter'] = df_ATM['Date'].dt.quarter
    df_ATM['Month'] = df_ATM['Date'].dt.month
    
# Sidebar navigation + global filters
    st.sidebar.title("📂 Navigation")
    selected_page = st.sidebar.radio("Go to", ['KPIs','ATM USAGE', 'CASH FLOW AND DEMAND','WITHDRAWAL SEASONALITY'])

    st.sidebar.markdown("---")
    st.sidebar.title("🔍 Filters")

# Year & Quarter multiselects (defaults to all available)
    years = sorted(df_ATM["Year"].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect("Year", options=years, default=years)

    quarters = [1, 2, 3, 4]
    selected_quarters = st.sidebar.multiselect("Quarter", options=quarters, default=quarters)

# ---------- Page-specific slicers (use descriptive variable names) ----------
    if selected_page == 'ATM USAGE':
        days = df_ATM["Day_of_Week"].dropna().unique().tolist()
        selected_days = st.sidebar.multiselect("Day", options=days, default=days)

        holidays = df_ATM["Holiday_Flag"].dropna().unique().tolist()
        selected_holidays = st.sidebar.multiselect("Holiday Flag", options=holidays, default=holidays)

    elif selected_page == 'CASH FLOW AND DEMAND':
        months = sorted(df_ATM["Month"].dropna().unique().tolist())
        selected_months = st.sidebar.multiselect("Month", options=months, default=months)

    elif selected_page == 'WITHDRAWAL SEASONALITY':
        days = df_ATM["Day_of_Week"].dropna().unique().tolist()
        selected_days = st.sidebar.multiselect("Day", options=days, default=days)

        holidays = df_ATM["Holiday_Flag"].dropna().unique().tolist()
        selected_holidays = st.sidebar.multiselect("Holiday Flag", options=holidays, default=holidays)

# ---------- Safe defaults if user deselects everything ----------

    if not selected_years:
        selected_years = years
    if not selected_quarters:
        selected_quarters = quarters

# For page-specific selectors ensure variables exist in any branch

    if 'selected_days' not in locals():
        selected_days = df_ATM["Day_of_Week"].dropna().unique().tolist()
    if 'selected_holidays' not in locals():
        selected_holidays = df_ATM["Holiday_Flag"].dropna().unique().tolist()
    if 'selected_months' not in locals():
        selected_months = sorted(df_ATM["Month"].dropna().unique().tolist())

# ---------- Build base filter ----------
    base_atm_filter = (
        df_ATM["Year"].isin(selected_years) &
        df_ATM["Quarter"].isin(selected_quarters)
)

# ---------- Apply page-specific filtering using the user's selections ----------
    if selected_page == 'ATM USAGE':
        df_ATM_Filtered = df_ATM.loc[
            base_atm_filter &
            df_ATM["Day_of_Week"].isin(selected_days) &
            df_ATM["Holiday_Flag"].isin(selected_holidays)].copy()

    elif selected_page == 'CASH FLOW AND DEMAND':
        df_ATM_Filtered = df_ATM.loc[
            base_atm_filter &
            df_ATM["Month"].isin(selected_months)].copy()

    elif selected_page == 'WITHDRAWAL SEASONALITY':
        df_ATM_Filtered = df_ATM.loc[
            base_atm_filter &
            df_ATM["Day_of_Week"].isin(selected_days) &
            df_ATM["Holiday_Flag"].isin(selected_holidays)].copy()

    elif selected_page == "KPIs":
        df_ATM_Filtered = df_ATM.loc[base_atm_filter].copy()


#LOGOUT BUTTON

    st.sidebar.markdown('<div class="logout-container">', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 Logged in as: `{st.session_state.username}`")

    if st.sidebar.button("🚪 Logout" , key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "login"
        st.session_state.login_attempted = False 
        st.rerun()

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# KPIs Section
# -------------------------------

    
    if selected_page == "KPIs":
        st.markdown(
    """
    <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
        Key Performance Indicators
    </h1>
    """,
    unsafe_allow_html=True
)
   

# -------------------------------
# KPI CALCULATIONS SECTION
# -------------------------------

        total_withdrawal = (df_ATM_Filtered['Total_Withdrawals'].sum())
        total_deposit = (df_ATM_Filtered['Total_Deposits'].sum())
        avg_daily_withdrawals = (df_ATM_Filtered.groupby('Date')['Total_Withdrawals'].sum().mean())
        avg_daily_deposits = (df_ATM_Filtered.groupby('Date')['Total_Deposits'].sum().mean())
        net_outflow = ((df_ATM_Filtered['Total_Withdrawals'].sum() - df_ATM_Filtered['Total_Deposits'].sum()))
        peak_withdrawal_day = df_ATM_Filtered.groupby('Day_of_Week')['Total_Withdrawals'].sum().sort_values(ascending = False).reset_index().head(1).iloc[0,0]
        peak_withdrawal_time = df_ATM_Filtered.groupby('Time_of_Day')['Total_Withdrawals'].sum().sort_values(ascending = False).reset_index().head(1).iloc[0,0]
        Total_ATM = df_ATM_Filtered['ATM_ID'].nunique()
        top_ATM = df_ATM_Filtered.groupby('ATM_ID')['Total_Withdrawals'].sum().sort_values(ascending = False).reset_index().head(1).iloc[0,0]
        avg_cash_demand = df_ATM_Filtered['Cash_Demand_Next_Day'].mean()
        ratio = (total_withdrawal/total_deposit)
        special_withdrawal = df_ATM_Filtered[df_ATM_Filtered['Special_Event_Flag'] == 1]['Total_Withdrawals'].mean()
        normal_withdrawal = df_ATM_Filtered[df_ATM_Filtered['Special_Event_Flag'] == 0]['Total_Withdrawals'].mean()
        special_event_impact = ((special_withdrawal - normal_withdrawal) * 100.00/normal_withdrawal).round(2)
        holiday_withdrawal = df_ATM_Filtered[df_ATM_Filtered['Holiday_Flag'] == 1]['Total_Withdrawals'].mean()
        no_holiday_withdrawal = df_ATM_Filtered[df_ATM_Filtered['Holiday_Flag'] == 0]['Total_Withdrawals'].mean()
        holiday_impact = ((holiday_withdrawal - no_holiday_withdrawal) * 100.00 / no_holiday_withdrawal).round(2)
        cash_utilization = ((df_ATM_Filtered['Total_Withdrawals'].sum() / df_ATM_Filtered['Previous_Day_Cash_Level'].sum()) * 100).round(2)
        peak_cash_demand = df_ATM_Filtered['Cash_Demand_Next_Day'].max()

# -------------------------------
# KPI DASHBOARD LAYOUT
# -------------------------------

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            custom_kpi("Total Withdrawals", f"{total_withdrawal/1000000:,.2f}M")
        with row1_col2:
            custom_kpi("Total Deposits", f"{total_deposit/1000000:,.2f}M")
        with row1_col3:
            custom_kpi("Average Withdrawal", f"{avg_daily_withdrawals/1000000:,.2f}M")

        row1_col4, row1_col5, row1_col6 = st.columns(3)
        with row1_col4:
            custom_kpi("Average Deposits", f"{avg_daily_deposits/1000000:,.2f}M")
        with row1_col5:
            custom_kpi("Net Flow", f"{net_outflow/1000000:,.2f}M")
        with row1_col6:
            custom_kpi("Peak Withdrawal Day", f"{peak_withdrawal_day}")

    
        row1_col7, row1_col8, row1_col9 = st.columns(3)
        with row1_col7:
            custom_kpi("Peak Withdrawal Time", f"{peak_withdrawal_time}")
        with row1_col8:
            custom_kpi("Total ATM", f"{Total_ATM}")
        with row1_col9:
            custom_kpi("Top Withdrawal ATM", f"{top_ATM}")

        row1_col10, row1_col11, row1_col12 = st.columns(3)
        with row1_col10:
            custom_kpi("Average Cash Demand", f"{avg_cash_demand/1000000:,.2f}M")
        with row1_col11:
            custom_kpi("Withdrawal-Deposit Ratio", f"{ratio:.2f} : 1")
        with row1_col12:
            custom_kpi("Special Event Impact", f"{special_event_impact}")

    
        row1_col11, row1_col12, row1_col13 = st.columns(3)
        with row1_col11:
            custom_kpi("Holiday Impact", f"{holiday_impact}")
        with row1_col12:
            custom_kpi("Cash Utilization", f"{cash_utilization}%")
        with row1_col13:
            custom_kpi("Peak Cash Demand", f"{peak_cash_demand/1000000:,.2f}M")

# -------------------------------
# ATM Usage Section
# -------------------------------

    
    elif selected_page == "ATM USAGE":
        st.markdown(
    """
    <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
        ATM USAGE ANALYSIS
    </h1>
    """,
    unsafe_allow_html=True
)
   
        # 1. ATM withdrawal volume

        ATM_withdrawal_volume = df_ATM_Filtered.groupby('ATM_ID')['Total_Withdrawals'].sum().sort_values(ascending = False)

        st.subheader("ATM Withdrawal Volume")
        fig, ax = plt.subplots(figsize = (10,14))
        bars = ATM_withdrawal_volume.plot(kind = 'barh', color = 'lightgreen', edgecolor = 'black', figsize=(14,12))

        plt.title("Withdrawal Volume by all ATM ", fontsize=16, pad=20)
        plt.xlabel("Withdrawal Volume (in Millions)", fontsize = 12)
        plt.ylabel("ATM ID", fontsize = 12)

    #data labels

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M'))

        for i, v in enumerate(ATM_withdrawal_volume):
            if i % 2 == 0: 
                ax.text(v + (0.01 * max(ATM_withdrawal_volume)), i,
                    f"{v/1e6:.2f}M",
                    va='center', ha='left', fontsize=9, color='black')

        plt.subplots_adjust(bottom=0.12, top=0.92)
        st.pyplot(fig)

        # 2. ATM Deposit volume

        ATM_deposit_volume = df_ATM_Filtered.groupby('ATM_ID')['Total_Deposits'].sum().sort_values(ascending = False)

        st.subheader("ATM Deposit Volume")
        fig, ax = plt.subplots(figsize = (10,14))
        bars = ATM_deposit_volume.plot(kind = 'barh', color = 'lightpink', edgecolor = 'black', figsize=(14,12))

        plt.title("Deposit Volume by all ATM ", fontsize=16, pad=20)
        plt.xlabel("Deposit Volume (in Millions)", fontsize = 12)
        plt.ylabel("ATM ID", fontsize = 12)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M'))

        for i, v in enumerate(ATM_deposit_volume):
            if i % 2 == 0: 
                ax.text(v + (0.01 * max(ATM_deposit_volume)), i,
                    f"{v/1e6:.2f}M",
                    va='center', ha='left', fontsize=9, color='black')



        plt.subplots_adjust(bottom=0.12, top=0.92)
        st.pyplot(fig)



        # Pareto Analysis - chart labels

        ATM_withdrawal_volume = df_ATM_Filtered.groupby('ATM_ID')['Total_Withdrawals'].sum().sort_values(ascending = False)

        atm_contribution = ATM_withdrawal_volume/ATM_withdrawal_volume.sum() * 100 
        cumulative_contribution = atm_contribution.cumsum() 

        x = np.arange(len(atm_contribution))
        labels = atm_contribution.index

        st.subheader("Pareto Analysis")
        fig, ax1 = plt.subplots(figsize=(12,6))
        bars = ax1.bar(x, atm_contribution.values, color='skyblue', edgecolor='black')
        ax1.set_ylabel('% Contribution (Withdrawals)', fontsize=11)
        ax1.set_xlabel('ATM ID', fontsize=11)
        ax1.set_title('Pareto Analysis', fontsize=16, pad = 20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=90,fontsize=12)

        ax2 = ax1.twinx()
        ax2.plot(x, cumulative_contribution.values, marker='o', color='red')
        ax2.set_ylabel('Cumulative % of Withdrawals', fontsize=12)
        ax2.axhline(80, linestyle='--', color='green', linewidth=1)  # 80% reference line

    # get all heights
        heights = [bar.get_height() for bar in bars]

    # find index of tallest and shortest
        max_idx = int(np.argmax(heights))
        min_idx = int(np.argmin(heights))

    # vertical offset
        max_h = max(heights) if heights else 1
        vert_offset = max(0.05, 0.02 * max_h)

    # add labels only for those two bars
        for i, bar in enumerate(bars):
            if i == max_idx or i == min_idx:
                h = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2,
                    h + vert_offset,
                    f"{h:.1f}%",
                    ha='center', va='bottom',
                    fontsize=9, color='black',
                    clip_on=False
            )

        for pct in [50, 80]:
            idx = np.abs(cumulative_contribution.values - pct).argmin()
            val = cumulative_contribution.values[idx]
            ax2.text(x[idx], val + 2, f"{val:.1f}%", ha='center', va='bottom',
                color='black', fontsize=9, fontweight='bold', clip_on=False)

    # handle 100% separately
        idx100 = np.abs(cumulative_contribution.values - 100).argmin()
        val100 = cumulative_contribution.values[idx100]
        ax2.text(x[idx100], val100 - 3, f"{val100:.1f}%", ha='center', va='top',
                color='black', fontsize=9, fontweight='bold', clip_on=False)

  
        ax2.set_ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig)

        # 4. Idle frequency 


        daily = df_ATM_Filtered.groupby(['ATM_ID', 'Date'])[['Total_Withdrawals', 'Total_Deposits']].sum().reset_index()
        daily['Total_Activity'] = daily['Total_Withdrawals'] + daily['Total_Deposits']

        thresholds = daily.groupby('Date')['Total_Activity'].quantile(0.10).reset_index()
        thresholds.rename(columns={'Total_Activity' : 'p10_threshold'}, inplace= True)

        daily = daily.merge(thresholds, on='Date', how='left')
        daily['Idle_Day'] = (daily['Total_Activity'] <= daily['p10_threshold']).astype(int)

        idle_stats = daily.groupby('ATM_ID').agg(idle_days = ('Idle_Day','sum'), total_days = ('Date','nunique')).assign(idle_pct=lambda d: (d['idle_days']/d['total_days']*100).round(2)).sort_values('idle_pct', ascending=False).reset_index()

        top_n = 15
        view = idle_stats.head(top_n).iloc[::-1]

        st.subheader("Idle Frequency")
        fig, ax = plt.subplots(figsize=(10,7))
        bars = plt.barh(view['ATM_ID'], view['idle_pct'], color='lightcoral', edgecolor='black')

        plt.title(f"Most Frequently Idle ATMs", fontsize=16, pad = 20)
        plt.xlabel("Idle Days (% of Total Days)", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)

    # Add labels
        for bar, pct in zip(bars, view['idle_pct']):
            plt.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{pct:.1f}%", va='center', fontsize=9)

        plt.xlim(0, view['idle_pct'].max() + 5)

        plt.tight_layout()
        st.pyplot(fig)

    # 5. Withdrawal Variability per ATM

        daily_withdrawals = df_ATM_Filtered.groupby(['ATM_ID', 'Date'])['Total_Withdrawals'].sum().reset_index()

    # variability metrics

        withdrawal_variability = (
            daily_withdrawals.groupby('ATM_ID')['Total_Withdrawals']
            .agg(['mean','std']).reset_index().rename(columns={'mean': 'avg_withdrawals','std':'std_withdrawals'}
                                                                                                                            )
    )

    #Coefficient of Variation

        withdrawal_variability['cv_withdrawals'] = (
            withdrawal_variability['std_withdrawals'] / withdrawal_variability['avg_withdrawals']).round(2)

        withdrawal_variability = withdrawal_variability.sort_values('cv_withdrawals', ascending=False)

        top_n = 15
        view = withdrawal_variability.head(top_n).iloc[::-1]  # reverse for plotting

        st.subheader("Withdrawal Varibility")
        fig, ax = plt.subplots(figsize=(10,7))
        bars = plt.barh(view['ATM_ID'], view['cv_withdrawals'], color='steelblue', edgecolor='black')

        plt.title(f"Top {top_n} ATMs by Withdrawal Variability (CV)", fontsize=16, pad = 20)
        plt.xlabel("Coefficient of Variation (CV)", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)

    # Add labels
        for bar, cv in zip(bars, view['cv_withdrawals']):
            plt.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{cv:.2f}", va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        #6.  ATM Utilization Rate

        ATM_Utilization = df_ATM_Filtered.groupby('ATM_ID').agg(Total_Withdrawals=('Total_Withdrawals','sum'),Previous_Day_Cash_Level=('Previous_Day_Cash_Level','mean')).reset_index()

        ATM_Utilization['utilization_rate'] = (
            ATM_Utilization['Total_Withdrawals'] / ATM_Utilization['Previous_Day_Cash_Level'])

        ATM_Utilization = ATM_Utilization.sort_values(by = 'utilization_rate', ascending = False)

        top_n = 10
        view = ATM_Utilization.head(top_n).iloc[::-1]

        st.subheader("ATM Utilization")
        fig,ax = plt.subplots(figsize=(10,8))
        bars = plt.barh(view['ATM_ID'], view['utilization_rate'], color='mediumseagreen', edgecolor='black')

        plt.title(f"Top {top_n} ATMs by Utilization Rate", fontsize=16, pad = 20)
        plt.xlabel("Utilization Rate (%)", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)

    # Add data labels   
        for bar, pct in zip(bars, view['utilization_rate']):
            plt.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"{pct:.1f}%", va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)


    #7. --- Bottom 15 ATMs ---
        view_bottom = ATM_Utilization.tail(top_n)

        fig,ax = plt.subplots(figsize=(10,7))
        bars = plt.bar(view_bottom['ATM_ID'], view_bottom['utilization_rate'], color='tomato', edgecolor='black')
        plt.title(f"Bottom {top_n} ATMs by Utilization Rate", fontsize=14)
        plt.xlabel("Utilization Rate (%)", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)

    # Data labels
        for bar, pct in zip(bars, view['utilization_rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{pct:.2f}%", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

    #  ATM Segmentation 

        atm_summary = (
            df_ATM_Filtered.groupby('ATM_ID')
            .agg(
                Total_Withdrawals=('Total_Withdrawals','sum'),
                Total_Deposits=('Total_Deposits','sum'),
                Previous_Day_Cash_Level=('Previous_Day_Cash_Level','mean')  
            )
            .reset_index()
    )

    #withdrawal_percentile

        q80 = atm_summary['Total_Withdrawals'].quantile(0.80)
        q20 = atm_summary['Total_Withdrawals'].quantile(0.20)


        def demand_segment(x):
            if x >= q80:
                return "High Demand"
            elif x <= q20:
                return "Low Demand"
            else:
                return "Medium Demand"
        
        atm_summary['Demand_Segment'] = atm_summary['Total_Withdrawals'].apply(demand_segment)
        

    # Deposit vs Withdrawal ratio
        atm_summary['Deposit_Ratio'] = atm_summary['Total_Deposits'] / (atm_summary['Total_Withdrawals'] + 1)

        atm_summary['Flow_Type'] = atm_summary['Deposit_Ratio'].apply(
            lambda r: "Deposit-Heavy" if r > 0.5 else "Withdrawal-Heavy"
    )

    # 8. Bar Chart: ATM count per Demand Segment ---
        st.subheader("ATM Segmentation")
        fig,ax = plt.subplots(figsize=(7,5))
        sns.countplot(data=atm_summary, x='Demand_Segment', order=['High Demand','Medium Demand','Low Demand'],
                palette=['#2ca02c','#1f77b4','#ff7f0e'], edgecolor='black')

        plt.title("ATM Segmentation by Demand", fontsize=16, pad = 20)
        plt.xlabel("Segment", fontsize=12)
        plt.ylabel("Number of ATMs", fontsize=12)

    # Add labels on top of bars
        for p in plt.gca().patches:
                plt.text(p.get_x() + p.get_width()/2, p.get_height()+0.5,
                f"{int(p.get_height())}", ha='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)


    # 9 Scatter Plot: Withdrawals vs Deposits by Demand Segment ---

        st.subheader("Withdrawal vs Deposit by Demand Segmentation")
        fig, ax = plt.subplots(figsize=(8,6))
        colors = {'High Demand':'#2ca02c','Medium Demand':'#1f77b4','Low Demand':'#ff7f0e'}

        sns.scatterplot(data=atm_summary, x='Total_Withdrawals', y='Total_Deposits',
                    hue='Demand_Segment', palette=colors, s=80, alpha=0.8, edgecolor='black')

        plt.title("ATM Segmentation (Withdrawals vs Deposits)", fontsize=16, pad = 20)
        plt.xlabel("Total Withdrawals", fontsize=12)
        plt.ylabel("Total Deposits", fontsize=12)
        plt.legend(title="Segment")
        ax.ticklabel_format(style='plain', axis='x')
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)

    # 10 idle ATM 

        daily = df_ATM_Filtered.groupby(['ATM_ID','Date'])[['Total_Withdrawals','Total_Deposits']].sum().reset_index()
        daily['Total_Activity'] = daily['Total_Withdrawals'] + daily['Total_Deposits']

    # Define idle (bottom 10% per day)
        thresholds = daily.groupby('Date')['Total_Activity'].quantile(0.10).reset_index().rename(columns={'Total_Activity':'p10'})
        daily = daily.merge(thresholds, on='Date')
        daily['Idle_Day'] = (daily['Total_Activity'] <= daily['p10']).astype(int)

        idle_stats = daily.groupby('ATM_ID')['Idle_Day'].mean().reset_index().rename(columns={'Idle_Day':'Idle_Rate'})

    # Merge with segmentation
        atm_seg_idle = atm_summary.merge(idle_stats, on='ATM_ID')

        seg_idle = atm_seg_idle.groupby('Demand_Segment')['Idle_Rate'].mean().reset_index()

    # Add Active rate
        seg_idle['Active_Rate'] = 1 - seg_idle['Idle_Rate']

    # Step 2: Plot pie charts for each segment
        st.subheader("Idle ATMs by Segment")
        fig, axes = plt.subplots(1, 3, figsize=(14,5))

        colors = [['lightcoral','mediumseagreen'], ['lightcoral','mediumseagreen'], ['lightcoral','mediumseagreen']]
        segments = ['High Demand','Medium Demand','Low Demand']

        for i, seg in enumerate(segments):
            values = [seg_idle.loc[seg_idle['Demand_Segment']==seg,'Idle_Rate'].values[0],
                seg_idle.loc[seg_idle['Demand_Segment']==seg,'Active_Rate'].values[0]]
        
            axes[i].pie(values, labels=['Idle','Active'], autopct='%1.1f%%',
                    startangle=90, colors=colors[i], wedgeprops={'edgecolor':'black'})
            axes[i].set_title(f"{seg} ATMs", fontsize=12)

        plt.suptitle("Idle vs Active Days by Demand Segment", fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)

    # 11. Net FLow 

        atm_flow = df_ATM_Filtered.groupby('ATM_ID')[['Total_Withdrawals', 'Total_Deposits']].sum().reset_index()

        atm_flow['Net_flow'] = atm_flow['Total_Withdrawals'] - atm_flow['Total_Deposits']

        atm_flow_sorted = atm_flow.sort_values('Net_flow', ascending=False)

        st.subheader("ATM Net Flow")
        fig, ax = plt.subplots(figsize=(11,8))
        bars = plt.barh(atm_flow_sorted['ATM_ID'], atm_flow_sorted['Net_flow'],
                    color=np.where(atm_flow_sorted['Net_flow']>=0, 'crimson','seagreen'),
                    edgecolor='black')

        plt.title("Net Cash Flow by ATM (Withdrawals - Deposits)", fontsize=15, pad = 20)
        plt.xlabel("Net Cash Flow", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)

    # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (0.005 * max(abs(atm_flow_sorted['Net_flow']))),
                bar.get_y() + bar.get_height()/2,
                f"{width/1e6:.2f}M", va='center', fontsize=9)

        ax.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------------
#  CASH FLOW AND DEMAND
# -------------------------------

    
    elif selected_page == "CASH FLOW AND DEMAND":
        st.markdown(
    """
    <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
        CASH FLOW AND DEMAND ANALYSIS
    </h1>
    """,
    unsafe_allow_html=True
)
        # cash flow and demand analysis

        # 12. Hoilday Impact on cash flow 

        holiday_stats =  (df_ATM_Filtered.groupby('Holiday_Flag')[['Total_Withdrawals','Total_Deposits']].mean().reset_index())

        holiday_stats['Net_Cash_Flow'] = holiday_stats['Total_Withdrawals'] - holiday_stats['Total_Deposits']

# Map flag to labels
        holiday_stats['Holiday_Type'] = holiday_stats['Holiday_Flag'].map({False:'Non-Holiday', True:'Holiday'})

        st.subheader("Hoilday Impact")
        fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# --- Withdrawals vs Deposits ---
        bars1 = ax[0].bar(holiday_stats['Holiday_Type'], holiday_stats['Total_Withdrawals'],color='skyblue', label='Withdrawals')
        bars2 = ax[0].bar(holiday_stats['Holiday_Type'], holiday_stats['Total_Deposits'], color='lightgreen', label='Deposits')

        ax[0].set_title("Withdrawals vs Deposits")
        ax[0].set_ylabel("Average Cash Flow per Day")
        ax[0].legend()

# Add data labels
        for bars in [bars1, bars2]:
            for bar in bars:
              height = bar.get_height()
              ax[0].text(bar.get_x() + bar.get_width()/2, height + 500, 
                   f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

# --- Net Cash Flow ---
        bars3 = ax[1].bar(holiday_stats['Holiday_Type'], holiday_stats['Net_Cash_Flow'], 
                  color='coral', edgecolor='black')

        ax[1].set_title("Net Cash Flow (Withdrawals - Deposits)")
        ax[1].set_ylabel("Net Cash Flow")
        

# Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2, height + 500, 
               f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

# Final layout
        plt.suptitle("Holiday Impact on Cash Flow", fontsize=16)
        plt.tight_layout()
        st.pyplot(fig) 

        # 13. Event flag -- 

        event_stats = (
            df_ATM_Filtered.groupby('Special_Event_Flag')[['Total_Withdrawals','Total_Deposits']].mean().reset_index())

        event_stats['Net_Cash_Flow'] = event_stats['Total_Withdrawals'] - event_stats['Total_Deposits']
        event_stats['Event_Type'] = event_stats['Special_Event_Flag'].map({False:'Normal Day', True:'Event Day'})

        st.subheader("Event Impact")
        fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# --- Withdrawals vs Deposits ---
        bars1 = ax[0].bar(event_stats['Event_Type'], event_stats['Total_Withdrawals'], color='skyblue', label='Withdrawals')
        bars2 = ax[0].bar(event_stats['Event_Type'], event_stats['Total_Deposits'], color='lightgreen', label='Deposits')

        ax[0].set_title("Withdrawals vs Deposits", fontsize = 16)
        ax[0].set_ylabel("Average Cash per Day")
        ax[0].legend()

# Add data labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax[0].text(bar.get_x() + bar.get_width()/2, height + 500, 
                   f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

# --- Net Cash Flow ---
        bars3 = ax[1].bar(event_stats['Event_Type'], event_stats['Net_Cash_Flow'], 
                  color='coral', edgecolor='black')

        ax[1].set_title("Net Cash Flow (Withdrawals - Deposits)")

# Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2, height + 500, 
               f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

# Final layout
        plt.suptitle("Special Event Impact on Cash Flow", fontsize=16)
        ax[1].set_ylabel("Net Cash Flow")
        plt.tight_layout()
        st.pyplot(fig)


        # 14. Weather impact 

        weather_stats = (
            df_ATM_Filtered.groupby('Weather_Condition')[['Total_Withdrawals','Total_Deposits']].mean().reset_index())

        weather_stats['Net_Cash_Flow'] = weather_stats['Total_Withdrawals'] - weather_stats['Total_Deposits']

        st.subheader("Withdrawals vs Deposits by Weather")
        fig, ax = plt.subplots(figsize=(12,6))
        bar_width = 0.35
        x = range(len(weather_stats))

        bars1 = plt.bar([i - bar_width/2 for i in x], weather_stats['Total_Withdrawals'], 
                width=bar_width, color='skyblue', label='Withdrawals')
        bars2 = plt.bar([i + bar_width/2 for i in x], weather_stats['Total_Deposits'], 
                width=bar_width, color='lightgreen', label='Deposits')

        plt.xticks(x, weather_stats['Weather_Condition'], rotation=45)
        plt.title("Withdrawals vs Deposits by Weather Condition", fontsize=16)
        plt.xlabel("Weather Condition", fontsize=12)
        plt.ylabel("Average Amount per Day", fontsize=12)
        plt.legend()

# --- Add Data Labels ---
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 500, 
             f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 500, 
             f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # 15. Cash Demand Distribution by Day of Week

        demand_by_day = df_ATM_Filtered.groupby('Day_of_Week')['Cash_Demand_Next_Day'].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()


        st.subheader("Cash Demand by Day of Week")
        fig, ax = plt.subplots(figsize=(10,6))
        bars = plt.bar(demand_by_day['Day_of_Week'], demand_by_day['Cash_Demand_Next_Day'], 
               color='cornflowerblue', edgecolor='black')

        plt.title("Cash Demand Distribution by Day of Week", fontsize=16)
        plt.xlabel("Day of Week", fontsize=12)
        plt.ylabel("Average Cash Demand Next Day", fontsize=12)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (0.01 * height), f"{height/1000:,.2f}k", ha = 'center', va = 'bottom', fontsize = 9)

        plt.tight_layout()
        st.pyplot(fig)

# 16. Cash Demand Distribution by Time of Day

        demand_by_time = df_ATM_Filtered.groupby('Time_of_Day')['Cash_Demand_Next_Day'].mean().reindex(["Morning","Afternoon","Evening","Night"]).reset_index()

        st.subheader("Cash Demand by Time of Day")
        fig, ax = plt.subplots(figsize=(8,6))
        bars = plt.bar(demand_by_time['Time_of_Day'], demand_by_time['Cash_Demand_Next_Day'], 
               color='orchid', edgecolor='black')

        plt.title("Cash Demand Distribution by Time of Day", fontsize=16)
        plt.xlabel("Time of Day", fontsize=12)
        plt.ylabel("Average Demand during the day", fontsize=12)

# Add labels
        for bar, value in zip(bars, demand_by_time['Cash_Demand_Next_Day']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f"{value/1000:.2f}K", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # 17 Cash Shortage Risk


        daily = df_ATM_Filtered.groupby(['ATM_ID', 'Date']).agg({'Total_Withdrawals':'sum','Total_Deposits':'sum','Previous_Day_Cash_Level':'mean'}).reset_index()

        daily['Available_Cash'] = daily['Total_Deposits'] + daily['Previous_Day_Cash_Level']
        daily['Shortage_Day'] = (daily['Total_Withdrawals'] > daily['Available_Cash']).astype(int)

        shortage_stats = daily.groupby('ATM_ID').agg(shortage_days=('Shortage_Day','sum'),total_days=('Date','nunique') ).assign(shortage_pct=lambda d: (d['shortage_days']/d['total_days']*100).round(2)).sort_values('shortage_pct', ascending=False).reset_index()


        top_n = 15
        view = shortage_stats.head(top_n).iloc[::-1]

        st.subheader("Cash Shortage")
        fig,ax = plt.subplots(figsize=(10,7))
        bars = plt.barh(view['ATM_ID'], view['shortage_pct'], color='crimson', edgecolor='black')

        plt.title(f"Top {top_n} ATMs by Cash Shortage Risk", fontsize=16)
        plt.xlabel("% of Days in Shortage", fontsize=12)
        plt.ylabel("ATM ID", fontsize=12)
    
# Add labels
        for bar, pct in zip(bars, view['shortage_pct']):
            width = bar.get_width()
            plt.text(
                width + (0.01 * max(view['shortage_pct'])),
                bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", 
                va='center', ha='left', fontsize=9
    )


        plt.tight_layout()
        st.pyplot(fig)



# -------------------------------
#  Withdrawal Seasonality
# -------------------------------

    
    elif selected_page == "WITHDRAWAL SEASONALITY":
        st.markdown(
    """
    <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
        WITHDRAWAL SEASONALITY ANALYSIS
    </h1>
    """,
    unsafe_allow_html=True
)  
        
        # 18. withdrawal by day


        withdrawal_by_day = df_ATM_Filtered.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()


        st.subheader("Withdrawal by Day of Week")
        fig, ax = plt.subplots(figsize=(10,6))
        bars = plt.bar(withdrawal_by_day['Day_of_Week'], withdrawal_by_day['Total_Withdrawals'], 
               color='cornflowerblue', edgecolor='black')

        plt.title("Withdrawal Distribution by Day of Week", fontsize=16)
        plt.xlabel("Day of Week", fontsize=12)
        plt.ylabel("Average Withdrawals", fontsize=12)

# Add labels
        for bar, value in zip(bars, withdrawal_by_day['Total_Withdrawals']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f"{value/1000:.2f}k", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

# 19. Withdrawal Distribution by Time of Day

        withdrawal_by_time = df_ATM_Filtered.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(["Morning","Afternoon","Evening","Night"]).reset_index()

        st.subheader("Withdrwal by Time of Day")
        fig, ax = plt.subplots(figsize=(8,6))
        bars = plt.bar(withdrawal_by_time['Time_of_Day'], withdrawal_by_time['Total_Withdrawals'], 
               color='orchid', edgecolor='black')

        plt.title("Withdrawal Distribution by Time of Day", fontsize=16)
        plt.xlabel("Time of Day", fontsize=12)
        plt.ylabel("Average Withdrawals during Day", fontsize=12)

        for bar, value in zip(bars, withdrawal_by_time['Total_Withdrawals']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f"{value/1000:.2f}K", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)


# 20 Withdrawal_by_month

        df_ATM_Filtered['Date'] = pd.to_datetime(df_ATM_Filtered['Date'], errors='coerce')

        df_ATM_Filtered['YearMonth'] = df_ATM_Filtered['Date'].dt.to_period('M')
        monthly_trend = (df_ATM_Filtered.groupby('YearMonth')['Total_Withdrawals'].sum().reset_index())

        monthly_trend['YearMonth'] = monthly_trend['YearMonth'].dt.to_timestamp()

        st.subheader("Monthly Withdrawal")
        fig, ax = plt.subplots(figsize=(12,6))
        plt.plot(monthly_trend['YearMonth'], monthly_trend['Total_Withdrawals'], 
             marker='o', color='teal', linewidth=2)

        plt.title("Monthly Trend of ATM Withdrawals", fontsize=16)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Total Withdrawals", fontsize=12)

# Add labels
        for x, y in zip(monthly_trend['YearMonth'], monthly_trend['Total_Withdrawals']):
            plt.text(x, y + (0.02 * monthly_trend['Total_Withdrawals'].max()),  # dynamic offset
             f"{y/1e6:.1f}M", ha='center', va='bottom', fontsize=9, fontweight='bold')  

        plt.xticks(rotation=45)
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)

#21. Overall MOM Growth 

        monthly_trend['MoM_Growth_%'] = monthly_trend['Total_Withdrawals'].pct_change() * 100

        st.subheader("Month on Month Growth")
        fig, ax = plt.subplots(figsize=(8,6))
        bars = plt.bar(monthly_trend['YearMonth'].astype(str), monthly_trend['MoM_Growth_%'], 
               color='teal', edgecolor='black')

        plt.title("Month-over-Month Growth in Withdrawals (Overall)", fontsize=16)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Growth Rate (%)", fontsize=12)

# Add % labels
        for bar, pct in zip(bars, monthly_trend['MoM_Growth_%']):
            if not np.isnan(pct):
                plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + (1 if pct >= 0 else -3),
                 f"{pct:.1f}%", ha='center',
                 va='bottom' if pct >= 0 else 'top', fontsize=9)

        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45)
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)

#22 weekday vs weekend 

        df_ATM_Filtered['Date'] = pd.to_datetime(df_ATM_Filtered['Date'], errors='coerce')

        df_ATM_Filtered['DayOfWeek'] = df_ATM_Filtered['Date'].dt.day_name()
        df_ATM_Filtered['Is_Weekend'] = df_ATM_Filtered['DayOfWeek'].isin(['Saturday','Sunday'])

        week_pattern = (df_ATM_Filtered.groupby('Is_Weekend')['Total_Withdrawals'].sum().reset_index())

        week_pattern['Day_Type'] = week_pattern['Is_Weekend'].map({False:'Weekday', True:'Weekend'})

        st.subheader("Weekday-Weekend Analysis")
        fig, ax = plt.subplots(figsize=(6,6))

        plt.pie(
            week_pattern['Total_Withdrawals'],
            labels=week_pattern['Day_Type'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['cornflowerblue','salmon'],
            wedgeprops={'edgecolor':'black'})

        plt.title("Weekend vs Weekday Withdrawal Share", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

#23 Quarterly trends 


        df_ATM_Filtered['Date'] = pd.to_datetime(df_ATM_Filtered['Date'], errors='coerce')

        df_ATM_Filtered['Quarter'] = df_ATM_Filtered['Date'].dt.to_period('Q')

        quarterly_trend = (df_ATM_Filtered.groupby('Quarter')['Total_Withdrawals'].sum().reset_index())

# Convert for plotting
        quarterly_trend['Quarter'] = quarterly_trend['Quarter'].astype(str)

# Plot line + markers
        st.subheader("Quarterly Trends")
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(quarterly_trend['Quarter'], quarterly_trend['Total_Withdrawals'],
         marker='o', color='teal', linewidth=2)

        plt.title("Quarterly Trend of ATM Withdrawals", fontsize=16)
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("Total Withdrawals", fontsize=12)

# Labels
        max_val = quarterly_trend['Total_Withdrawals'].max()
        for i, (x, y) in enumerate(zip(quarterly_trend['Quarter'], quarterly_trend['Total_Withdrawals'])):
            plt.text(
                i, y + (0.02 * max_val),
                f"{y/1e6:.1f}M",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
    )
        plt.xticks(rotation=45)
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
# --------------------------
# MAIN APP
# --------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if not st.session_state.logged_in:
        login_page()

    elif st.session_state.page == "business":
        business_context()
    elif st.session_state.page == "dashboard":
        dashboard()

    else:
        business_context()
 

if __name__ == "__main__":
    main()








