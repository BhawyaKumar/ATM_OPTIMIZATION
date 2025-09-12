import streamlit as st
import pickle
import joblib
import os, base64, runpy
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import json
import functools

st.set_page_config(layout="wide", page_title="ATM Cash Forecast", page_icon="🏧")

# ---------- CONFIG ----------
LOGIN_BG = "assets/ml_login_bg.webp.jpg"
USERS = {"Bhawya": "1234"}
OLS_SCRIPT = "Ols.py"

# ---------- helpers ----------
def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_and_apply_css(logged_in: bool):
    if not logged_in:
        if os.path.exists(LOGIN_BG):
            bg_style = 'background-image: url("data:image/png;base64,{}");'.format(to_b64(LOGIN_BG))
        else:
            bg_style = "background: linear-gradient(90deg,#0b1226,#1f2a6b);"
    else:
        bg_style = "background: linear-gradient(180deg,#09263f,#0f2a44 40%,#102d4a 100%);"

    css = f"""
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu, header, footer {{ visibility: hidden; }}

    .stApp {{
        {bg_style}
        background-size: cover;
        background-position: center;
        min-height: 100vh;
    }}

    /* ---------- NAVBAR ---------- */
    .nav-strip {{
        width: 100%;
        background: rgba(255,255,255,0.04);
        padding: 12px 28px;
        box-sizing: border-box;
        position: relative;
        top: 0;
        z-index: 9999;
        display:flex;
        align-items:center;
        justify-content:space-between;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 6px 18px rgba(2,10,20,0.35);
    }}

    .nav-left, .nav-center, .nav-right {{
        display:flex;
        align-items:center;
        gap: 12px;
    }}
    .nav-left {{ flex: 0 0 auto; }}
    .nav-center {{ flex: 1 1 auto; justify-content:center; }}
    .nav-right {{ flex: 0 0 auto; justify-content:flex-end; }}

    .hello-text {{
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
    }}

    .title-main {{
        color: #ffffff !important;
        font-size: 40px;
        font-weight: 900;
        margin: 0;
        text-decoration: underline;
        text-underline-offset: 8px;
        text-decoration-thickness: 3px;
        text-shadow: 0 3px 12px rgba(0,0,0,0.45);
    }}

    /* ---------- PAGE BODY ---------- */
    .page-body {{
        padding: 28px 44px;
    }}

    .input-card {{
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(2,10,20,0.25);
        border: 1px solid rgba(255,255,255,0.03);
        min-height: 180px;
        justify-content: center;
        align-items: center; 
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        display: flex;

    }}

    .section-title {{
        color: rgba(255,255,255,0.92);
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
        font-size: 15px;
        margin: 0;
        letter-spacing: 0.6px;
    }}

    /* ---------- LABELS ---------- */
  
    .page-body label,
    
    .page-body div[class^="stTextInput"] label,
    .page-body div[class^="stNumberInput"] label,
    .page-body div[class^="stDateInput"] label,
    .page-body div[class^="stSelectbox"] label,
    .page-body div[class*="stTextInput"] label,
    .page-body div[class*="stNumberInput"] label,
    .page-body div[class*="stSelectbox"] label,
    /* target data-testid based widget containers */
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stSelectbox"] label,
    /* when Streamlit places label as a sibling element after your raw HTML */
    .input-card + div label,
    .input-card ~ div label,
    /* generic fallback: block-container/element-container patterns */
    .stApp .block-container .element-container label
    {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        display: block !important;
        margin-bottom: 1px !important;
        margin-top: 7px !important;
        line-height: 1.2 !important;
        opacity: 1 !important;
        text-align: center;
    }}

    .page-body span[role="label"],
    .page-body span[class*="label"],
    .page-body div[role="group"] > span,
    .page-body div[role="group"] > label
    {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        display: block !important;
        margin-bottom: 6px !important;
        line-height: 1.2 !important;
        opacity: 1 !important;
    }}

    /* ---------- INPUT FIELDS ---------- */
    .stTextInput>div>div>input,
    .stNumberInput>div>input,
    .stDateInput>div>input,
    .stSelectbox>div>div {{
        background: #ffffff !important;
        color: #0b2a44 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }}


    /* Metrics / prediction box */
    .pred-box {{
        background: rgba(255,255,255,0.98);
        color: #0b2a44;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }}

    /* ---------- BUTTONS ---------- */
    
    .stButton>button:hover {{
        transform: translateY(-2px);
    }}

    div[data-testid="stButton"][key="predict_btn"] > button {{
        background: #ffffff !important;
        color: #0b2a44 !important;
        border: 1px solid #ccc !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 10px 28px !important;   
        white-space: nowrap !important; 
        min-width: 250px !important;    
        text-align: center !important;
}}


    /* Logout button (light pill) */
    div[data-testid="stButton"][key="logout_bottom"] > button,
    button[data-testid="logout_bottom"]
    {{
        background: #ffffff !important;
        color: #0b2a44 !important;
        border: 1px solid #ccc !important;
        padding: 10px 20px !important;
        border-radius: 18px !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stButton"][key="logout_bottom"] > button:hover {{
        background: #f5f5f5 !important;
    }}

    /* ---------- SELECTBOX ---------- */
   
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stSelectbox"] div[role="combobox"],
    div[data-testid="stSelectbox"] div[role="button"] {{
        width: 100% !important;
        box-sizing: border-box !important;
}}

/* Visible value area / combobox look */
    div[data-testid="stSelectbox"] > div > div {{
        background: #ffffff !important;      /* white background */
        color: #0b2a44 !important;           /* dark text */
        border-radius: 8px !important;
        padding: 8px 12px !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        overflow: visible !important;
}}

/* Some Streamlit versions render the "value" in a div[value] — keep it readable */
    .page-body div[value] {{
        color: #0b2a44 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
}}

/* If the select uses a native <select> element */
    div[data-testid="stSelectbox"] select {{
        width: 100% !important;
        background: #ffffff !important;
        color: #0b2a44 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        box-sizing: border-box !important;
}}

/* Dropdown caret alignment (optional) */
    div[data-testid="stSelectbox"] svg {{
        color: #0b2a44 !important;
}}

/* Reduce extra left-margin that some Streamlit wrappers have */
    div[data-testid="stSelectbox"] > div {{
        margin: 0 !important;
}}

/* Small responsive tweak */
    @media (max-width: 900px) {{
        div[data-testid="stSelectbox"] > div > div {{
            min-height: 42px !important;
            padding: 8px 10px !important;
    }}
}}

    """
    
    st.markdown(css, unsafe_allow_html=True)

# ---------- Login page ----------

def login_page():
    col_left, col_right = st.columns([0.45, 0.55])
    with col_left:
        st.markdown("<div style='padding:40px 6px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:white; margin-top:0;'>Cash Demand Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:white; margin-top:6px;'>🔐 Login</h3>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("", placeholder="Employee name")
            password = st.text_input("", placeholder="Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login Successful")
                    st.rerun()
                else:
                    st.error("Invalid credentials", icon="🚫")

        st.markdown("<div style='margin-top:8px; color:white; font-size:13px;'>Forgot Password?</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.empty()


# ---------- Dashboard ----------
def dashboard():
    username = st.session_state.get("username", "User")

    # NAV STRIP (full width)
    st.markdown(
        f"""
        <div class="nav-strip">
            <div class="nav-left">
                <div class="hello-text">Hello, {username} 👋</div>
            </div>

        <div class="nav-center">
            <div style="text-align:center;">
                <h1 class="title-main">ATM Cash Prediction</h1>
            </div>
        </div>


       
        """,
        unsafe_allow_html=True,
    )


    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    # Input cards in a single row of 3 cards
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("""<div class="input-card"> 
                        <div class="section-title" style="text-align:center;">ATM DETAILS</div>""",
                    unsafe_allow_html=True,
                    )
    
        atm_id = st.text_input("ATM ID", value = "ATM_0001", key="atm_select")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="input-card"> 
                        <div class="section-title" style="text-align:center;">Daily Transaction Data</div>""",
                    unsafe_allow_html=True,)
        cash_withdrawn = st.number_input("Cash Withdrawn", value=20000.0, step=100.0, key="cash_withdrawn")
        cash_deposited = st.number_input("Cash Deposited", value=5000.0, step=100.0, key="cash_deposited")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="input-card"> 
                        <div class="section-title" style="text-align:center;">Contextual Features</div>""",
                     unsafe_allow_html=True,)
        date = st.date_input("Date", key="date_field")
        holiday_flag_label = st.selectbox("Holiday Flag",["Select holiday","Holiday", "Not Holiday"],index=0, key="holiday_flag")
        holiday_flag = None if holiday_flag_label == "Select holiday" else (1 if holiday_flag_label == "Holiday" else 0)
        special_event_label = st.selectbox("Special Event Flag", ["Select Event","Not Special", "Special"], index=0,key="special_event_flag")
        special_event_flag = None if special_event_label == "Select event" else (1 if special_event_label == "Special" else 0)
        st.markdown("</div>", unsafe_allow_html=True)

    # === PREDICTION HANDLER ===
    @functools.lru_cache(maxsize=1)
    def load_ols_coeffs(path="ols_coeffs.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Coefficients file not found: {path}")
        with open(path, "r") as f:
            coefs = json.load(f)
        return {k: float(v) for k, v in coefs.items()}

    def predict_using_coeffs(cash_withdrawn, cash_deposited, holiday_flag, special_event_flag, coeffs):
        # Build input dict with training column names
        X = {
            "Total_Withdrawals": float(cash_withdrawn),
            "Total_Deposits": float(cash_deposited),
            "Holiday_Flag": int(0 if holiday_flag is None else holiday_flag),
            "Special_Event_Flag": int(0 if special_event_flag is None else special_event_flag)
        }

        # Identify intercept (if present)
        intercept_key = None
        for k in coeffs.keys():
            if k.lower() in ("intercept", "const"):
                intercept_key = k
                break
        intercept = coeffs.get(intercept_key, 0.0) if intercept_key is not None else 0.0

        # Sum contributions for features that exist in coeffs
        pred = intercept
        used_features = []
        contributions = {}
        for feat, val in X.items():
            if feat in coeffs:
                contrib = coeffs[feat] * val
                pred += contrib
                contributions[feat] = contrib
                used_features.append(feat)
        return float(pred), intercept_key, used_features, contributions

    # centered model & predict
    c_left, c_center, c_right = st.columns([3, 1, 3])
    with c_center:
        
        if st.button("👉 Predict Next Day Demand", key="predict_btn"):
            try:
                coeffs = load_ols_coeffs("ols_coeffs.json")
                predicted_amount, intercept_key, used_features, contributions = predict_using_coeffs(
                    cash_withdrawn,
                    cash_deposited,
                    holiday_flag,
                    special_event_flag,
                    coeffs
                )
                st.session_state["last_prediction"] = {
                    "atm_id": atm_id,
                    "predicted_amount": round(float(predicted_amount), 2),
                    "model_used": "OLS (coeffs json)",
                    "input_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "used_features": used_features,
                    "intercept": float(coeffs.get(intercept_key, 0.0)),
                    "contributions": {k: round(float(v), 2) for k, v in contributions.items()}
                }
                
            except FileNotFoundError as e:
                st.error(str(e))
                # fallback behavior (optional)
                fallback = float(cash_withdrawn - cash_deposited)
                st.warning("Returning fallback heuristic (withdrawn - deposited). Save ols_coeffs.json to use model.")
                st.session_state["last_prediction"] = {
                    "atm_id": atm_id,
                    "predicted_amount": round(float(fallback), 2),
                    "model_used": "Fallback heuristic",
                    "input_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

    # Latest prediction card
    if "last_prediction" in st.session_state:
        p = st.session_state["last_prediction"]
        st.markdown(
            f"""
            <div class="pred-box">
                <div style="font-weight:700; font-size:20px;">Predicted cash required:  {p['predicted_amount']}</div>
                <div style="color:#444; margin-top:6px;"> Predicted at: {p['input_ts']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Logout button

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # spacer
    if st.button("🚪 Logout", key="logout_bottom"):
        for k in ("logged_in", "username"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ---------- Main ----------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    build_and_apply_css(st.session_state.logged_in)

    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()

