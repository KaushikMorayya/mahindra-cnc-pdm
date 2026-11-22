import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
)
import streamlit as st

st.set_page_config(
    page_title="MAHINDRA PULSE™ | CNC Asset Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 1. INDUSTRIAL ENTERPRISE THEME & BRANDED HEADER CSS
# ---------------------------------------------------------
st.markdown(
    """
<style>
    /* Clean Industrial Dark-Slate & Mahindra Crimson Palette */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .brand-navbar {
        background: linear-gradient(90deg, #141b29 0%, #1f1218 55%, #141b29 100%);
        border: 1px solid #2d3748;
        border-left: 5px solid #e31837;
        border-radius: 10px;
        padding: 14px 22px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.6px;
        color: #ffffff;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 2px;
    }
    .live-pill {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .cnc-card {
        background-color: #131a26;
        border: 1px solid #263042;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 8px;
    }
    .cnc-card-selected {
        border: 2px solid #e31837;
        box-shadow: 0 0 14px rgba(227, 24, 55, 0.25);
    }
    div[data-testid="stMetric"] {
        background-color: #131a26;
        border: 1px solid #253044;
        padding: 12px 14px;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. PERSISTENT SQLITE SAP-PM CMMS DATABASE
# ---------------------------------------------------------
DB_FILE = "mahindra_pulse_cmms.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sap_notif_id TEXT,
            created_at TEXT,
            machine TEXT,
            action_type TEXT,
            target_component TEXT,
            rul_hours REAL,
            core_team TEXT,
            spare_status TEXT,
            est_cost_inr INTEGER,
            scheduled_window TEXT,
            status TEXT,
            iso_close_code TEXT
        )
    """)
    c.execute("SELECT COUNT(*) FROM work_orders")
    if c.fetchone()[0] == 0:
        seed_orders = [
            (
                "SAP-PM-40912",
                "2026-11-20 10:15",
                "CNC-02 (Boring)",
                "REPLACE_SPARE",
                "SKF_7014_Spindle_Bearing",
                52.0,
                "Mechanical Team",
                "RESERVED (Store Bin B-14)",
                42000,
                "SATURDAY 09:00 AM (OFF)",
                "APPROVED_WO",
                "Scheduled for Sat LOTO",
            ),
            (
                "SAP-PM-40908",
                "2026-11-19 15:40",
                "CNC-03 (Drilling)",
                "REPAIR_CALIBRATE",
                "Coolant_Chip_Strainer_Flush",
                64.5,
                "Mechanical Team",
                "NO SPARE NEEDED (Clean/Flush)",
                1500,
                "SATURDAY 11:00 AM (OFF)",
                "AUTO_NOTIFICATION",
                "Awaiting Planner Approval",
            ),
            (
                "SAP-PM-40891",
                "2026-11-15 09:30",
                "CNC-04 (Finishing)",
                "REPAIR_CALIBRATE",
                "Z_Axis_PLC_Backlash_Offset",
                48.0,
                "Electrical Team",
                "NO SPARE NEEDED (PLC Calib)",
                1200,
                "SUNDAY 10:00 AM (OFF)",
                "CLOSED",
                "ISO-14224: EL-09 (Calibrated)",
            ),
        ]
        c.executemany(
            "INSERT INTO work_orders (sap_notif_id, created_at, machine, action_type,"
            " target_component, rul_hours, core_team, spare_status, est_cost_inr,"
            " scheduled_window, status, iso_close_code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            seed_orders,
        )
    conn.commit()
    conn.close()


def create_work_order(machine, action_type, component, rul, team, window):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT id FROM work_orders WHERE machine=? AND status != 'CLOSED'",
        (machine,),
    )
    if c.fetchone() is None:
        notif_id = f"SAP-PM-{np.random.randint(41000, 49999)}"
        if action_type == "REPAIR_CALIBRATE":
            spare_st = "NO SPARE NEEDED (Adjust/Clean)"
            est_cost = 1500
        else:
            spare_st = "PENDING STORE RESERVATION"
            est_cost = 42000 if "Bearing" in component or "Capacitor" in component else 8500
        c.execute(
            "INSERT INTO work_orders (sap_notif_id, created_at, machine, action_type,"
            " target_component, rul_hours, core_team, spare_status, est_cost_inr,"
            " scheduled_window, status, iso_close_code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                notif_id,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                machine,
                action_type,
                component,
                rul,
                team,
                spare_st,
                est_cost,
                window,
                "AUTO_NOTIFICATION",
                "Open",
            ),
        )
        conn.commit()
        created = True
    else:
        created = False
    conn.close()
    return created


def approve_notification(order_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT action_type FROM work_orders WHERE id=?", (order_id,))
    row = c.fetchone()
    spare_txt = (
        "NO SPARE NEEDED (Tools Ready)"
        if row and row[0] == "REPAIR_CALIBRATE"
        else "RESERVED (Store Bin A-09)"
    )
    c.execute(
        "UPDATE work_orders SET status='APPROVED_WO', spare_status=? WHERE id=?",
        (spare_txt, order_id),
    )
    conn.commit()
    conn.close()


def close_work_order(order_id, iso_code):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE work_orders SET status='CLOSED', spare_status='COMPLETED',"
        " iso_close_code=? WHERE id=?",
        (iso_code, order_id),
    )
    conn.commit()
    conn.close()


def fetch_work_orders():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM work_orders ORDER BY id DESC", conn)
    conn.close()
    return df


init_db()

# ---------------------------------------------------------
# 3. 12 HIGH-VALUE ENGINEERED PARAMETERS & ML ENGINE
# ---------------------------------------------------------
ENGINEERED_12_COLS = [
    "Vib_Crest_Severity_Idx",
    "Thermal_Drift_Slope_Hr",
    "True_Active_Power_kW",
    "Specific_Cutting_Energy",
    "Coolant_Circuit_Health",
    "Axis_Backlash_Servo_Idx",
    "Lube_Film_Integrity_Idx",
    "Chuck_Clamping_Stability",
    "Drive_Harmonic_Health",
    "Hydraulic_Power_Efficiency",
    "Spindle_Runout_Chatter",
    "Composite_OEE_Risk_Score",
]

RAW_44_COLS = [
    "Vib_X_mm_s", "Vib_Y_mm_s", "Vib_Z_mm_s", "Peak_Accel_G", "Acoustic_dB",
    "Spindle_Current_A", "Voltage_L1_V", "Voltage_L2_V", "Voltage_L3_V",
    "Power_Factor", "X_Servo_Amps", "Y_Servo_Amps", "Z_Servo_Amps",
    "Front_Brg_Temp_C", "Rear_Brg_Temp_C", "Ambient_Temp_C", "Coolant_Temp_C",
    "Slideway_Temp_C", "Hyd_Oil_Temp_C", "IGBT_Heatsink_Temp_C",
    "Coolant_Press_Bar", "Coolant_Flow_LPM", "Filter_Diff_Press_Bar",
    "Lube_Oil_Press_Bar", "Lube_Flow_CC_Min", "Lube_Tank_Level_Pct",
    "Pneumatic_Line_Bar", "Chuck_Clamp_Press_Bar", "Workpiece_Slip_Flag",
    "Hyd_Pump_Press_Bar", "Hyd_Motor_Amps", "Accumulator_Press_Bar",
    "Spindle_RPM", "Cutting_Torque_Nm", "Feed_Rate_mm_min", "Depth_of_Cut_mm",
    "Tool_Run_Min", "Following_Error_um", "Actual_Cycle_Sec", "Drive_DC_Bus_V",
    "DC_Bus_Ripple_V", "Drive_Load_Pct", "Surface_Roughness_Ra", "Unbalance_Phase_Deg",
]

ACTION_LABELS = {0: "HEALTHY", 1: "REPAIR_CALIBRATE", 2: "REPLACE_SPARE"}


def compute_12_engineered_features(df):
    out = df.copy()
    rms_vib = np.sqrt(out["Vib_X_mm_s"] ** 2 + out["Vib_Y_mm_s"] ** 2 + out["Vib_Z_mm_s"] ** 2)
    out["Vib_Crest_Severity_Idx"] = np.round(rms_vib * (out["Peak_Accel_G"] / np.maximum(rms_vib, 0.5)), 2)
    temp_gap = np.maximum(0.0, out["Front_Brg_Temp_C"] - out["Ambient_Temp_C"])
    out["Thermal_Drift_Slope_Hr"] = np.round((temp_gap / 8.5) * (1.0 + np.maximum(0.0, out["Vib_Crest_Severity_Idx"] - 2.0) * 0.18), 2)
    v_avg = (out["Voltage_L1_V"] + out["Voltage_L2_V"] + out["Voltage_L3_V"]) / 3.0
    out["True_Active_Power_kW"] = np.round((1.732 * v_avg * out["Spindle_Current_A"] * out["Power_Factor"]) / 1000.0, 2)
    out["Specific_Cutting_Energy"] = np.round((out["Cutting_Torque_Nm"] * (out["Spindle_RPM"] * 2 * np.pi / 60.0)) / np.maximum(out["Feed_Rate_mm_min"] * out["Depth_of_Cut_mm"], 1.0), 2)
    out["Coolant_Circuit_Health"] = np.round(np.clip((out["Coolant_Flow_LPM"] / 24.5) * (1.0 - out["Filter_Diff_Press_Bar"] / 2.2), 0.05, 1.0), 2)
    out["Axis_Backlash_Servo_Idx"] = np.round(out["Following_Error_um"] * (out["Z_Servo_Amps"] / 5.2), 2)
    out["Lube_Film_Integrity_Idx"] = np.round((out["Lube_Oil_Press_Bar"] * out["Lube_Flow_CC_Min"]) / np.maximum(out["Slideway_Temp_C"], 10.0), 2)
    out["Chuck_Clamping_Stability"] = np.round((out["Chuck_Clamp_Press_Bar"] / np.maximum(out["Pneumatic_Line_Bar"], 1.0)) * ((out["Spindle_RPM"] / 1650.0) ** 2), 2)
    out["Drive_Harmonic_Health"] = np.round((out["DC_Bus_Ripple_V"] / np.maximum(out["Drive_DC_Bus_V"], 100.0)) * (out["IGBT_Heatsink_Temp_C"] / 60.0) * 100.0, 2)
    out["Hydraulic_Power_Efficiency"] = np.round((out["Hyd_Pump_Press_Bar"] * 18.0) / np.maximum(out["Hyd_Motor_Amps"] * 415.0, 1.0), 2)
    out["Spindle_Runout_Chatter"] = np.round((out["Acoustic_dB"] * np.abs(out["Vib_X_mm_s"] - out["Vib_Z_mm_s"])) / np.maximum(out["Spindle_RPM"] / 1000.0, 0.5), 2)
    out["Composite_OEE_Risk_Score"] = np.round(
        np.clip(
            100.0
            - (out["Vib_Crest_Severity_Idx"] * 4.0)
            - (out["Thermal_Drift_Slope_Hr"] * 4.8)
            - ((1.0 - out["Coolant_Circuit_Health"]) * 28.0)
            - (np.maximum(0.0, out["Axis_Backlash_Servo_Idx"] - 6.0) * 1.6)
            - (np.maximum(0.0, out["Drive_Harmonic_Health"] - 1.2) * 8.5),
            5.0,
            99.5,
        ),
        1,
    )
    return out


def prescribe_part_action(row):
    if row["Vib_Crest_Severity_Idx"] > 7.5 and row["Thermal_Drift_Slope_Hr"] > 2.4:
        return "REPLACE_SPARE", "SKF_7014_Spindle_Bearing", "Mechanical Team"
    if row["Specific_Cutting_Energy"] > 2.45 and row.get("Tool_Run_Min", 180) > 165:
        return "REPLACE_SPARE", "CNMG_Carbide_Cutting_Insert", "Mechanical Team"
    if row["Coolant_Circuit_Health"] < 0.45:
        return "REPAIR_CALIBRATE", "Coolant_Chip_Strainer_Flush", "Mechanical Team"
    if row["Drive_Harmonic_Health"] > 2.65:
        return "REPLACE_SPARE", "Siemens_S120_DCLink_Capacitor", "Electrical Team"
    if row["Axis_Backlash_Servo_Idx"] > 21.0:
        return "REPLACE_SPARE", "Z_Axis_Ball_Screw_Nut", "Mechanical Team"
    if row["Axis_Backlash_Servo_Idx"] > 14.5:
        return "REPAIR_CALIBRATE", "Z_Axis_PLC_Backlash_Offset", "Electrical Team"
    return "HEALTHY", "None (Nominal)", "Preventive Monitor"


@st.cache_resource
def generate_plant_corpus(num_days=10):
    np.random.seed(42)
    all_days = pd.date_range("2026-10-05", periods=num_days, freq="B")
    minute_index = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(d, d + pd.Timedelta(hours=23, minutes=59), freq="1min")
            for d in all_days
        ])
    )
    n_m = len(minute_index)
    machines = ["CNC-01", "CNC-02", "CNC-03", "CNC-04"]
    frames = []

    for m_idx, m_id in enumerate(machines):
        day_num = np.arange(n_m) / 1440.0
        bearing_wave = np.where((m_id == "CNC-02") & (day_num >= 4.0) & (day_num <= 8.5), ((day_num - 4.0) / 4.5) ** 1.3 * 2.1, 0.06 * np.sin(day_num * 1.4) ** 2)
        tool_min = (np.arange(n_m) % (210 if m_id == "CNC-02" else 260)) * 0.95
        tool_factor = np.clip(tool_min / 200.0, 0, 1.3)
        clog_wave = np.where((m_id == "CNC-03") & (day_num >= 2.0) & (day_num <= 5.0), ((day_num - 2.0) / 3.0) * 1.75, 0.08 * (day_num % 2.5) / 2.5)
        drive_wave = np.where((m_id == "CNC-03") & (day_num >= 7.0) & (day_num <= 9.8), ((day_num - 7.0) / 2.8) * 2.0, 0.05)
        servo_cal = np.where((m_id == "CNC-04") & (day_num >= 3.0) & (day_num <= 6.0), ((day_num - 3.0) / 3.0) * 1.45, 0.0)
        servo_rep = np.where((m_id == "CNC-04") & (day_num >= 7.5) & (day_num <= 9.8), ((day_num - 7.5) / 2.3) * 2.1, 0.0)
        servo_tot = servo_cal + servo_rep

        hour_of_day = minute_index.hour + minute_index.minute / 60.0
        amb_temp = 27.5 + 2.6 * np.sin((hour_of_day - 9) * np.pi / 12.0) + np.random.normal(0, 0.15, n_m)
        rpm = 1650 - bearing_wave * 140 - tool_factor * 80 + np.random.normal(0, 12, n_m)
        torque = 35.5 + tool_factor * 18.5 + bearing_wave * 9.0 + np.random.normal(0, 1.0, n_m)
        pf = 0.88 - drive_wave * 0.04 - tool_factor * 0.015 + np.random.normal(0, 0.004, n_m)

        df_m = pd.DataFrame({
            "Timestamp": minute_index,
            "Machine_ID": m_id,
            "Shift_ID": np.where(minute_index.hour < 6, "Shift_C", np.where(minute_index.hour < 14, "Shift_A", "Shift_B")),
            "Part_Batch_ID": f"BCH-2026-10-M{m_idx+1}",
            "Vib_X_mm_s": 1.10 + bearing_wave * 2.3 + tool_factor * 0.65 + np.random.normal(0, 0.07, n_m),
            "Vib_Y_mm_s": 1.02 + bearing_wave * 2.1 + tool_factor * 0.55 + np.random.normal(0, 0.07, n_m),
            "Vib_Z_mm_s": 0.95 + bearing_wave * 1.9 + tool_factor * 0.50 + np.random.normal(0, 0.06, n_m),
            "Peak_Accel_G": 1.65 + (bearing_wave ** 1.45) * 4.8 + tool_factor * 1.1 + np.random.normal(0, 0.14, n_m),
            "Acoustic_dB": 71.5 + bearing_wave * 11.2 + tool_factor * 4.2 + np.random.normal(0, 0.5, n_m),
            "Spindle_Current_A": (torque * rpm * (2 * np.pi / 60.0)) / (1.732 * 415.0 * pf) + drive_wave * 3.8 + np.random.normal(0, 0.22, n_m),
            "Voltage_L1_V": 415.0 + np.random.normal(0, 1.6, n_m),
            "Voltage_L2_V": 414.2 + np.random.normal(0, 1.6, n_m),
            "Voltage_L3_V": 415.8 + np.random.normal(0, 1.6, n_m),
            "Power_Factor": pf,
            "X_Servo_Amps": 4.1 + servo_tot * 1.1 + np.random.normal(0, 0.12, n_m),
            "Y_Servo_Amps": 3.9 + servo_tot * 0.9 + np.random.normal(0, 0.12, n_m),
            "Z_Servo_Amps": 5.0 + servo_cal * 0.8 + servo_rep * 2.4 + np.random.normal(0, 0.18, n_m),
            "Front_Brg_Temp_C": amb_temp + 8.5 + bearing_wave * 10.5 + clog_wave * 3.4 + np.random.normal(0, 0.25, n_m),
            "Rear_Brg_Temp_C": amb_temp + 7.0 + bearing_wave * 8.0 + clog_wave * 2.6 + np.random.normal(0, 0.25, n_m),
            "Ambient_Temp_C": amb_temp,
            "Coolant_Temp_C": 24.2 + clog_wave * 3.6 + np.random.normal(0, 0.18, n_m),
            "Slideway_Temp_C": amb_temp + 4.0 + servo_tot * 2.8 + np.random.normal(0, 0.2, n_m),
            "Hyd_Oil_Temp_C": amb_temp + 10.5 + np.random.normal(0, 0.25, n_m),
            "IGBT_Heatsink_Temp_C": amb_temp + 16.5 + drive_wave * 9.5 + np.random.normal(0, 0.35, n_m),
            "Coolant_Press_Bar": 4.2 + clog_wave * 0.42 + np.random.normal(0, 0.07, n_m),
            "Coolant_Flow_LPM": 24.5 - clog_wave * 9.4 + np.random.normal(0, 0.35, n_m),
            "Filter_Diff_Press_Bar": 0.24 + clog_wave * 1.12 + np.random.normal(0, 0.03, n_m),
            "Lube_Oil_Press_Bar": 2.85 - (day_num % 5.0) * 0.04 + np.random.normal(0, 0.04, n_m),
            "Lube_Flow_CC_Min": 12.2 - (day_num % 5.0) * 0.18 + np.random.normal(0, 0.18, n_m),
            "Lube_Tank_Level_Pct": 92.0 - (day_num % 5.0) * 4.5,
            "Pneumatic_Line_Bar": 6.4 + np.random.normal(0, 0.07, n_m),
            "Chuck_Clamp_Press_Bar": 5.85 + np.random.normal(0, 0.05, n_m),
            "Workpiece_Slip_Flag": np.zeros(n_m, dtype=np.int8),
            "Hyd_Pump_Press_Bar": 115.0 + np.random.normal(0, 1.1, n_m),
            "Hyd_Motor_Amps": 6.8 + np.random.normal(0, 0.14, n_m),
            "Accumulator_Press_Bar": 92.0 - (day_num % 11.0) * 0.35 + np.random.normal(0, 0.35, n_m),
            "Spindle_RPM": rpm,
            "Cutting_Torque_Nm": torque,
            "Feed_Rate_mm_min": 425 - tool_factor * 32 + np.random.normal(0, 4, n_m),
            "Depth_of_Cut_mm": 2.0 + 0.2 * np.sin(day_num * 3.1) + np.random.normal(0, 0.05, n_m),
            "Tool_Run_Min": tool_min,
            "Following_Error_um": 5.8 + servo_cal * 11.5 + servo_rep * 14.0 + np.random.normal(0, 0.6, n_m),
            "Actual_Cycle_Sec": 48.0 + servo_tot * 2.6 + np.random.normal(0, 0.18, n_m),
            "Drive_DC_Bus_V": 560.0 + np.random.normal(0, 2.2, n_m),
            "DC_Bus_Ripple_V": 4.2 + drive_wave * 9.2 + np.random.normal(0, 0.28, n_m),
            "Drive_Load_Pct": 51.0 + tool_factor * 18.0 + drive_wave * 16.0 + np.random.normal(0, 1.1, n_m),
            "Surface_Roughness_Ra": 0.78 + tool_factor * 0.95 + bearing_wave * 0.65 + np.random.normal(0, 0.04, n_m),
            "Unbalance_Phase_Deg": 140.0 + np.random.normal(0, 3.5, n_m),
        })
        for col in RAW_44_COLS:
            if col != "Workpiece_Slip_Flag":
                df_m[col] = df_m[col].astype(np.float32)

        df_m = compute_12_engineered_features(df_m)
        act_code = np.zeros(n_m, dtype=np.int8)
        m_bear = (df_m["Vib_Crest_Severity_Idx"] > 7.5) & (df_m["Thermal_Drift_Slope_Hr"] > 2.4)
        m_tool = (~m_bear) & (df_m["Specific_Cutting_Energy"] > 2.45) & (df_m["Tool_Run_Min"] > 165)
        m_cool = (~m_bear) & (~m_tool) & (df_m["Coolant_Circuit_Health"] < 0.45)
        m_drv = (~m_bear) & (~m_tool) & (~m_cool) & (df_m["Drive_Harmonic_Health"] > 2.65)
        m_srv_rep = (~m_bear) & (~m_tool) & (~m_cool) & (~m_drv) & (df_m["Axis_Backlash_Servo_Idx"] > 21.0)
        m_srv_cal = (~m_bear) & (~m_tool) & (~m_cool) & (~m_drv) & (~m_srv_rep) & (df_m["Axis_Backlash_Servo_Idx"] > 14.5)

        act_code[m_cool | m_srv_cal] = 1
        act_code[m_bear | m_tool | m_drv | m_srv_rep] = 2
        df_m["Action_Code"] = act_code
        df_m["Action_Decision"] = [ACTION_LABELS[c] for c in act_code]
        df_m["RUL_Hours"] = np.round(np.clip((df_m["Composite_OEE_Risk_Score"] / 100.0) * 140.0, 2.0, 140.0), 1).astype(np.float32)
        frames.append(df_m)

    return pd.concat(frames, ignore_index=True)


def compress_to_15min_windows(raw_df):
    df = raw_df.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Window_15m"] = df["Timestamp"].dt.floor("15min")
    else:
        df["Window_15m"] = np.arange(len(df)) // 15
    if "Machine_ID" not in df.columns:
        df["Machine_ID"] = "CNC-01"

    agg_dict = {col: "mean" for col in ENGINEERED_12_COLS if col in df.columns}
    if "Tool_Run_Min" in df.columns:
        agg_dict["Tool_Run_Min"] = "max"
    if "Action_Code" in df.columns:
        agg_dict["Action_Code"] = "max"
    if "RUL_Hours" in df.columns:
        agg_dict["RUL_Hours"] = "min"

    return df.groupby(["Machine_ID", "Window_15m"], as_index=False).agg(agg_dict).sort_values(["Window_15m", "Machine_ID"]).reset_index(drop=True)


def train_chronological_models(win_df):
    split_idx = int(len(win_df) * 0.75)
    train_df, test_df = win_df.iloc[:split_idx], win_df.iloc[split_idx:]
    X_tr, y_act_tr, y_rul_tr = train_df[ENGINEERED_12_COLS], train_df["Action_Code"], train_df["RUL_Hours"]
    X_te, y_act_te, y_rul_te = test_df[ENGINEERED_12_COLS], test_df["Action_Code"], test_df["RUL_Hours"]

    clf = RandomForestClassifier(n_estimators=90, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_tr, y_act_tr)
    reg = RandomForestRegressor(n_estimators=90, random_state=42, n_jobs=-1).fit(X_tr, y_rul_tr)
    pred_act, pred_rul = clf.predict(X_te), reg.predict(X_te)

    metrics = {
        "accuracy": round(accuracy_score(y_act_te, pred_act) * 100, 2),
        "precision": round(precision_score(y_act_te, pred_act, average="weighted", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_act_te, pred_act, average="weighted", zero_division=0) * 100, 2),
        "rul_mae": round(mean_absolute_error(y_rul_te, pred_rul), 2),
        "test_windows": len(test_df),
        "cm": confusion_matrix(y_act_te, pred_act, labels=[0, 1, 2]),
    }
    imp = pd.DataFrame({"Parameter": ENGINEERED_12_COLS, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=True)
    return clf, reg, metrics, imp


# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "plant_days" not in st.session_state:
    st.session_state.plant_days = 10
if "selected_cnc" not in st.session_state:
    st.session_state.selected_cnc = "CNC-02 (Boring)"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏭 Command Center & Conveyor Twin"

base_raw_df = generate_plant_corpus(st.session_state.plant_days)
active_raw_df = st.session_state.get("custom_raw_df") if st.session_state.get("custom_raw_df") is not None else base_raw_df
active_win_df = compress_to_15min_windows(active_raw_df)
act_clf, rul_reg, val_metrics, feat_imp_df = train_chronological_models(active_win_df)

HEALTHY_VEC = [1.85, 0.55, 7.2, 1.48, 0.91, 4.2, 1.05, 0.92, 0.65, 1.35, 1.15, 92.5]
DEFAULT_FLEET = {
    "CNC-01 (Roughing)": list(HEALTHY_VEC),
    "CNC-02 (Boring)": [8.40, 2.85, 12.4, 2.62, 0.84, 9.5, 0.98, 0.90, 0.85, 1.32, 3.40, 44.0],
    "CNC-03 (Drilling)": [2.45, 1.85, 7.8, 1.60, 0.31, 5.2, 1.01, 0.91, 0.72, 1.34, 1.45, 66.5],
    "CNC-04 (Finishing)": [2.10, 1.15, 7.5, 1.55, 0.89, 16.4, 0.99, 0.92, 0.68, 1.35, 1.30, 71.0],
}
if "fleet_vectors" not in st.session_state:
    st.session_state.fleet_vectors = {k: list(v) for k, v in DEFAULT_FLEET.items()}

# ---------------------------------------------------------
# 5. BRANDED TOP NAVIGATION BAR (MAHINDRA PULSE™ SVG LOGO)
# ---------------------------------------------------------
st.markdown(
    f"""
<div class="brand-navbar">
    <div style="display:flex; align-items:center; gap:16px;">
        <svg width="52" height="52" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="64" height="64" rx="14" fill="#E31837"/>
            <path d="M12 46V20L24 34L32 23L40 34L52 20V46" stroke="white" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="32" cy="13" r="3.5" fill="#FBBF24"/>
        </svg>
        <div>
            <div class="brand-title">MAHINDRA PULSE™ <span style="color:#e31837;">|</span> CNC ASSET INTELLIGENCE</div>
            <div class="brand-sub">Chakan Powertrain Line-04 • 24/5 Continuous Conveyor PdM &amp; Weekend SAP-PM Orchestrator</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span class="live-pill">● IIoT STREAM ONLINE ({len(active_raw_df):,} ROWS × {active_raw_df.shape[1]} TAGS)</span>
        <div style="font-size:0.75rem; color:#9ca3af; margin-top:5px;">ML Accuracy: <b>{val_metrics['accuracy']}%</b> | RUL MAE: <b>±{val_metrics['rul_mae']} hrs</b></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 6. LEFT NAVIGATION BAR & SYNCHRONIZED TELEMETRY CONSOLE
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 PLATFORM NAVIGATION")
    nav_choice = st.radio(
        "Module Navigation",
        [
            "🏭 Command Center & Conveyor Twin",
            "🔬 Asset Diagnostics & 12-Sensor Fusion",
            "🔧 SAP-PM Work Orders & Dispatch",
            "🗄️ Multi-Lakh Data & ML Audit",
        ],
        index=[
            "🏭 Command Center & Conveyor Twin",
            "🔬 Asset Diagnostics & 12-Sensor Fusion",
            "🔧 SAP-PM Work Orders & Dispatch",
            "🗄️ Multi-Lakh Data & ML Audit",
        ].index(st.session_state.nav_page),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = nav_choice

    st.divider()
    st.markdown("### ⚡ FLOOR SCENARIO SIMULATOR")
    b1, b2 = st.columns(2)
    if b1.button("🟢 All Healthy", use_container_width=True):
        st.session_state.fleet_vectors = {m: list(HEALTHY_VEC) for m in DEFAULT_FLEET}
        st.rerun()
    if b2.button("🟡 Filter Clog (#3)", use_container_width=True):
        st.session_state.fleet_vectors["CNC-03 (Drilling)"][4] = 0.28
        st.session_state.selected_cnc = "CNC-03 (Drilling)"
        st.rerun()

    b3, b4 = st.columns(2)
    if b3.button("🟠 Backlash (#4)", use_container_width=True):
        st.session_state.fleet_vectors["CNC-04 (Finishing)"][5] = 16.8
        st.session_state.selected_cnc = "CNC-04 (Finishing)"
        st.rerun()
    if b4.button("🔴 Bearing (#2)", use_container_width=True):
        st.session_state.fleet_vectors["CNC-02 (Boring)"][0] = 8.9
        st.session_state.fleet_vectors["CNC-02 (Boring)"][1] = 3.1
        st.session_state.selected_cnc = "CNC-02 (Boring)"
        st.rerun()

    st.divider()
    st.markdown("### 🎛️ LIVE MACHINE PARAMETER TUNER")
    cnc_keys = list(st.session_state.fleet_vectors.keys())
    picked_cnc = st.selectbox(
        "Active Machine Focus:",
        cnc_keys,
        index=cnc_keys.index(st.session_state.selected_cnc),
    )
    if picked_cnc != st.session_state.selected_cnc:
        st.session_state.selected_cnc = picked_cnc
        st.rerun()

    cur_v = st.session_state.fleet_vectors[st.session_state.selected_cnc]
    s_vib = st.slider("Vib Crest Severity Idx", 0.8, 12.0, float(cur_v[0]), 0.1, key=f"sv_{picked_cnc}")
    s_thm = st.slider("Thermal Drift Slope (°C/hr)", 0.2, 5.0, float(cur_v[1]), 0.1, key=f"st_{picked_cnc}")
    s_cut = st.slider("Specific Cutting Energy", 1.0, 3.5, float(cur_v[3]), 0.05, key=f"sc_{picked_cnc}")
    s_col = st.slider("Coolant Circuit Health", 0.10, 1.00, float(cur_v[4]), 0.02, key=f"sl_{picked_cnc}")
    s_bkl = st.slider("Axis Backlash Servo Idx", 2.0, 26.0, float(cur_v[5]), 0.2, key=f"sb_{picked_cnc}")
    s_drv = st.slider("Drive Harmonic Health", 0.4, 4.0, float(cur_v[8]), 0.05, key=f"sd_{picked_cnc}")

    flatline_sim = st.checkbox(f"Simulate Tag Flatline ({picked_cnc.split()[0]})", value=False, key=f"fl_{picked_cnc}")

    new_oee = round(
        float(
            np.clip(
                100.0
                - (s_vib * 4.0)
                - (s_thm * 4.8)
                - ((1.0 - s_col) * 28.0)
                - (max(0.0, s_bkl - 6.0) * 1.6)
                - (max(0.0, s_drv - 1.2) * 8.5),
                5.0,
                99.5,
            )
        ),
        1,
    )
    st.session_state.fleet_vectors[picked_cnc] = [
        s_vib, s_thm, cur_v[2], s_cut, s_col, s_bkl,
        cur_v[6], cur_v[7], s_drv, cur_v[9], cur_v[10], new_oee,
    ]

# ---------------------------------------------------------
# 7. REAL-TIME ML INFERENCE ACROSS ALL 4 CONVEYOR STATIONS
# ---------------------------------------------------------
top_bar1, top_bar2, top_bar3, top_bar4, top_bar5 = st.columns([2.2, 1, 1, 1, 1])
with top_bar1:
    shift_sel = st.selectbox(
        "Mon–Fri Production Shift Clock:",
        [
            ("Monday 10:00 AM", 110),
            ("Wednesday 02:00 PM", 58),
            ("Thursday 06:00 PM", 30),
            ("Friday 04:00 PM", 8),
        ],
        index=1,
        format_func=lambda x: f"{x[0]} ({x[1]} hrs left till Fri Midnight)",
    )
hrs_to_fri = shift_sel[1]

fleet_status = {}
line_avail = 100.0
rep_cnt, rpl_cnt = 0, 0

for m_name, vec in st.session_state.fleet_vectors.items():
    r_df = pd.DataFrame([vec], columns=ENGINEERED_12_COLS)
    p_code = int(act_clf.predict(r_df)[0])
    p_rul = round(float(rul_reg.predict(r_df)[0]), 1)
    rule_act, part_nm, team_nm = prescribe_part_action(r_df.iloc[0])
    fin_act = rule_act if rule_act != "HEALTHY" else ACTION_LABELS[p_code]
    if fin_act != "HEALTHY" and part_nm == "None (Nominal)":
        part_nm = "Coolant_Chip_Strainer_Flush" if fin_act == "REPAIR_CALIBRATE" else "SKF_7014_Spindle_Bearing"
        team_nm = "Mechanical Team"

    is_flt = flatline_sim if m_name == st.session_state.selected_cnc else False
    if is_flt:
        tier, color_hex = "SENSOR_FAULT", "#f59e0b"
    elif p_rul <= 8.0:
        tier, color_hex = "CRITICAL", "#ef4444"
        line_avail = min(line_avail, 86.0)
    elif p_rul < hrs_to_fri and fin_act != "HEALTHY":
        tier, color_hex = "MIDWEEK_RISK", "#f97316"
        line_avail = min(line_avail, 97.5)
    elif fin_act != "HEALTHY" or p_rul < (hrs_to_fri + 48):
        tier, color_hex = "WEEKEND_BOOK", "#3b82f6"
    else:
        tier, color_hex = "HEALTHY", "#10b981"

    if not is_flt:
        if fin_act == "REPAIR_CALIBRATE":
            rep_cnt += 1
        elif fin_act == "REPLACE_SPARE":
            rpl_cnt += 1

    fleet_status[m_name] = {
        "vec": vec,
        "action": fin_act,
        "component": part_nm,
        "team": team_nm,
        "rul": p_rul,
        "tier": tier,
        "color": color_hex,
    }

live_oee = round((line_avail / 100.0) * (0.986 if line_avail >= 97.5 else 0.925) * 0.994 * 100, 1)
dyn_savings = max(0, int((line_avail - 78.1) * 28000) + (rep_cnt * 38500))

top_bar2.metric("Line OEE", f"{live_oee}%", f"Uptime: {line_avail}%")
top_bar3.metric("Repair / Calibrate", f"{rep_cnt} CNCs", "₹0 Spare Needed")
top_bar4.metric("Replace Spare", f"{rpl_cnt} CNCs", "SAP Reservation")
top_bar5.metric("Net Monthly Savings", f"₹{dyn_savings:,}", "Live Triage ROI")

# ---------------------------------------------------------
# 8. PAGE ROUTER (4 SYNCHRONIZED PRODUCT SCREENS)
# ---------------------------------------------------------

# ==================== SCREEN 1: COMMAND CENTER ====================
if st.session_state.nav_page == "🏭 Command Center & Conveyor Twin":
    # Visual Interactive CNC Conveyor Schematic (Embedded SVG Graphic)
    c1_col = fleet_status["CNC-01 (Roughing)"]["color"]
    c2_col = fleet_status["CNC-02 (Boring)"]["color"]
    c3_col = fleet_status["CNC-03 (Drilling)"]["color"]
    c4_col = fleet_status["CNC-04 (Finishing)"]["color"]

    st.markdown(
        f"""
    <div style="background:#131a26; border:1px solid #263042; border-radius:10px; padding:14px; margin-bottom:16px;">
        <div style="font-size:0.85rem; font-weight:700; color:#9ca3af; margin-bottom:8px;">LIVE SERIES CONVEYOR DIGITAL TWIN (POWER-TRAIN BLOCK LINE-04)</div>
        <svg width="100%" height="115" viewBox="0 0 920 115" xmlns="http://www.w3.org/2000/svg">
            <!-- Conveyor Belt Line -->
            <line x1="80" y1="58" x2="840" y2="58" stroke="#374151" stroke-width="8" stroke-dasharray="12 6"/>
            <!-- CNC-01 -->
            <rect x="30" y="18" width="165" height="80" rx="10" fill="#1b2434" stroke="{c1_col}" stroke-width="3"/>
            <circle cx="52" cy="36" r="6" fill="{c1_col}"/>
            <text x="66" y="41" fill="#ffffff" font-size="14" font-weight="bold">CNC-01 Roughing</text>
            <text x="45" y="64" fill="#9ca3af" font-size="12">RUL: {fleet_status['CNC-01 (Roughing)']['rul']} hrs</text>
            <text x="45" y="83" fill="{c1_col}" font-size="11" font-weight="bold">{fleet_status['CNC-01 (Roughing)']['action']}</text>
            <!-- Arrow 1->2 -->
            <polygon points="215,50 235,58 215,66" fill="#6b7280"/>
            <!-- CNC-02 -->
            <rect x="255" y="18" width="165" height="80" rx="10" fill="#1b2434" stroke="{c2_col}" stroke-width="3"/>
            <circle cx="277" cy="36" r="6" fill="{c2_col}"/>
            <text x="291" y="41" fill="#ffffff" font-size="14" font-weight="bold">CNC-02 Boring</text>
            <text x="270" y="64" fill="#9ca3af" font-size="12">RUL: {fleet_status['CNC-02 (Boring)']['rul']} hrs</text>
            <text x="270" y="83" fill="{c2_col}" font-size="11" font-weight="bold">{fleet_status['CNC-02 (Boring)']['action']}</text>
            <!-- Arrow 2->3 -->
            <polygon points="440,50 460,58 440,66" fill="#6b7280"/>
            <!-- CNC-03 -->
            <rect x="480" y="18" width="165" height="80" rx="10" fill="#1b2434" stroke="{c3_col}" stroke-width="3"/>
            <circle cx="502" cy="36" r="6" fill="{c3_col}"/>
            <text x="516" y="41" fill="#ffffff" font-size="14" font-weight="bold">CNC-03 Drilling</text>
            <text x="495" y="64" fill="#9ca3af" font-size="12">RUL: {fleet_status['CNC-03 (Drilling)']['rul']} hrs</text>
            <text x="495" y="83" fill="{c3_col}" font-size="11" font-weight="bold">{fleet_status['CNC-03 (Drilling)']['action']}</text>
            <!-- Arrow 3->4 -->
            <polygon points="665,50 685,58 665,66" fill="#6b7280"/>
            <!-- CNC-04 -->
            <rect x="705" y="18" width="175" height="80" rx="10" fill="#1b2434" stroke="{c4_col}" stroke-width="3"/>
            <circle cx="727" cy="36" r="6" fill="{c4_col}"/>
            <text x="741" y="41" fill="#ffffff" font-size="14" font-weight="bold">CNC-04 Finishing</text>
            <text x="720" y="64" fill="#9ca3af" font-size="12">RUL: {fleet_status['CNC-04 (Finishing)']['rul']} hrs</text>
            <text x="720" y="83" fill="{c4_col}" font-size="11" font-weight="bold">{fleet_status['CNC-04 (Finishing)']['action']}</text>
        </svg>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for idx, (m_name, info) in enumerate(fleet_status.items()):
        is_sel = m_name == st.session_state.selected_cnc
        with cols[idx]:
            st.markdown(
                f'<div class="cnc-card {"cnc-card-selected" if is_sel else ""}">'
                f'<b style="font-size:1.05rem;">{m_name}</b> '
                f'<span style="color:{info["color"]}; font-weight:700;">● {info["tier"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.metric("Predicted RUL", f"{info['rul']} hrs", f"{round(info['rul'] - hrs_to_fri, 1)} hrs vs Fri")
            st.write(f"**Decision:** `{info['action']}`")
            st.write(f"**Component:** `{info['component']}`")
            st.write(f"**Core Team:** {info['team']}")

            b_c1, b_c2 = st.columns(2)
            if b_c1.button("🔍 Inspect", key=f"insp_{idx}", use_container_width=True):
                st.session_state.selected_cnc = m_name
                st.session_state.nav_page = "🔬 Asset Diagnostics & 12-Sensor Fusion"
                st.rerun()

            if info["tier"] not in ["HEALTHY", "SENSOR_FAULT"]:
                win_str = "IMMEDIATE EMERGENCY" if info["tier"] == "CRITICAL" else "SATURDAY 09:00 AM (OFF)"
                if b_c2.button("📩 SAP Ticket", key=f"sap_{idx}", use_container_width=True):
                    ok = create_work_order(m_name, info["action"], info["component"], info["rul"], info["team"], win_str)
                    if ok:
                        st.toast(f"SAP-PM Notification created for {m_name}!")
                    else:
                        st.toast(f"Active ticket already open for {m_name} in SAP-PM.")
                    st.rerun()

    st.divider()
    g1, g2 = st.columns([1.6, 1.4])
    with g1:
        m_list = list(fleet_status.keys())
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p.add_trace(go.Bar(x=m_list, y=[fleet_status[m]["vec"][11] for m in m_list], name="Composite Health (0–100)", marker_color="#10b981"), secondary_y=False)
        fig_p.add_trace(go.Bar(x=m_list, y=[fleet_status[m]["vec"][0] * 8 for m in m_list], name="Vib Severity (Scaled)", marker_color="#3b82f6"), secondary_y=False)
        fig_p.add_trace(go.Scatter(x=m_list, y=[fleet_status[m]["rul"] for m in m_list], name="Predicted RUL (Hrs)", mode="lines+markers", line=dict(color="#e31837", width=4)), secondary_y=True)
        fig_p.update_layout(title="Peer-to-Peer Conveyor Health & RUL Comparison", barmode="group", height=300, template="plotly_dark", paper_bgcolor="#131a26", plot_bgcolor="#131a26", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_p, use_container_width=True)
    with g2:
        st.subheader("📊 Series Conveyor Reliability Math")
        st.dataframe(
            pd.DataFrame([
                {"Operational Metric": "Reactive Conveyor Reliability (0.94⁴)", "Live Value": "78.1% (~26 hrs lost/wk)"},
                {"Operational Metric": "Mahindra Pulse™ Mon–Fri Availability", "Live Value": f"{line_avail}%"},
                {"Operational Metric": "Live Line OEE", "Live Value": f"{live_oee}%"},
                {"Operational Metric": "Repair-Before-Replace Savings", "Live Value": f"₹{rep_cnt * 38500:,} / mo"},
                {"Operational Metric": "Total Monthly Value Protected", "Live Value": f"₹{dyn_savings:,} / mo"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

# ==================== SCREEN 2: ASSET DIAGNOSTICS ====================
elif st.session_state.nav_page == "🔬 Asset Diagnostics & 12-Sensor Fusion":
    sel_m = st.session_state.selected_cnc
    cur = fleet_status[sel_m]
    v12 = cur["vec"]
    st.subheader(f"🔬 Deep Asset Diagnostics & Sensor Fusion — {sel_m}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Composite Health Index", f"{v12[11]} / 100", cur["action"])
    k2.metric("Vib Crest Severity", f"{v12[0]}", "Replace Limit: >7.5")
    k3.metric("Coolant Circuit Health", f"{v12[4]}", "Flush Limit: <0.45")
    k4.metric("Axis Backlash Index", f"{v12[5]}", "Calib >14.5 | Replace >21.0")

    m_code = sel_m.split()[0]
    sub_w = active_win_df[active_win_df["Machine_ID"] == m_code].tail(120)
    fig_t = make_subplots(specs=[[{"secondary_y": True}]])
    fig_t.add_trace(go.Scatter(x=sub_w["Window_15m"], y=sub_w["Vib_Crest_Severity_Idx"], name="Vib Crest Severity Idx", line=dict(color="#3b82f6", width=2.5)), secondary_y=False)
    fig_t.add_trace(go.Scatter(x=sub_w["Window_15m"], y=sub_w["Coolant_Circuit_Health"], name="Coolant Circuit Health", line=dict(color="#10b981", width=2.5)), secondary_y=True)
    fig_t.add_trace(go.Scatter(x=sub_w["Window_15m"], y=sub_w["Axis_Backlash_Servo_Idx"], name="Axis Backlash Servo Idx", line=dict(color="#f97316", width=2.5)), secondary_y=False)
    fig_t.update_layout(title=f"15-Minute Rolling Telemetry Trajectory ({sel_m})", height=320, template="plotly_dark", paper_bgcolor="#131a26", plot_bgcolor="#131a26", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_t, use_container_width=True)

    c_d1, c_d2 = st.columns([1.3, 1.2])
    with c_d1:
        st.markdown("**12 High-Value Fused Decision Parameters (44 Raw Sensors ➔ 12 Indicators)**")
        st.dataframe(
            pd.DataFrame({
                "Engineered Indicator": ENGINEERED_12_COLS,
                "Live Value": v12,
                "Fused Raw Sensors": [
                    "Vib_X/Y/Z + Peak_G + Acoustic_dB",
                    "Front/Rear Brg Temp + Ambient + RPM",
                    "Current_A + Voltage_L1/L2/L3 + PF",
                    "Torque + RPM + Feed + Depth_of_Cut",
                    "Coolant Press + Flow + Filter_dP",
                    "X/Y/Z Servo Amps + Following_Error",
                    "Lube Press + Flow + Slideway_Temp",
                    "Pneumatic + Chuck_Press + RPM",
                    "DC_Bus_V + Ripple_V + IGBT_Temp",
                    "Hyd_Press + Motor_Amps + Accumulator",
                    "Vib_X/Z + Acoustic_dB + RPM",
                    "Multi-Domain Survival Fusion",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )
    with c_d2:
        st.markdown("**⬛ 60-Second Digital Black-Box Pre-Trip Freeze**")
        sec_g = np.arange(-60, 1, 10)
        st.dataframe(
            pd.DataFrame({
                "Sec_Before_Now": sec_g,
                "Vib_Crest": np.round(np.linspace(max(1.2, v12[0] - 0.8), v12[0], len(sec_g)), 2),
                "Coolant_Idx": np.round(np.linspace(min(0.95, v12[4] + 0.08), v12[4], len(sec_g)), 2),
                "Action": cur["action"],
                "Target_Part": cur["component"],
            }),
            use_container_width=True,
            hide_index=True,
        )

# ==================== SCREEN 3: SAP-PM WORK ORDERS ====================
elif st.session_state.nav_page == "🔧 SAP-PM Work Orders & Dispatch":
    st.subheader("🔧 SAP-PM Closed-Loop Maintenance Lifecycle (Repair/Calibrate vs. Replace Spare)")
    wo_df = fetch_work_orders()
    st.dataframe(wo_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Stage 2: Planner Approval & Store Check**")
        pend = wo_df[wo_df["status"] == "AUTO_NOTIFICATION"]
        if not pend.empty:
            s_id = st.selectbox("Pending Notification ID:", pend["id"].tolist())
            if st.button("✅ Approve & Reserve Store Spare", use_container_width=True):
                approve_notification(s_id)
                st.success(f"Ticket #{s_id} approved!")
                st.rerun()
        else:
            st.info("No pending notifications.")
    with c2:
        st.markdown("**Stage 4: Weekend ISO 14224 Sign-Off**")
        open_w = wo_df[wo_df["status"] != "CLOSED"]
        if not open_w.empty:
            cl_id = st.selectbox("Open Ticket ID:", open_w["id"].tolist())
            iso_c = st.selectbox(
                "ISO 14224 Root Cause:",
                [
                    "ISO-14224: ME-02 (Spindle Bearing Replaced)",
                    "ISO-14224: ME-08 (Coolant Strainer Flushed)",
                    "ISO-14224: EL-09 (Backlash Offset Calibrated)",
                    "ISO-14224: FA-00 (False Alarm — Update Threshold)",
                ],
            )
            if st.button("🔒 Close Ticket & Log Feedback", use_container_width=True):
                close_work_order(cl_id, iso_c)
                st.success(f"Ticket #{cl_id} closed!")
                st.rerun()
        else:
            st.info("All tickets closed.")
    with c3:
        st.markdown("**Friday Shift Handover Dispatch**")
        st.download_button(
            "⬇️ Export Sat/Sun Dispatch Sheet (CSV)",
            wo_df.to_csv(index=False),
            "mahindra_pulse_dispatch.csv",
            "text/csv",
            use_container_width=True,
        )

# ==================== SCREEN 4: MULTI-LAKH DATA & ML AUDIT ====================
else:
    st.subheader("🗄️ High-Frequency Plant Data Engine & Chronological ML Verification")
    d_choice = st.select_slider(
        "Active Minute-by-Minute Plant Corpus Size:",
        options=[5, 10, 22],
        value=st.session_state.plant_days,
        format_func=lambda d: f"{d} Working Days ({d*1440*4:,} Minute-by-Minute Rows × 63 Cols)",
    )
    if d_choice != st.session_state.plant_days:
        st.session_state.plant_days = d_choice
        st.session_state.custom_raw_df = None
        st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chronological Test Accuracy", f"{val_metrics['accuracy']}%", f"n={val_metrics['test_windows']:,} future windows")
    m2.metric("Weighted Precision", f"{val_metrics['precision']}%", "Zero Data Leakage")
    m3.metric("Weighted Recall", f"{val_metrics['recall']}%", "Fault Capture Rate")
    m4.metric("RUL Error (MAE)", f"±{val_metrics['rul_mae']} hrs", "15-Min Rolling Windows")

    u1, u2 = st.columns([1, 2])
    u1.download_button(
        "⬇️ Download Sample 1,000-Row × 63-Col CSV",
        active_raw_df.head(1000).to_csv(index=False),
        "mahindra_pulse_63col_sample.csv",
        "text/csv",
        use_container_width=True,
    )
    up_csv = u2.file_uploader("Upload Company Plant CSV (44 Raw PLC Cols or 63 Full Cols):", type=["csv"])
    if up_csv is not None:
        df_up = pd.read_csv(up_csv)
        if all(c in df_up.columns for c in ENGINEERED_12_COLS) or all(c in df_up.columns for c in RAW_44_COLS):
            if not all(c in df_up.columns for c in ENGINEERED_12_COLS):
                df_up = compute_12_engineered_features(df_up)
            if "Action_Code" not in df_up.columns:
                rev = {"HEALTHY": 0, "REPAIR_CALIBRATE": 1, "REPLACE_SPARE": 2}
                df_up["Action_Code"] = [rev[prescribe_part_action(r)[0]] for _, r in df_up.iterrows()]
            if "RUL_Hours" not in df_up.columns:
                df_up["RUL_Hours"] = np.round((df_up["Composite_OEE_Risk_Score"] / 100.0) * 140.0, 1)
            if st.button("🚀 Ingest Plant CSV & Retrain Models Chronologically"):
                st.session_state.custom_raw_df = df_up
                st.rerun()
        else:
            st.error("CSV must contain the 44 raw PLC sensor columns or 12 engineered columns.")

    ch1, ch2 = st.columns(2)
    with ch1:
        fig_imp = go.Figure(go.Bar(x=feat_imp_df["Importance"], y=feat_imp_df["Parameter"], orientation="h", marker_color="#e31837"))
        fig_imp.update_layout(title="Top Driving Engineered Parameters", height=320, template="plotly_dark", paper_bgcolor="#131a26", plot_bgcolor="#131a26")
        st.plotly_chart(fig_imp, use_container_width=True)
    with ch2:
        cm = val_metrics["cm"]
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred Healthy", "Pred Repair", "Pred Replace"], y=["Actual Healthy", "Actual Repair", "Actual Replace"], text=cm, texttemplate="%{text}", colorscale="Reds"))
        fig_cm.update_layout(title="Out-of-Time Chronological Confusion Matrix", height=320, template="plotly_dark", paper_bgcolor="#131a26", plot_bgcolor="#131a26")
        st.plotly_chart(fig_cm, use_container_width=True)
