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
    page_title="Mahindra CNC Line — High-Frequency IIoT & PdM Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 1. PERSISTENT SQLITE CMMS (REPAIR/CALIBRATE VS REPLACE SPARE)
# ---------------------------------------------------------
DB_FILE = "mahindra_hf_cmms_v3.db"


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
                "NOTIF-20411",
                "2026-11-19 11:15",
                "CNC #2 (Boring)",
                "REPLACE_SPARE",
                "SKF_7014_Spindle_Bearing",
                54.0,
                "Mechanical Team",
                "RESERVED (Store Bin B-14)",
                42000,
                "SATURDAY 09:00 AM (OFF)",
                "APPROVED_WO",
                "Pending Sat Execution",
            ),
            (
                "NOTIF-20408",
                "2026-11-18 16:40",
                "CNC #3 (Drilling)",
                "REPAIR_CALIBRATE",
                "Coolant_Chip_Strainer_Flush",
                66.5,
                "Mechanical Team",
                "NO SPARE NEEDED (Clean/Flush)",
                1500,
                "SATURDAY 11:00 AM (OFF)",
                "AUTO_NOTIFICATION",
                "Awaiting Planner Approval",
            ),
            (
                "NOTIF-20395",
                "2026-11-15 10:30",
                "CNC #4 (Finishing)",
                "REPAIR_CALIBRATE",
                "Z_Axis_PLC_Backlash_Offset",
                48.0,
                "Electrical Team",
                "NO SPARE NEEDED (PLC Pitch Calib)",
                1200,
                "SUNDAY 10:00 AM (OFF)",
                "CLOSED",
                "ISO-14224: EL-09 (Backlash Calibrated)",
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
        notif_id = f"NOTIF-{np.random.randint(20500, 29999)}"
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
    conn.close()


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
# 2. 12 HIGH-VALUE ENGINEERED PARAMETERS (SENSOR FUSION ENGINE)
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

ACTION_LABELS = {
    0: "HEALTHY",
    1: "REPAIR_CALIBRATE",
    2: "REPLACE_SPARE",
}


def compute_12_engineered_features(df):
    """Fuses 44 raw PLC/IoT sensor columns into the 12 High-Value Decision Columns."""
    out = df.copy()
    rms_vib = np.sqrt(
        out["Vib_X_mm_s"] ** 2 + out["Vib_Y_mm_s"] ** 2 + out["Vib_Z_mm_s"] ** 2
    )
    out["Vib_Crest_Severity_Idx"] = np.round(
        rms_vib * (out["Peak_Accel_G"] / np.maximum(rms_vib, 0.5)), 2
    )
    temp_gap = np.maximum(0.0, out["Front_Brg_Temp_C"] - out["Ambient_Temp_C"])
    out["Thermal_Drift_Slope_Hr"] = np.round(
        (temp_gap / 8.5) * (1.0 + np.maximum(0.0, out["Vib_Crest_Severity_Idx"] - 2.0) * 0.18), 2
    )
    v_avg = (out["Voltage_L1_V"] + out["Voltage_L2_V"] + out["Voltage_L3_V"]) / 3.0
    out["True_Active_Power_kW"] = np.round(
        (1.732 * v_avg * out["Spindle_Current_A"] * out["Power_Factor"]) / 1000.0, 2
    )
    out["Specific_Cutting_Energy"] = np.round(
        (out["Cutting_Torque_Nm"] * (out["Spindle_RPM"] * 2 * np.pi / 60.0))
        / np.maximum(out["Feed_Rate_mm_min"] * out["Depth_of_Cut_mm"], 1.0),
        2,
    )
    out["Coolant_Circuit_Health"] = np.round(
        np.clip(
            (out["Coolant_Flow_LPM"] / 24.5) * (1.0 - out["Filter_Diff_Press_Bar"] / 2.2),
            0.05,
            1.0,
        ),
        2,
    )
    out["Axis_Backlash_Servo_Idx"] = np.round(
        out["Following_Error_um"] * (out["Z_Servo_Amps"] / 5.2), 2
    )
    out["Lube_Film_Integrity_Idx"] = np.round(
        (out["Lube_Oil_Press_Bar"] * out["Lube_Flow_CC_Min"])
        / np.maximum(out["Slideway_Temp_C"], 10.0),
        2,
    )
    out["Chuck_Clamping_Stability"] = np.round(
        (out["Chuck_Clamp_Press_Bar"] / np.maximum(out["Pneumatic_Line_Bar"], 1.0))
        * ((out["Spindle_RPM"] / 1650.0) ** 2),
        2,
    )
    out["Drive_Harmonic_Health"] = np.round(
        (out["DC_Bus_Ripple_V"] / np.maximum(out["Drive_DC_Bus_V"], 100.0))
        * (out["IGBT_Heatsink_Temp_C"] / 60.0)
        * 100.0,
        2,
    )
    out["Hydraulic_Power_Efficiency"] = np.round(
        (out["Hyd_Pump_Press_Bar"] * 18.0)
        / np.maximum(out["Hyd_Motor_Amps"] * 415.0, 1.0),
        2,
    )
    out["Spindle_Runout_Chatter"] = np.round(
        (out["Acoustic_dB"] * np.abs(out["Vib_X_mm_s"] - out["Vib_Z_mm_s"]))
        / np.maximum(out["Spindle_RPM"] / 1000.0, 0.5),
        2,
    )
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
    """Maps fused parameter signatures to exact Repair/Calibrate vs Replace Spare prescriptions."""
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


# ---------------------------------------------------------
# 3. HIGH-FREQUENCY MINUTE-BY-MINUTE DATA GENERATOR (63 COLS)
# ---------------------------------------------------------
@st.cache_resource
def generate_plant_corpus(num_days=10):
    """Generates minute-by-minute 63-column telemetry across 4 CNCs (float32 memory optimized)."""
    np.random.seed(42)
    all_days = pd.date_range("2026-10-05", periods=num_days, freq="B")
    minute_index = pd.DatetimeIndex(
        np.concatenate([
            pd.date_range(d, d + pd.Timedelta(hours=23, minutes=59), freq="1min")
            for d in all_days
        ])
    )
    n_m = len(minute_index)
    machines = ["CNC_01", "CNC_02", "CNC_03", "CNC_04"]
    frames = []

    for m_idx, m_id in enumerate(machines):
        day_num = np.arange(n_m) / 1440.0
        bearing_wave = np.where(
            (m_id == "CNC_02") & (day_num >= 4.0) & (day_num <= 8.5),
            ((day_num - 4.0) / 4.5) ** 1.3 * 2.1,
            0.06 * np.sin(day_num * 1.4) ** 2,
        )
        tool_min = (np.arange(n_m) % (210 if m_id == "CNC_02" else 260)) * 0.95
        tool_factor = np.clip(tool_min / 200.0, 0, 1.3)
        clog_wave = np.where(
            (m_id == "CNC_03") & (day_num >= 2.0) & (day_num <= 5.0),
            ((day_num - 2.0) / 3.0) * 1.75,
            0.08 * (day_num % 2.5) / 2.5,
        )
        drive_wave = np.where(
            (m_id == "CNC_03") & (day_num >= 7.0) & (day_num <= 9.8),
            ((day_num - 7.0) / 2.8) * 2.0,
            0.05,
        )
        servo_cal = np.where(
            (m_id == "CNC_04") & (day_num >= 3.0) & (day_num <= 6.0),
            ((day_num - 3.0) / 3.0) * 1.45,
            0.0,
        )
        servo_rep = np.where(
            (m_id == "CNC_04") & (day_num >= 7.5) & (day_num <= 9.8),
            ((day_num - 7.5) / 2.3) * 2.1,
            0.0,
        )
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

        # Assign ground-truth Action_Decision & RUL_Hours
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
        df_m["RUL_Hours"] = np.round(
            np.clip((df_m["Composite_OEE_Risk_Score"] / 100.0) * 140.0, 2.0, 140.0), 1
        ).astype(np.float32)
        frames.append(df_m)

    master = pd.concat(frames, ignore_index=True)
    return master


def compress_to_15min_windows(raw_df):
    """Compresses minute-by-minute logs into 15-min rolling statistical windows (anti-noise)."""
    df = raw_df.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Window_15m"] = df["Timestamp"].dt.floor("15min")
    else:
        df["Window_15m"] = np.arange(len(df)) // 15

    if "Machine_ID" not in df.columns:
        df["Machine_ID"] = "CNC_01"

    agg_dict = {col: "mean" for col in ENGINEERED_12_COLS if col in df.columns}
    if "Tool_Run_Min" in df.columns:
        agg_dict["Tool_Run_Min"] = "max"
    if "Action_Code" in df.columns:
        agg_dict["Action_Code"] = "max"
    if "RUL_Hours" in df.columns:
        agg_dict["RUL_Hours"] = "min"

    win_df = df.groupby(["Machine_ID", "Window_15m"], as_index=False).agg(agg_dict)
    win_df = win_df.sort_values(["Window_15m", "Machine_ID"]).reset_index(drop=True)
    return win_df


def train_chronological_models(win_df):
    """Strict Chronological Split (First 75% time windows = Train, Last 25% = Unseen Future Test)."""
    split_idx = int(len(win_df) * 0.75)
    train_df = win_df.iloc[:split_idx]
    test_df = win_df.iloc[split_idx:]

    X_tr, y_act_tr, y_rul_tr = (
        train_df[ENGINEERED_12_COLS],
        train_df["Action_Code"],
        train_df["RUL_Hours"],
    )
    X_te, y_act_te, y_rul_te = (
        test_df[ENGINEERED_12_COLS],
        test_df["Action_Code"],
        test_df["RUL_Hours"],
    )

    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_tr, y_act_tr)
    reg = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ).fit(X_tr, y_rul_tr)

    pred_act = clf.predict(X_te)
    pred_rul = reg.predict(X_te)

    metrics = {
        "accuracy": round(accuracy_score(y_act_te, pred_act) * 100, 2),
        "precision": round(
            precision_score(y_act_te, pred_act, average="weighted", zero_division=0) * 100,
            2,
        ),
        "recall": round(
            recall_score(y_act_te, pred_act, average="weighted", zero_division=0) * 100,
            2,
        ),
        "rul_mae": round(mean_absolute_error(y_rul_te, pred_rul), 2),
        "train_windows": len(train_df),
        "test_windows": len(test_df),
        "cm": confusion_matrix(y_act_te, pred_act, labels=[0, 1, 2]),
    }
    imp = pd.DataFrame(
        {"Engineered_Parameter": ENGINEERED_12_COLS, "Importance": clf.feature_importances_}
    ).sort_values("Importance", ascending=True)
    return clf, reg, metrics, imp


# ---------------------------------------------------------
# 4. INITIALIZE CORPUS & SESSION STATE
# ---------------------------------------------------------
if "plant_days" not in st.session_state:
    st.session_state.plant_days = 10  # 10 working days = 57,600 minute-by-minute rows

base_raw_df = generate_plant_corpus(st.session_state.plant_days)
if "custom_raw_df" in st.session_state and st.session_state.custom_raw_df is not None:
    active_raw_df = st.session_state.custom_raw_df
else:
    active_raw_df = base_raw_df

active_win_df = compress_to_15min_windows(active_raw_df)
act_clf, rul_reg, val_metrics, feat_imp_df = train_chronological_models(active_win_df)

# Live 4-CNC Conveyor State (12 High-Value Engineered Parameters per station)
HEALTHY_CNC = [1.85, 0.55, 7.2, 1.48, 0.91, 4.2, 1.05, 0.92, 0.65, 1.35, 1.15, 92.5]
DEFAULT_FLEET = {
    "CNC #1 (Roughing)": list(HEALTHY_CNC),
    "CNC #2 (Boring)": [8.40, 2.85, 12.4, 2.62, 0.84, 9.5, 0.98, 0.90, 0.85, 1.32, 3.40, 44.0],
    "CNC #3 (Drilling)": [2.45, 1.85, 7.8, 1.60, 0.31, 5.2, 1.01, 0.91, 0.72, 1.34, 1.45, 66.5],
    "CNC #4 (Finishing)": [2.10, 1.15, 7.5, 1.55, 0.89, 16.4, 0.99, 0.92, 0.68, 1.35, 1.30, 71.0],
}

if "fleet_12_states" not in st.session_state:
    st.session_state.fleet_12_states = {k: list(v) for k, v in DEFAULT_FLEET.items()}

# ---------------------------------------------------------
# 5. SIDEBAR: DATA SCALE, PRESETS & 12-PARAMETER TUNER
# ---------------------------------------------------------
st.sidebar.header("🗄️ Plant Data Scale (Minute-by-Minute)")
days_choice = st.sidebar.select_slider(
    "Historical Telemetry Window:",
    options=[5, 10, 22],
    value=st.session_state.plant_days,
    format_func=lambda d: f"{d} Days ({d*1440*4:,} rows)",
)
if days_choice != st.session_state.plant_days:
    st.session_state.plant_days = days_choice
    st.session_state.custom_raw_df = None
    st.rerun()

st.sidebar.divider()
st.sidebar.header("⚡ 1-Click Floor Fault Scenarios")
sc1, sc2 = st.sidebar.columns(2)
if sc1.button("🟢 All Healthy"):
    st.session_state.fleet_12_states = {m: list(HEALTHY_CNC) for m in DEFAULT_FLEET}
    st.rerun()
if sc2.button("🟡 Clog Filter (#3)"):
    st.session_state.fleet_12_states["CNC #3 (Drilling)"][4] = 0.28  # Coolant_Circuit_Health drop -> REPAIR
    st.rerun()

sc3, sc4 = st.sidebar.columns(2)
if sc3.button("🟠 Backlash (#4)"):
    st.session_state.fleet_12_states["CNC #4 (Finishing)"][5] = 16.8  # Axis_Backlash_Servo_Idx -> CALIBRATE
    st.rerun()
if sc4.button("🔴 Bearing (#2)"):
    st.session_state.fleet_12_states["CNC #2 (Boring)"][0] = 8.9  # Vib_Crest -> REPLACE
    st.session_state.fleet_12_states["CNC #2 (Boring)"][1] = 3.1  # Thermal_Drift -> REPLACE
    st.rerun()

st.sidebar.divider()
st.sidebar.header("🎛️ Live Fused Parameter Controller")
active_cnc = st.sidebar.selectbox(
    "Select CNC Station to Tune:", list(st.session_state.fleet_12_states.keys()), index=1
)
cur_vec = st.session_state.fleet_12_states[active_cnc]

p_vib = st.sidebar.slider("1. Vib_Crest_Severity_Idx", 0.8, 12.0, float(cur_vec[0]), 0.1, key=f"p0_{active_cnc}")
p_thm = st.sidebar.slider("2. Thermal_Drift_Slope_Hr", 0.2, 5.0, float(cur_vec[1]), 0.1, key=f"p1_{active_cnc}")
p_pwr = st.sidebar.slider("3. True_Active_Power_kW", 4.0, 16.0, float(cur_vec[2]), 0.2, key=f"p2_{active_cnc}")
p_cut = st.sidebar.slider("4. Specific_Cutting_Energy", 1.0, 3.5, float(cur_vec[3]), 0.05, key=f"p3_{active_cnc}")
p_col = st.sidebar.slider("5. Coolant_Circuit_Health", 0.10, 1.00, float(cur_vec[4]), 0.02, key=f"p4_{active_cnc}")
p_bkl = st.sidebar.slider("6. Axis_Backlash_Servo_Idx", 2.0, 26.0, float(cur_vec[5]), 0.2, key=f"p5_{active_cnc}")
p_drv = st.sidebar.slider("9. Drive_Harmonic_Health", 0.4, 4.0, float(cur_vec[8]), 0.05, key=f"p8_{active_cnc}")

sensor_flatline_sim = st.sidebar.checkbox(
    f"Simulate PLC Tag Flatline on {active_cnc}", value=False, key=f"flt_{active_cnc}"
)

# Recompute Composite OEE Risk Score live from tuned parameters
comp_oee = round(
    float(
        np.clip(
            100.0
            - (p_vib * 4.0)
            - (p_thm * 4.8)
            - ((1.0 - p_col) * 28.0)
            - (max(0.0, p_bkl - 6.0) * 1.6)
            - (max(0.0, p_drv - 1.2) * 8.5),
            5.0,
            99.5,
        )
    ),
    1,
)

updated_vec = [
    p_vib, p_thm, p_pwr, p_cut, p_col, p_bkl,
    cur_vec[6], cur_vec[7], p_drv, cur_vec[9], cur_vec[10], comp_oee,
]
st.session_state.fleet_12_states[active_cnc] = updated_vec

# ---------------------------------------------------------
# 6. LIVE ML INFERENCE & DYNAMIC PLANT KPI ENGINE
# ---------------------------------------------------------
st.title("🏭 Mahindra CNC Line — High-Frequency IIoT & Repair/Replace Engine")
st.caption(
    f"**Master Telemetry Stream:** {len(active_raw_df):,} Minute-by-Minute Rows ×"
    f" {active_raw_df.shape[1]} Cols | **15-Min Rolling Windows:** {len(active_win_df):,} |"
    f" **Chronological Test Accuracy:** {val_metrics['accuracy']}% (MAE: ±{val_metrics['rul_mae']} hrs)"
)

top1, top2, top3, top4, top5 = st.columns([2.2, 1, 1, 1, 1])
with top1:
    shift_opt = st.selectbox(
        "Active Mon–Fri Production Clock:",
        [
            ("Monday 10:00 AM", 110),
            ("Wednesday 02:00 PM", 58),
            ("Thursday 06:00 PM", 30),
            ("Friday 04:00 PM", 8),
        ],
        index=1,
        format_func=lambda x: f"{x[0]} ({x[1]} hrs left till Fri Midnight)",
    )
hrs_to_fri = shift_opt[1]

fleet_eval = {}
line_avail = 100.0
repair_count = 0
replace_count = 0
critical_stops = 0

for m_name, vec in st.session_state.fleet_12_states.items():
    row_df = pd.DataFrame([vec], columns=ENGINEERED_12_COLS)
    pred_act_code = int(act_clf.predict(row_df)[0])
    pred_rul = round(float(rul_reg.predict(row_df)[0]), 1)

    rule_act, part_name, team_name = prescribe_part_action(row_df.iloc[0])
    final_act = rule_act if rule_act != "HEALTHY" else ACTION_LABELS[pred_act_code]
    if final_act != "HEALTHY" and part_name == "None (Nominal)":
        part_name = (
            "Coolant_Chip_Strainer_Flush"
            if final_act == "REPAIR_CALIBRATE"
            else "SKF_7014_Spindle_Bearing"
        )
        team_name = "Mechanical Team"

    is_flatlined = sensor_flatline_sim if m_name == active_cnc else False

    if is_flatlined:
        status_tier = "SENSOR_FAULT"
    elif pred_rul <= 8.0:
        status_tier = "CRITICAL"
        line_avail = min(line_avail, 86.0)
        critical_stops += 1
    elif pred_rul < hrs_to_fri and final_act != "HEALTHY":
        status_tier = "MIDWEEK_RISK"
        line_avail = min(line_avail, 97.5)
    elif final_act != "HEALTHY" or pred_rul < (hrs_to_fri + 48):
        status_tier = "WEEKEND_BOOK"
    else:
        status_tier = "HEALTHY"

    if not is_flatlined:
        if final_act == "REPAIR_CALIBRATE":
            repair_count += 1
        elif final_act == "REPLACE_SPARE":
            replace_count += 1

    fleet_eval[m_name] = {
        "vec": vec,
        "action": final_act,
        "component": part_name,
        "team": team_name,
        "rul": pred_rul,
        "tier": status_tier,
        "flatline": is_flatlined,
    }

perf_rate = 98.6 if line_avail >= 97.5 else 92.5
qual_rate = 99.4
live_oee = round((line_avail / 100.0) * (perf_rate / 100.0) * (qual_rate / 100.0) * 100, 1)
# Dynamic monthly savings calculated directly from avoided weekday stops + repair-vs-replace triage
avoided_halt_savings = int((line_avail - 78.1) * 28000)
smart_triage_savings = repair_count * 38500  # Saving spare replacement by repairing/calibrating early
total_dynamic_savings = max(0, avoided_halt_savings + smart_triage_savings)

top2.metric("Live Line OEE", f"{live_oee}%", f"Uptime: {line_avail}%")
top3.metric("Repair / Calibrate Queue", f"{repair_count} CNCs", "₹0 Spare Cost")
top4.metric("Replace Spare Queue", f"{replace_count} CNCs", "Reserve in SAP")
top5.metric("Dynamic Monthly Savings", f"₹{total_dynamic_savings:,}", "Live Triage + Uptime")

# ---------------------------------------------------------
# 7. 4 INTERACTIVE WORKFLOW MODULES
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 1. Shop-Floor Command & Repair/Replace Triage",
    "🔬 2. 12 Fused Parameters & Black-Box Freeze",
    "🔧 3. SAP-PM Closed-Loop CMMS",
    "🗄️ 4. Multi-Lakh CSV Ingestion & Chronological Audit",
])

# ==================== TAB 1: COMMAND CENTER ====================
with tab1:
    st.subheader("Real-Time Conveyor Line (`CNC #1 ➔ #2 ➔ #3 ➔ #4`) — Part-Level Prescriptions")
    cols = st.columns(4)
    for idx, (m_name, info) in enumerate(fleet_eval.items()):
        tier = info["tier"]
        badge = " *(Inspecting)*" if m_name == active_cnc else ""
        with cols[idx]:
            if tier == "SENSOR_FAULT":
                st.warning(f"### {m_name}{badge}\n**⚠️ PLC SENSOR FLATLINE**")
                st.metric("Composite Health", "TAG FROZEN", "Gate Blocked False WO")
                st.write("**Diagnosis:** Zero-variance sensor tag detected (`std=0`).")
                st.write("**Action:** Inspect M12 sensor cable; do NOT replace mechanical parts.")
            else:
                if tier == "CRITICAL":
                    st.error(f"### {m_name}{badge}\n**🔴 CRITICAL TRIP**")
                    win_label = "IMMEDIATE EMERGENCY"
                elif tier == "MIDWEEK_RISK":
                    st.warning(f"### {m_name}{badge}\n**🟠 MID-WEEK RISK**")
                    win_label = "SHIFT CHANGE / THROTTLE"
                elif tier == "WEEKEND_BOOK":
                    st.info(f"### {m_name}{badge}\n**🟡 BOOK SAT/SUN**")
                    win_label = "SATURDAY 09:00 AM (OFF)"
                else:
                    st.success(f"### {m_name}{badge}\n**🟢 HEALTHY**")
                    win_label = None

                st.metric(
                    "Predicted RUL",
                    f"{info['rul']} hrs",
                    delta=f"{round(info['rul'] - hrs_to_fri, 1)} hrs vs Fri",
                )
                st.write(f"**Decision:** `{info['action']}`")
                st.write(f"**Target Part:** `{info['component']}`")
                st.write(f"**Core Team:** {info['team']}")

                if win_label and st.button(f"Log SAP Notification: {m_name}", key=f"btn_sap_{idx}"):
                    create_work_order(
                        m_name,
                        info["action"],
                        info["component"],
                        info["rul"],
                        info["team"],
                        win_label,
                    )
                    st.toast(f"SAP Notification created for {m_name} ({info['action']})!")
                    st.rerun()

    st.divider()
    c_b1, c_b2 = st.columns([1.6, 1.4])
    with c_b1:
        st.subheader("⚖️ Peer-to-Peer Fleet Health Comparison (12-Parameter Fusion)")
        m_list = list(fleet_eval.keys())
        fig_peer = make_subplots(specs=[[{"secondary_y": True}]])
        fig_peer.add_trace(
            go.Bar(
                x=m_list,
                y=[fleet_eval[m]["vec"][11] for m in m_list],
                name="Composite Health Score (0–100)",
                marker_color="#2ca02c",
            ),
            secondary_y=False,
        )
        fig_peer.add_trace(
            go.Bar(
                x=m_list,
                y=[fleet_eval[m]["vec"][0] * 8 for m in m_list],
                name="Vib Crest Severity (Scaled)",
                marker_color="#1f77b4",
            ),
            secondary_y=False,
        )
        fig_peer.add_trace(
            go.Scatter(
                x=m_list,
                y=[fleet_eval[m]["rul"] for m in m_list],
                name="Predicted RUL (Hrs)",
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=4),
            ),
            secondary_y=True,
        )
        fig_peer.update_layout(
            barmode="group", height=300, legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_peer, use_container_width=True)

    with c_b2:
        st.subheader("📈 Dynamic Reliability & Savings Breakdown")
        st.dataframe(
            pd.DataFrame([
                {"Metric": "Series Conveyor Reliability (Reactive 94%^4)", "Live Value": "78.1% (~26 hrs lost/wk)"},
                {"Metric": "Current Mon–Fri Conveyor Availability", "Live Value": f"{line_avail}%"},
                {"Metric": "Current Line OEE", "Live Value": f"{live_oee}%"},
                {"Metric": "Avoided Breakdown Loss (Dynamic)", "Live Value": f"₹{avoided_halt_savings:,} / mo"},
                {"Metric": "Repair-Before-Replace Spare Savings", "Live Value": f"₹{smart_triage_savings:,} / mo"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

# ==================== TAB 2: 12 FUSED PARAMETERS & BLACK-BOX ====================
with tab2:
    st.subheader(f"🔬 12 High-Value Fused Decision Parameters — {active_cnc}")
    cur_f = fleet_eval[active_cnc]
    v12 = cur_f["vec"]

    k_a, k_b, k_c, k_d = st.columns(4)
    k_a.metric("Composite Health Index", f"{v12[11]} / 100", cur_f["action"])
    k_b.metric("Vib Crest Severity", f"{v12[0]}", "Replace Threshold: >7.5")
    k_c.metric("Coolant Circuit Health", f"{v12[4]}", "Flush Threshold: <0.45")
    k_d.metric("Axis Backlash Index", f"{v12[5]}", "Calib >14.5 | Replace >21.0")

    # Show all 12 engineered values in a clean interactive table
    fused_table = pd.DataFrame({
        "High-Value Engineered Parameter": ENGINEERED_12_COLS,
        "Live Fused Value": v12,
        "Raw Sensors Fused (4–5 Inputs)": [
            "Vib_X, Vib_Y, Vib_Z, Peak_Accel_G, Acoustic_dB",
            "Front_Brg_Temp, Rear_Brg_Temp, Ambient_Temp, Spindle_RPM",
            "Spindle_Current_A, Voltage_L1/L2/L3, Power_Factor",
            "Cutting_Torque_Nm, Spindle_RPM, Feed_Rate, Depth_of_Cut",
            "Coolant_Press, Coolant_Flow, Filter_Diff_Press",
            "X/Y/Z_Servo_Amps, Following_Error_um, Actual_Cycle_Sec",
            "Lube_Oil_Press, Lube_Flow, Slideway_Temp",
            "Pneumatic_Line_Bar, Chuck_Clamp_Press, Spindle_RPM",
            "Drive_DC_Bus_V, DC_Bus_Ripple_V, IGBT_Heatsink_Temp",
            "Hyd_Pump_Press, Hyd_Motor_Amps, Accumulator_Press",
            "Vib_X, Vib_Z, Acoustic_dB, Spindle_RPM",
            "Multi-Domain Weighted Survival Fusion",
        ],
    })
    st.dataframe(fused_table, use_container_width=True, hide_index=True)

    # Interactive Multi-Window Trend Chart from the Active Dataset
    st.subheader(f"📊 Historical 15-Minute Rolling Degradation Trend ({active_cnc.split()[0].replace('#', '0')})")
    m_code = active_cnc.split()[0].replace("#", "0")
    sub_win = active_win_df[active_win_df["Machine_ID"] == m_code].tail(120)

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(
            x=sub_win["Window_15m"],
            y=sub_win["Vib_Crest_Severity_Idx"],
            name="Vib Crest Severity Idx",
            line=dict(color="#1f77b4", width=2.5),
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=sub_win["Window_15m"],
            y=sub_win["Coolant_Circuit_Health"],
            name="Coolant Circuit Health",
            line=dict(color="#2ca02c", width=2.5),
        ),
        secondary_y=True,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=sub_win["Window_15m"],
            y=sub_win["Axis_Backlash_Servo_Idx"],
            name="Axis Backlash Servo Idx",
            line=dict(color="#ff7f0e", width=2.5),
        ),
        secondary_y=False,
    )
    fig_trend.update_layout(height=320, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("⬛ 60-Second Digital Black-Box Pre-Trip Freeze")
    sec_grid = np.arange(-60, 1, 5)
    bb_freeze = pd.DataFrame({
        "Seconds_Before_Now": sec_grid,
        "Vib_Crest_Severity_Idx": np.round(np.linspace(max(1.2, v12[0] - 0.8), v12[0], len(sec_grid)), 2),
        "Thermal_Drift_Slope_Hr": np.round(np.linspace(max(0.4, v12[1] - 0.3), v12[1], len(sec_grid)), 2),
        "Coolant_Circuit_Health": np.round(np.linspace(min(0.95, v12[4] + 0.08), v12[4], len(sec_grid)), 2),
        "Prescribed_Action": cur_f["action"],
        "Target_Component": cur_f["component"],
    })
    st.dataframe(bb_freeze.tail(6), use_container_width=True, hide_index=True)

# ==================== TAB 3: CLOSED-LOOP CMMS ====================
with tab3:
    st.subheader("🔧 SAP-PM Closed-Loop Maintenance Lifecycle (`REPAIR_CALIBRATE` vs `REPLACE_SPARE`)")
    wo_df = fetch_work_orders()
    st.dataframe(wo_df, use_container_width=True, hide_index=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown("**Stage 2: Planner Approval & Store Check**")
        pend = wo_df[wo_df["status"] == "AUTO_NOTIFICATION"]
        if not pend.empty:
            sel_id = st.selectbox("Select Notification ID:", pend["id"].tolist())
            if st.button("✅ Approve & Dispatch Order"):
                approve_notification(sel_id)
                st.success(f"Work Order #{sel_id} approved!")
                st.rerun()
        else:
            st.info("No pending notifications awaiting approval.")

    with c_m2:
        st.markdown("**Stage 4: Weekend ISO 14224 Sign-Off**")
        open_w = wo_df[wo_df["status"] != "CLOSED"]
        if not open_w.empty:
            close_id = st.selectbox("Select Ticket ID Executed:", open_w["id"].tolist())
            iso_opt = st.selectbox(
                "Verified ISO 14224 Action Taken:",
                [
                    "ISO-14224: ME-02 (Spindle Bearing Replaced)",
                    "ISO-14224: ME-08 (Coolant Strainer Cleaned — No Spare)",
                    "ISO-14224: EL-09 (Z-Axis PLC Backlash Calibrated)",
                    "ISO-14224: FA-00 (False Alarm — Update Threshold)",
                ],
            )
            if st.button("🔒 Close & Log Feedback"):
                close_work_order(close_id, iso_opt)
                st.success(f"Ticket #{close_id} closed!")
                st.rerun()
        else:
            st.info("All tickets closed.")

    with c_m3:
        st.markdown("**Friday Afternoon Shift Handover**")
        st.download_button(
            "⬇️ Download Sat/Sun Dispatch Sheet (CSV)",
            wo_df.to_csv(index=False),
            "mahindra_weekend_work_orders.csv",
            "text/csv",
        )

# ==================== TAB 4: MULTI-LAKH CSV & CHRONOLOGICAL AUDIT ====================
with tab4:
    st.subheader("🗄️ High-Frequency Plant CSV Ingestion, Auto-Fusion & Chronological ML Audit")

    m_a, m_b, m_c, m_d = st.columns(4)
    m_a.metric("Chronological Test Accuracy", f"{val_metrics['accuracy']}%", f"Last 25% Windows (n={val_metrics['test_windows']:,})")
    m_b.metric("Weighted Precision", f"{val_metrics['precision']}%", "Zero Data Leakage")
    m_c.metric("Weighted Recall", f"{val_metrics['recall']}%", "Fault Capture Rate")
    m_d.metric("RUL Prediction Error (MAE)", f"±{val_metrics['rul_mae']} hrs", "15-Min Rolling Windows")

    st.divider()
    st.markdown("### 📤 Upload Raw (44-Sensor) or Full (63-Column) Plant CSV")
    st.write(
        "Upload any company minute-by-minute CSV. If your file has **only the 44 raw PLC sensor columns**,"
        " the ingestion engine automatically computes all **12 High-Value Decision Columns** and compresses"
        " the logs into 15-minute rolling windows to retrain the models live."
    )

    up_c1, up_c2 = st.columns([1, 2])
    sample_export = active_raw_df.head(1000)
    up_c1.download_button(
        "⬇️ Download Sample 1,000-Row × 63-Col CSV",
        sample_export.to_csv(index=False),
        "mahindra_63col_sample.csv",
        "text/csv",
    )
    uploaded_plant_csv = up_c2.file_uploader(
        "Upload Plant Telemetry CSV (44 Raw Cols or 63 Full Cols):", type=["csv"]
    )

    if uploaded_plant_csv is not None:
        up_df = pd.read_csv(uploaded_plant_csv)
        has_12 = all(c in up_df.columns for c in ENGINEERED_12_COLS)
        has_44 = all(c in up_df.columns for c in RAW_44_COLS)
        if has_12 or has_44:
            if not has_12 and has_44:
                up_df = compute_12_engineered_features(up_df)
            if "Action_Code" not in up_df.columns:
                acts = [prescribe_part_action(r)[0] for _, r in up_df.iterrows()]
                rev_map = {"HEALTHY": 0, "REPAIR_CALIBRATE": 1, "REPLACE_SPARE": 2}
                up_df["Action_Code"] = [rev_map[a] for a in acts]
            if "RUL_Hours" not in up_df.columns:
                up_df["RUL_Hours"] = np.round((up_df["Composite_OEE_Risk_Score"] / 100.0) * 140.0, 1)

            if st.button("🚀 Ingest Uploaded CSV, Fuse Parameters & Retrain Models"):
                st.session_state.custom_raw_df = up_df
                st.success(f"Ingested {len(up_df):,} rows and retrained models chronologically!")
                st.rerun()
        else:
            st.error("CSV must contain either the 44 raw PLC sensor columns or the 12 engineered columns.")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            go.Figure(
                go.Bar(
                    x=feat_imp_df["Importance"],
                    y=feat_imp_df["Engineered_Parameter"],
                    orientation="h",
                    marker_color="#1f77b4",
                )
            ).update_layout(title="Top Driving Engineered Parameters (Random Forest)", height=340),
            use_container_width=True,
        )
    with g2:
        cm = val_metrics["cm"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred Healthy", "Pred Repair/Calib", "Pred Replace"],
                y=["Actual Healthy", "Actual Repair/Calib", "Actual Replace"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig_cm.update_layout(title="Chronological Out-of-Time Confusion Matrix", height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("🔍 Inspect Live Master Telemetry Table (63 Columns)"):
        st.dataframe(active_raw_df.head(50), use_container_width=True)
