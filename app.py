import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import streamlit as st

st.set_page_config(
    page_title="Mahindra CNC Line - Production PdM System", layout="wide"
)

# ---------------------------------------------------------
# 1. SQLITE DATABASE WITH AUTO-SEED FOR FUTURE DEMOS
# ---------------------------------------------------------
DB_FILE = "mahindra_maintenance.db"


def init_db():
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            machine TEXT,
            rul_hours REAL,
            assigned_team TEXT,
            fault_mode TEXT,
            scheduled_window TEXT,
            status TEXT
        )
    """)
  # Auto-seed realistic Sat/Sun tickets if container restarted after inactivity
  c.execute("SELECT COUNT(*) FROM tickets")
  if c.fetchone()[0] == 0:
    seed_data = [
        (
            "2026-11-18 14:20",
            "CNC #2 (Boring - Live)",
            64.5,
            "Mechanical Team",
            "Mechanical (Tool/Overstrain/Heat)",
            "SATURDAY 09:00 AM (OFF)",
            "OPEN",
        ),
        (
            "2026-11-14 16:10",
            "CNC #3 (Drilling)",
            41.0,
            "Electrical Team",
            "Electrical (Spindle Power/Drive)",
            "SUNDAY 10:00 AM (OFF)",
            "COMPLETED",
        ),
    ]
    c.executemany(
        "INSERT INTO tickets (created_at, machine, rul_hours, assigned_team,"
        " fault_mode, scheduled_window, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        seed_data,
    )
  conn.commit()
  conn.close()


def add_ticket(machine, rul, team, fault_mode, window):
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute(
      "SELECT id FROM tickets WHERE machine=? AND status='OPEN'", (machine,)
  )
  if c.fetchone() is None:
    c.execute(
        "INSERT INTO tickets (created_at, machine, rul_hours, assigned_team,"
        " fault_mode, scheduled_window, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            machine,
            rul,
            team,
            fault_mode,
            window,
            "OPEN",
        ),
    )
    conn.commit()
  conn.close()


def close_ticket(ticket_id):
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute("UPDATE tickets SET status='COMPLETED' WHERE id=?", (ticket_id,))
  conn.commit()
  conn.close()


def get_tickets():
  conn = sqlite3.connect(DB_FILE)
  df = pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", conn)
  conn.close()
  return df


init_db()


# ---------------------------------------------------------
# 2. LOAD REAL UCI AI4I 2020 CNC DATASET & TRAIN MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_and_train():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
  try:
    df = pd.read_csv(url)
    data_source = "UCI AI4I 2020 Industrial CNC Dataset (10,000 Real Rows)"
  except Exception:
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300.0, 2.0, n),
        "Process temperature [K]": np.random.normal(310.0, 1.5, n),
        "Rotational speed [rpm]": np.random.normal(1538, 179, n),
        "Torque [Nm]": np.random.normal(40.0, 9.9, n),
        "Tool wear [min]": np.random.uniform(0, 250, n),
        "TWF": np.random.binomial(1, 0.005, n),
        "HDF": np.random.binomial(1, 0.011, n),
        "PWF": np.random.binomial(1, 0.009, n),
        "OSF": np.random.binomial(1, 0.010, n),
    })
    data_source = "Offline AI4I-Schema CNC Telemetry (10,000 Rows)"

  # Map UCI failure modes to Core Teams:
  # 0 = Healthy | 1 = Mechanical (TWF/OSF/HDF) | 2 = Electrical (PWF - Power/Motor)
  fault_class = np.zeros(len(df), dtype=int)
  mech_mask = (df["TWF"] == 1) | (df["OSF"] == 1) | (df["HDF"] == 1)
  elec_mask = (df["PWF"] == 1) & (~mech_mask)
  fault_class[mech_mask] = 1
  fault_class[elec_mask] = 2

  # Derive Remaining Useful Life (RUL in hours) from wear, torque overstrain, and thermal delta
  temp_delta = df["Process temperature [K]"] - df["Air temperature [K]"]
  power_watts = (
      df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
  )
  rul = (
      140
      - (df["Tool wear [min]"] * 0.35)
      - (np.maximum(0, df["Torque [Nm]"] - 42) * 1.8)
      - (np.maximum(0, 10.5 - temp_delta) * 12.0)
      - (np.maximum(0, np.abs(power_watts - 6200) - 2200) * 0.015)
  )
  rul = np.clip(rul, 0, 140)
  rul[(mech_mask | elec_mask)] = np.random.uniform(
      0, 8, size=(mech_mask | elec_mask).sum()
  )

  features = [
      "Air temperature [K]",
      "Process temperature [K]",
      "Rotational speed [rpm]",
      "Torque [Nm]",
      "Tool wear [min]",
  ]
  X = df[features]

  clf = RandomForestClassifier(
      n_estimators=100, class_weight="balanced", random_state=42
  ).fit(X, fault_class)
  reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, rul)
  return clf, reg, data_source, df


fault_model, rul_model, dataset_label, raw_df = load_and_train()

# ---------------------------------------------------------
# 3. DASHBOARD UI & PRODUCTION SHIFT CLOCK
# ---------------------------------------------------------
st.title("🏭 Mahindra CNC Conveyor Line — Production PdM & Weekend Scheduler")
st.caption(
    f"**Active ML Training Source:** {dataset_label} | **Database:** SQLite"
    " Persistent Storage"
)

col_clock, col_kpi1, col_kpi2, col_kpi3 = st.columns([2, 1, 1, 1])
with col_clock:
  current_day = st.selectbox(
      "Current Shift Timestamp (Mon–Fri 24h Line):",
      [
          ("Monday 10:00 AM", 110),
          ("Wednesday 02:00 PM", 58),
          ("Thursday 06:00 PM", 30),
          ("Friday 04:00 PM", 8),
      ],
      index=1,
      format_func=lambda x: f"{x[0]} ({x[1]} hrs until Fri Night Shutdown)",
  )
hours_to_friday = current_day[1]

# ---------------------------------------------------------
# ---------------------------------------------------------
# 4. LIVE MULTI-MACHINE PLC & SENSOR FEED (ALL 4 CNCs)
# ---------------------------------------------------------
if "machine_states" not in st.session_state:
  st.session_state.machine_states = {
      "CNC #1 (Roughing)": [298.5, 309.0, 1550, 36.0, 40],
      "CNC #2 (Boring)": [300.5, 310.2, 1420, 52.0, 165],
      "CNC #3 (Drilling)": [299.0, 309.5, 1620, 38.0, 55],
      "CNC #4 (Finishing)": [298.2, 308.8, 1750, 32.0, 25],
  }

st.sidebar.header("🎛️ Live Shop-Floor Sensor Feed")
selected_cnc = st.sidebar.selectbox(
    "Select Machine to Inject Telemetry:",
    list(st.session_state.machine_states.keys()),
)
curr_vals = st.session_state.machine_states[selected_cnc]

air_t = st.sidebar.slider("Air Temp (K)", 295.0, 305.0, float(curr_vals[0]), 0.1)
proc_t = st.sidebar.slider(
    "Process Temp (K)", 305.0, 315.0, float(curr_vals[1]), 0.1
)
rpm = st.sidebar.slider("Spindle Speed (RPM)", 1150, 2800, int(curr_vals[2]), 10)
torque = st.sidebar.slider(
    "Cutting Torque (Nm)", 15.0, 75.0, float(curr_vals[3]), 0.5
)
wear = st.sidebar.slider("Tool Wear (mins)", 0, 250, int(curr_vals[4]), 5)

st.session_state.machine_states[selected_cnc] = [
    air_t,
    proc_t,
    rpm,
    torque,
    wear,
]
machines = st.session_state.machine_states

fault_map = {
    0: "Nominal (No Fault)",
    1: "Mechanical (Tool/Overstrain/Heat)",
    2: "Electrical (Spindle Power/Drive)",
}
line_avail = 100.0

st.subheader("1. Real-Time Conveyor Line Health (`CNC #1 ➔ #2 ➔ #3 ➔ #4`)")
cols = st.columns(4)

for idx, (m_name, vals) in enumerate(machines.items()):
  in_df = pd.DataFrame(
      [vals],
      columns=[
          "Air temperature [K]",
          "Process temperature [K]",
          "Rotational speed [rpm]",
          "Torque [Nm]",
          "Tool wear [min]",
      ],
  )
  pred_rul = round(float(rul_model.predict(in_df)[0]), 1)
  pred_fault = int(fault_model.predict(in_df)[0])
  team = (
      "Mechanical Team"
      if pred_fault == 1
      else (
          "Electrical Team"
          if pred_fault == 2
          else "Mechanical Team (Preventive)"
      )
  )

  with cols[idx]:
    if pred_rul <= 5:
      st.error(f"### {m_name}\n**🔴 CRITICAL TRIP**")
      decision = "Black-Box Lock + Buffer Bypass Active"
      window = "IMMEDIATE EMERGENCY"
      line_avail = 84.0
    elif pred_rul < hours_to_friday:
      st.warning(f"### {m_name}\n**🟠 MID-WEEK RISK**")
      decision = "Throttle Feed 10% to Survive Till Fri"
      window = "SHIFT CHANGE / THROTTLE"
      line_avail = min(line_avail, 97.0)
    elif pred_rul < (hours_to_friday + 48):
      st.info(f"### {m_name}\n**🟡 BOOK SAT/SUN**")
      decision = f"Schedule Sat 09:00 AM ({team})"
      window = "SATURDAY 09:00 AM (OFF)"
    else:
      st.success(f"### {m_name}\n**🟢 HEALTHY**")
      decision = "Run 24/5 Continuous"
      window = None

    st.metric(
        "Predicted RUL",
        f"{pred_rul} hrs",
        delta=f"{round(pred_rul - hours_to_friday, 1)} hrs vs Fri",
    )
    st.write(f"**Signature:** {fault_map[pred_fault]}")
    st.write(f"**Action:** {decision}")

    if window and st.button(f"Log Ticket: {m_name}", key=f"btn_{idx}"):
      add_ticket(m_name, pred_rul, team, fault_map[pred_fault], window)
      st.toast(f"Saved {m_name} ticket to SQLite DB!")

col_kpi1.metric("Mon–Fri Line Availability", f"{line_avail}%", "Target: >98%")
col_kpi2.metric("Dataset Rows Trained", f"{len(raw_df):,}", "UCI AI4I 2020")
col_kpi3.metric(
    "Open Weekend Tickets",
    len(get_tickets()[get_tickets()["status"] == "OPEN"]),
)

# ---------------------------------------------------------
# 5. PERSISTENT SQLITE WORK-ORDER QUEUE & TELEMETRY CHART
# ---------------------------------------------------------
st.divider()
left_col, right_col = st.columns([3, 2])

with left_col:
  st.subheader("2. Persistent Sat/Sun Maintenance Work Orders (SQLite DB)")
  tickets_df = get_tickets()
  if not tickets_df.empty:
    st.dataframe(tickets_df, use_container_width=True)
    open_ids = tickets_df[tickets_df["status"] == "OPEN"]["id"].tolist()
    if open_ids:
      c1, c2 = st.columns([2, 1])
      sel_id = c1.selectbox("Select Ticket ID Completed on Sat/Sun:", open_ids)
      if c2.button("✅ Mark Completed"):
        close_ticket(sel_id)
        st.rerun()
  else:
    st.info(
        "No tickets logged yet. Click 'Log Ticket' on any Yellow/Orange/Red"
        " machine above."
    )

with right_col:
  st.subheader("3. CNC #2 Torque & Tool Wear Degradation")
  power_kw = round((torque * rpm * (2 * np.pi / 60)) / 1000, 2)
  st.metric(
      "Calculated Spindle Power Draw",
      f"{power_kw} kW",
      "Safe Range: 3.5 – 8.5 kW",
  )
  hrs = np.arange(-24, 1)
  torque_hist = np.linspace(36.0, torque, len(hrs)) + np.random.normal(
      0, 0.8, len(hrs)
  )
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=hrs, y=torque_hist, mode="lines+markers", name="Torque (Nm)"
      )
  )
  fig.update_layout(
      height=260,
      xaxis_title="Hours Relative to Now",
      yaxis_title="Torque (Nm)",
      margin=dict(l=10, r=10, t=10, b=10),
  )
  st.plotly_chart(fig, use_container_width=True)
