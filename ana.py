# app.py
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import tempfile
import logging

# --- Setup ---
st.set_page_config(page_title="F1 Data Dashboard", layout="wide")
logging.getLogger("fastf1").setLevel(logging.ERROR)

# FastF1 cache: use a writable temp dir on Streamlit Cloud
cache_dir = tempfile.mkdtemp()
fastf1.Cache.enable_cache(cache_dir)

st.sidebar.title("Mode & Session")
mode = st.sidebar.radio("Choose mode", ["Simple", "Advanced"])

# Optional: path to the original uploaded notebook (for reference)
original_notebook_path = "/mnt/data/formula1.py"

# --- Session controls ---
with st.sidebar.expander("Session selection", expanded=True):
    year = st.number_input("Year", min_value=2018, max_value=2026, value=2025, step=1)
    gp = st.text_input("Grand Prix (name)", "Las Vegas")
    session_type = st.selectbox("Session", ["FP1", "FP2", "FP3", "Q", "R"], index=4)
    load_telemetry = st.checkbox("Load telemetry (slower)", value=False)
    load_btn = st.button("Load session")

# container for messages
status = st.empty()

session = None
if load_btn:
    try:
        status.info("Loading session... (may take 10-30s on first load)")
        session = fastf1.get_session(int(year), gp, session_type)
        session.load(telemetry=load_telemetry)
        status.success("Session loaded successfully ✅")
    except Exception as e:
        status.error(f"Failed to load session: {e}")
        session = None

# If session not loaded yet, show quick instructions
if session is None:
    st.title("F1 Data Dashboard")
    st.markdown(
        """
        **Instructions**
        1. Use the sidebar to choose Year, Grand Prix name (e.g. `Las Vegas`), and Session (`R` for race).  
        2. Click **Load session**.  
        3. Turn on `Load telemetry` if you plan to view speed traces (this will be slower).
        """
    )
    st.caption(f"Reference notebook (uploaded): `{original_notebook_path}`")
    st.stop()

# --- Common derived objects ---
laps = session.laps.copy()
drivers = sorted(laps['Driver'].unique())

# Sidebar driver selection (used in both modes)
st.sidebar.markdown("---")
driver1 = st.sidebar.selectbox("Driver 1", drivers, index=0)
driver2 = st.sidebar.selectbox("Driver 2", drivers, index=min(1, len(drivers)-1))

# -----------------------
# SIMPLE MODE (Option A)
# -----------------------
if mode == "Simple":
    st.header(f"Simple View — {year} {gp} ({session_type})")
    # Pit stop table
    st.subheader("Pit stop summary")
    pit_stops = laps.loc[laps['PitInTime'].notna(), ['Driver','LapNumber','Stint','PitInTime','PitOutTime','Compound','TyreLife','FreshTyre']].sort_values(['Driver','LapNumber'])
    st.dataframe(pit_stops)

    # Lap time trend (Plotly)
    st.subheader("Lap time trend (all drivers)")
    laps_clean = laps.dropna(subset=['LapTime']).copy()
    laps_clean['LapTime_s'] = laps_clean['LapTime'].dt.total_seconds()
    fig = px.line(laps_clean, x='LapNumber', y='LapTime_s', color='Driver', title="Lap Time Trend", labels={'LapTime_s': 'Lap Time (s)'})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Driver comparison (simple)
    st.subheader("Driver lap times: Driver 1 vs Driver 2")
    v1 = laps[laps['Driver'] == driver1].dropna(subset=['LapTime'])
    v2 = laps[laps['Driver'] == driver2].dropna(subset=['LapTime'])
    if v1.empty or v2.empty:
        st.warning("No lap-time data for one of the selected drivers.")
    else:
        fig2 = px.line(pd.concat([
            v1.assign(DriverLabel=driver1)[['LapNumber','LapTime']].assign(LapTime_s=lambda df: df['LapTime'].dt.total_seconds()),
            v2.assign(DriverLabel=driver2)[['LapNumber','LapTime']].assign(LapTime_s=lambda df: df['LapTime'].dt.total_seconds())
        ]).rename(columns={'DriverLabel':'Driver'}),
        x='LapNumber', y='LapTime_s', color='Driver', title=f"{driver1} vs {driver2} Lap Times", labels={'LapTime_s':'Lap Time (s)'})
        st.plotly_chart(fig2, use_container_width=True)

    st.success("Simple view ready — try Advanced mode for more features!")

# -------------------------
# ADVANCED MODE (Option B)
# -------------------------
else:
    st.header(f"Advanced Analytics — {year} {gp} ({session_type})")

    # 1) Sector heatmap (average sector times)
    st.subheader("Sector heatmap (avg sector times)")
    sector_df = laps.dropna(subset=['Sector1Time','Sector2Time','Sector3Time']).copy()
    if sector_df.empty:
        st.warning("Sector data not available.")
    else:
        sector_df['S1'] = sector_df['Sector1Time'].dt.total_seconds()
        sector_df['S2'] = sector_df['Sector2Time'].dt.total_seconds()
        sector_df['S3'] = sector_df['Sector3Time'].dt.total_seconds()
        avg = sector_df.groupby('Driver')[['S1','S2','S3']].mean().sort_values('S1')
        st.dataframe(avg)
        st.pyplot(plt.figure(figsize=(8,6)))
        plt.imshow(avg, aspect='auto', cmap='viridis_r')
        plt.colorbar(label='Seconds')
        plt.yticks(range(len(avg.index)), avg.index)
        plt.xticks(range(3), ['Sector1','Sector2','Sector3'])
        plt.title('Avg Sector Times Heatmap')
        st.pyplot(plt.gcf())

    # 2) Stint / tyre strategy plot (matplotlib)
    st.subheader("Tyre stint visualization")
    fig_stint, ax_stint = plt.subplots(figsize=(12, max(4, len(drivers)*0.3)))
    for i, d in enumerate(drivers):
        d_laps = laps[laps['Driver']==d]
        if d_laps.empty:
            continue
        for stn in d_laps['Stint'].unique():
            s = d_laps[d_laps['Stint']==stn]
            start = int(s['LapNumber'].min())
            end = int(s['LapNumber'].max())
            comp = s['Compound'].iloc[0] if 'Compound' in s.columns else 'unknown'
            color = {'SOFT':'red','MEDIUM':'gold','HARD':'silver'}.get(comp.upper() if isinstance(comp,str) else comp, 'grey')
            ax_stint.barh(i, end-start+1, left=start, color=color, edgecolor='black', height=0.6)
            ax_stint.text(start + (end-start)/2, i, comp, va='center', ha='center', fontsize=7)
    ax_stint.set_yticks(range(len(drivers)))
    ax_stint.set_yticklabels(drivers)
    ax_stint.set_xlabel('Lap Number')
    ax_stint.set_title('Tyre Stints by Driver')
    st.pyplot(fig_stint)

    # 3) Speed trace comparison (telemetry required)
    st.subheader("Speed trace comparison (fastest laps)")
    lap1 = session.laps.pick_driver(driver1).pick_fastest() if not session.laps.pick_driver(driver1).empty else None
    lap2 = session.laps.pick_driver(driver2).pick_fastest() if not session.laps.pick_driver(driver2).empty else None

    if lap1 is None or lap2 is None:
        st.warning("One of the selected drivers has no laps in this session.")
    else:
        if not load_telemetry:
            st.info("Enable 'Load telemetry' in sidebar and reload session to view speed traces.")
        else:
            try:
                tel1 = lap1.get_car_data().add_distance()
                tel2 = lap2.get_car_data().add_distance()
                fig_speed, ax_speed = plt.subplots(figsize=(12,5))
                ax_speed.plot(tel1['Distance'], tel1['Speed'], label=driver1)
                ax_speed.plot(tel2['Distance'], tel2['Speed'], label=driver2)
                ax_speed.set_xlabel('Distance (m)')
                ax_speed.set_ylabel('Speed (km/h)')
                ax_speed.set_title(f'Speed trace: {driver1} vs {driver2}')
                ax_speed.legend()
                st.pyplot(fig_speed)
            except Exception as e:
                st.error(f"Telemetry load error: {e}")

    # 4) Throttle distribution boxplot (if telemetry available)
    st.subheader("Throttle distribution (per driver) — requires telemetry")
    if not load_telemetry:
        st.info("Telemetry disabled — enable 'Load telemetry' to view throttle distributions.")
    else:
        avg_throttle = {}
        for d in drivers:
            d_laps = session.laps.pick_driver(d)
            if d_laps.empty:
                continue
            try:
                car = d_laps.get_car_data().add_distance()
                if 'Throttle' in car.columns:
                    avg_throttle[d] = car['Throttle'].mean()
            except Exception:
                continue
        if not avg_throttle:
            st.warning("No throttle telemetry available.")
        else:
            df_th = pd.DataFrame.from_dict(avg_throttle, orient='index', columns=['AvgThrottle']).sort_values('AvgThrottle', ascending=False)
            st.bar_chart(df_th)

    # 5) Position evolution (lap-by-lap)
    st.subheader("Lap-by-lap positions")
    if 'Position' not in laps.columns or laps['Position'].isna().all():
        st.warning("Position data not available.")
    else:
        pos_df = laps.dropna(subset=['Position']).copy()
        pos_df['Position'] = pos_df['Position'].astype(int)
        fig_pos = px.line(pos_df, x='LapNumber', y='Position', color='Driver', title='Position over laps')
        fig_pos.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_pos, use_container_width=True)

    # 6) Finishing time difference from winner
    st.subheader("Finishing time difference (from winner)")
    try:
        results = session.results.copy()
        if 'Time' in results.columns and pd.api.types.is_timedelta64_dtype(results['Time']):
            winner_time = results.iloc[0]['Time']
            results['TimeDiff_s'] = (results['Time'] - winner_time).dt.total_seconds()
            fig_time = px.bar(results, x='TimeDiff_s', y='FullName', orientation='h', title='Time difference from winner (s)')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("Results `Time` column not present or not timedelta.")
    except Exception as e:
        st.error(f"Error building results chart: {e}")

    st.markdown("---")
    st.caption("Reference notebook (uploaded): `%s`" % original_notebook_path)
