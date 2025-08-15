"""
AI-Powered Machine Diagnostics Analyzer
"""

import streamlit as st
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import sqlite3
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import libsql_client
from sklearn.ensemble import IsolationForest # Add this with your other imports at the top of the file
import plotly.express as px
import math

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")

# --- Global Configuration & Constants ---
FAULT_LABELS = [
    "Normal", "Valve Leakage", "Valve Wear", "Valve Sticking or Fouling",
    "Valve Impact or Slamming", "Broken or Missing Valve Parts",
    "Valve Misalignment", "Spring Fatigue or Failure", "Other"
]
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- Database Setup ---
@st.cache_resource
def init_db():
    """Initializes the database connection using Turso."""
    try:
        url = st.secrets["TURSO_DATABASE_URL"]
        auth_token = st.secrets["TURSO_AUTH_TOKEN"]
        client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
        client.batch([
            "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, machine_id TEXT, rpm TEXT)",
            "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, anomaly_count INTEGER, threshold REAL, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, analysis_id INTEGER, label_text TEXT, FOREIGN KEY (analysis_id) REFERENCES analyses (id))",
            "CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, analysis_id INTEGER, event_type TEXT, crank_angle REAL, FOREIGN KEY (analysis_id) REFERENCES analyses (id))"
        ])
        return client
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()

# --- Helper Functions ---

import plotly.express as px

def display_historical_analysis(db_client):
    """
    Queries the database for historical data and displays it as a trend chart.
    """
    st.subheader("Anomaly Count Trend Over Time")

    query = """
        SELECT
            s.timestamp,
            s.machine_id,
            SUM(a.anomaly_count) as total_anomalies
        FROM analyses a
        JOIN sessions s ON a.session_id = s.id
        GROUP BY s.id, s.timestamp, s.machine_id
        ORDER BY s.timestamp ASC
    """
    try:
        rs = db_client.execute(query)
        if not rs.rows:
            st.info("No historical analysis data found to display.")
            return

        # --- THIS IS THE CORRECTED LINE ---
        # We manually provide the column names instead of reading them from the result object.
        df = pd.DataFrame(rs.rows, columns=['timestamp', 'machine_id', 'total_anomalies'])
        # ------------------------------------
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if df.empty:
            st.info("No historical analysis data found to display.")
            return

        # Create the plot
        fig = px.line(
            df,
            x='timestamp',
            y='total_anomalies',
            color='machine_id',
            markers=True,
            title='Total Anomaly Count by Machine Over Time',
            labels={
                "timestamp": "Date of Analysis",
                "total_anomalies": "Total Anomalies Found",
                "machine_id": "Machine ID"
            }
        )
        fig.update_layout(template="ggplot2")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
# The function signature now accepts contamination_level
def run_anomaly_detection(df, curve_names, contamination_level=0.05): 
    """
    Applies Isolation Forest to detect anomalies and calculate their scores.
    ...
    """
    for curve in curve_names:
        if curve in df.columns:
            data = df[[curve]].values
            # Use the new parameter here instead of 'auto'
            model = IsolationForest(contamination=contamination_level, random_state=42)
            
            # ... the rest of the function remains the same ...
            
            # Fit the model and get binary predictions (-1 for anomalies)
            predictions = model.fit_predict(data)
            df[f'{curve}_anom'] = predictions == -1

            # Get the raw anomaly scores. Lower scores are more anomalous.
            anomaly_scores = model.score_samples(data)
            
            # We invert the scores so that higher values indicate a more severe anomaly.
            # This is more intuitive for visualization and reporting.
            df[f'{curve}_anom_score'] = -1 * anomaly_scores
            
    return df
def run_rule_based_diagnostics(report_data):
    """
    Applies simple rule-based logic and returns a dict mapping report items
    to a suggested label (or None if no suggestion).
    """
    suggestions = {}
    for item in report_data:
        if item['name'] == 'Pressure' and item['count'] > 10:
            suggestions[item['name']] = 'Valve Leakage'
        elif item['name'] != 'Pressure' and item['count'] > 5:
            suggestions[item['name']] = f'{item["name"]} wear'
    return suggestions
    
def compute_health_score(report_data, diagnostics):
    """
    Computes a simple health index from anomalies and rule-based diagnostics.
    Starts from 100 and subtracts a penalty for each anomaly and each diagnostic message.
    Returns a value between 0 and 100.
    """
    score = 100
    # Subtract one point per anomaly detected
    for item in report_data:
        score -= item['count'] * 0.5
    # Subtract an additional five points per diagnostic rule triggered
    score -= 2 * len(diagnostics)
    # Keep the score within 0–100
    return max(min(score, 100), 0)
    
def get_last_row_id(_client):
    rs = _client.execute("SELECT last_insert_rowid()")
    return rs.rows[0][0] if rs.rows else None

def find_xml_value(root, sheet_name, partial_key, col_offset, occurrence=1):
    try:
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
        if ws is None: return "N/A"
        rows = ws.findall('.//ss:Row', NS)
        match_count = 0
        for row in rows:
            all_cells_in_row = row.findall('ss:Cell', NS)
            if not all_cells_in_row: continue
            first_cell_data_node = all_cells_in_row[0].find('ss:Data', NS)
            if first_cell_data_node is None or first_cell_data_node.text is None: continue
            if partial_key.upper() in (first_cell_data_node.text or "").strip().upper():
                match_count += 1
                if match_count == occurrence:
                    target_idx = col_offset + 1
                    dense_cells = {}
                    current_idx = 1
                    for cell in all_cells_in_row:
                        ss_index_str = cell.get(f'{{{NS["ss"]}}}Index')
                        if ss_index_str: current_idx = int(ss_index_str)
                        dense_cells[current_idx] = cell
                        current_idx += 1
                    if target_idx in dense_cells:
                        value_node = dense_cells[target_idx].find('ss:Data', NS)
                        return value_node.text if value_node is not None and value_node.text else "N/A"
                    return "N/A"
        return "N/A"
    except Exception:
        return "N/A"
        
@st.cache_data
def load_all_curves_data(_curves_xml_content):
    try:
        root = ET.fromstring(_curves_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'), None)
        if ws is None: return None, None
        table = ws.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]
        data = [[cell.find('ss:Data', NS).text for cell in r.findall('ss:Cell', NS)] for r in rows[6:]]
        if not data: return None, None
        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]
        df = pd.DataFrame(data, columns=actual_columns).apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)
        return df, actual_columns
    except Exception as e:
        st.error(f"Failed to load curves data: {e}")
        return None, None

def extract_rpm(_levels_xml_content):
    try:
        root = ET.fromstring(_levels_xml_content)
        rpm_str = find_xml_value(root, 'Levels', 'RPM', 1)
        if rpm_str != "N/A":
            return f"{float(rpm_str):.0f}"
    except Exception:
        return "N/A"
    return "N/A"

@st.cache_data
def auto_discover_configuration(_source_xml_content, all_curve_names):
    try:
        source_root = ET.fromstring(_source_xml_content)

        # Determine number of cylinders
        num_cyl_str = find_xml_value(
            source_root, 'Source', "COMPRESSOR NUMBER OF CYLINDERS", 2
        )
        if num_cyl_str == "N/A" or int(num_cyl_str) == 0:
            return None
        num_cylinders = int(num_cyl_str)

        # Machine identifier
        machine_id = find_xml_value(source_root, 'Source', "Machine", 1)

        # Additional machine-level metadata
        machine_model = find_xml_value(
            source_root, 'Source', 'COMPRESSOR MODEL', 2
        )
        serial_number = find_xml_value(
            source_root, 'Source', 'COMPRESSOR SERIAL NUMBER', 2
        )
        rated_rpm = find_xml_value(
            source_root, 'Source', 'COMPRESSOR RATED RPM', 2
        )
        rated_hp = find_xml_value(
            source_root, 'Source', 'COMPRESSOR RATED HP', 2
        )

        # Cylinder-specific metadata
        bore_values = []
        rod_diameter_values = []
        stroke_values = []
        num_cols_offset_start = 2  # first cylinder’s value is in column index 2

        for cyl_idx in range(num_cylinders):
            bore = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER BORE',
                num_cols_offset_start + cyl_idx,
            )
            bore_values.append(float(bore) if bore not in [None, '', 'N/A'] else None)

            rod_dia = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER PISTON ROD DIAMETER',
                num_cols_offset_start + cyl_idx,
            )
            rod_diameter_values.append(float(rod_dia) if rod_dia not in [None, '', 'N/A'] else None)

            stroke = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR THROW STROKE LENGTH',
                num_cols_offset_start + cyl_idx,
            )
            stroke_values.append(float(stroke) if stroke not in [None, '', 'N/A'] else None)

        # Build configuration for each cylinder
        cylinders_config = []
        for i in range(1, num_cylinders + 1):
            # Identify the pressure and vibration curves using original pattern matching
            pressure_curve = next(
                (c for c in all_curve_names if f".{i}H." in c and "STATIC" in c),
                None,
            ) or next(
                (c for c in all_curve_names if f".{i}C." in c and "STATIC" in c),
                None,
            )

            valve_curves = [
                {
                    "name": "HE Discharge",
                    "curve": next(
                        (c for c in all_curve_names if f".{i}HD" in c and "VIBRATION" in c),
                        None,
                    ),
                },
                {
                    "name": "HE Suction",
                    "curve": next(
                        (c for c in all_curve_names if f".{i}HS" in c and "VIBRATION" in c),
                        None,
                    ),
                },
                {
                    "name": "CE Discharge",
                    "curve": next(
                        (c for c in all_curve_names if f".{i}CD" in c and "VIBRATION" in c),
                        None,
                    ),
                },
                {
                    "name": "CE Suction",
                    "curve": next(
                        (c for c in all_curve_names if f".{i}CS" in c and "VIBRATION" in c),
                        None,
                    ),
                },
            ]

            if pressure_curve and any(vc['curve'] for vc in valve_curves):
                bore = bore_values[i - 1]
                rod_dia = rod_diameter_values[i - 1]
                stroke = stroke_values[i - 1]
                volume = None
                if bore is not None and stroke is not None:
                    volume = math.pi * (bore / 2) ** 2 * stroke

                cylinders_config.append(
                    {
                        "cylinder_name": f"Cylinder {i}",
                        "pressure_curve": pressure_curve,
                        "valve_vibration_curves": [vc for vc in valve_curves if vc['curve']],
                        "bore": bore,
                        "rod_diameter": rod_dia,
                        "stroke": stroke,
                        "volume": volume,
                    }
                )

        return {
            "machine_id": machine_id,
            "model": machine_model,
            "serial_number": serial_number,
            "rated_rpm": rated_rpm,
            "rated_hp": rated_hp,
            "num_cylinders": num_cylinders,
            "cylinders": cylinders_config,
        }

    except Exception as e:
        st.error(f"Error during auto-discovery: {e}")
        return None


def generate_health_report_table(_source_xml_content, _levels_xml_content, cylinder_index):
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        col_idx = cylinder_index
        
        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str: return "N/A"
            try: return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError): return kpa_str

        # This is the new helper function from Step 1
        def format_numeric_value(value_str, precision=2):
            if value_str == "N/A" or not value_str: return "N/A"
            try: return f"{float(value_str):.{precision}f}"
            except (ValueError, TypeError): return value_str

        suction_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2))
        discharge_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2))
        suction_temp = find_xml_value(levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2)
        discharge_temp = find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx + 1)
        bore = find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx + 1)
        rod_diam = find_xml_value(source_root, 'Source', 'PISTON ROD DIAMETER', col_idx + 1)
        
        # ✅ CHANGED: Extract raw values first
        comp_ratio_he_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=2)
        comp_ratio_ce_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=1)
        power_he_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=2)
        power_ce_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=1)

        # ✅ CHANGED: Apply formatting to the extracted values
        comp_ratio_he = format_numeric_value(comp_ratio_he_raw, precision=2)
        comp_ratio_ce = format_numeric_value(comp_ratio_ce_raw, precision=2)
        power_he = format_numeric_value(power_he_raw, precision=1)
        power_ce = format_numeric_value(power_ce_raw, precision=1)

        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'], 'Bore (ins)': [bore] * 2, 'Rod Diam (ins)': ['N/A', rod_diam],
            'Pressure Ps/Pd (psig)': [f"{suction_p} / {discharge_p}"] * 2, 'Temp Ts/Td (°C)': [f"{suction_temp} / {discharge_temp}"] * 2,
            'Comp. Ratio': [comp_ratio_he, comp_ratio_ce], 
            'Indicated Power (ihp)': [power_he, power_ce]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not generate health report: {e}")
        return pd.DataFrame()

def get_all_cylinder_details(_source_xml_content, _levels_xml_content, num_cylinders):
    details = []
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str:
                return "N/A"
            try:
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str

        def format_flow_balance(value_str):
            if value_str == "N/A" or not value_str:
                return "N/A"
            try:
                return f"{float(value_str) * 100:.1f} %"
            except (ValueError, TypeError):
                return value_str

        stage_suction_p_psi = convert_kpa_to_psi(
            find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2)
        )
        stage_discharge_p_psi = convert_kpa_to_psi(
            find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2)
        )
        stage_suction_temp = find_xml_value(
            levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2
        )

        for i in range(1, num_cylinders + 1):
            col_idx = i + 1

            # Retrieve flow balance values
            fb_ce_raw = find_xml_value(
                source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=1
            )
            fb_he_raw = find_xml_value(
                source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=2
            )

            # Extract dimension values
            bore_value = find_xml_value(
                source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx
            )
            rod_dia_value = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER PISTON ROD DIAMETER',
                col_idx,
            )
            stroke_value = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR THROW STROKE LENGTH',
                col_idx,
            )

            # Compute volume (in³) if bore and stroke are numeric
            volume_val = "N/A"
            try:
                if bore_value not in [None, '', 'N/A'] and stroke_value not in [
                    None,
                    '',
                    'N/A',
                ]:
                    bore_f = float(bore_value)
                    stroke_f = float(stroke_value)
                    volume_val = f"{math.pi * (bore_f / 2) ** 2 * stroke_f:.2f}"
            except Exception:
                volume_val = "N/A"

            detail = {
                "name": f"Cylinder {i}",
                "bore": f"{bore_value} in",
                "rod_diameter": f"{rod_dia_value} in"
                if rod_dia_value not in [None, '', 'N/A']
                else "N/A",
                "stroke": f"{stroke_value} in"
                if stroke_value not in [None, '', 'N/A']
                else "N/A",
                "volume": f"{volume_val} in³" if volume_val != "N/A" else "N/A",
                "suction_temp": f"{stage_suction_temp} °C",
                "discharge_temp": f"{find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx)} °C",
                "suction_pressure": f"{stage_suction_p_psi} psig",
                "discharge_pressure": f"{stage_discharge_p_psi} psig",
                "flow_balance_ce": format_flow_balance(fb_ce_raw),
                "flow_balance_he": format_flow_balance(fb_he_raw),
            }
            details.append(detail)

        return details
    except Exception as e:
        st.warning(f"Could not extract cylinder details: {e}")
        return []


def generate_pdf_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None):
    if not REPORTLAB_AVAILABLE:
        st.warning("ReportLab not installed. PDF generation unavailable.")
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Machine Diagnostics Report - {machine_id}", styles['Title']))
    info_text = f"<b>RPM:</b> {rpm}<br/><b>Cylinder:</b> {cylinder_name}"
    story.append(Paragraph(info_text, styles['Normal']))
    story.append(Spacer(1, 12))
    if chart_fig:
        img_buffer = io.BytesIO()
        chart_fig.write_image(img_buffer, format='png')
        img_buffer.seek(0)
        from reportlab.platypus import Image
        story.append(Image(img_buffer, width=450, height=250))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Health Report", styles['h2']))
    table_data = [health_report_df.columns.tolist()] + health_report_df.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_cylinder_view(_db_client, df, cylinder_config, envelope_view, vertical_offset, analysis_ids, contamination_level):
    pressure_curve = cylinder_config.get('pressure_curve')
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []

    curves_to_analyze = [vc['curve'] for vc in valve_curves if vc['curve'] in df.columns]
    if pressure_curve and pressure_curve in df.columns:
        curves_to_analyze.append(pressure_curve)

    df = run_anomaly_detection(df, curves_to_analyze, contamination_level)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if pressure_curve and pressure_curve in df.columns:
        # --- FIX #1 IS HERE ---
        anomaly_count = int(df[f'{pressure_curve}_anom'].sum())
        # ----------------------
        avg_score = df.loc[df[f'{pressure_curve}_anom'], f'{pressure_curve}_anom_score'].mean() if anomaly_count > 0 else 0.0
        report_data.append({"name": "Pressure", "curve_name": pressure_curve, "threshold": avg_score, "count": anomaly_count, "unit": "PSI"})
        fig.add_trace(go.Scatter(x=df['Crank Angle'], y=df[pressure_curve], name='Pressure (PSI)', line=dict(color='black', width=2)), secondary_y=False)

    for vc in valve_curves:
        curve_name = vc['curve']
        if curve_name in df.columns:
            # --- FIX #2 IS HERE ---
            anomaly_count = int(df[f'{curve_name}_anom'].sum())
            # ----------------------
            avg_score = df.loc[df[f'{curve_name}_anom'], f'{curve_name}_anom_score'].mean() if anomaly_count > 0 else 0.0
            report_data.append({"name": vc['name'], "curve_name": curve_name, "threshold": avg_score, "count": anomaly_count, "unit": "G"})

    colors = plt.cm.viridis(np.linspace(0, 1, len(valve_curves)))
    current_offset = 0

    for i, vc in enumerate(valve_curves):
        curve_name, label_name = vc['curve'], vc['name']
        if curve_name not in df.columns:
            continue
            
        color_rgba = f'rgba({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255},0.4)'

        if envelope_view:
            upper_bound = df[curve_name] + current_offset
            lower_bound = -df[curve_name] + current_offset
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=upper_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), showlegend=False, hoverinfo='none'), secondary_y=True)
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=lower_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), fill='tonexty', fillcolor=color_rgba, name=label_name, hoverinfo='none'), secondary_y=True)
        else:
            vibration_data = df[curve_name] + current_offset
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=vibration_data, name=label_name, mode='lines', line=dict(color=color_rgba.replace('0.4','1'))), secondary_y=True)

        anomalies_df = df[df[f'{curve_name}_anom']]
        if not anomalies_df.empty:
            anomaly_vibration_data = anomalies_df[curve_name] + current_offset
            fig.add_trace(go.Scatter(
                x=anomalies_df['Crank Angle'],
                y=anomaly_vibration_data,
                mode='markers',
                name=f"{label_name} Anomalies",
                marker=dict(
                    color=anomalies_df[f'{curve_name}_anom_score'],
                    colorscale='Reds',
                    showscale=False
                ),
                hoverinfo='text',
                text=[f'Score: {score:.2f}' for score in anomalies_df[f'{curve_name}_anom_score']],
                showlegend=False
            ), secondary_y=True)
        
        analysis_id = analysis_ids.get(vc['name'])
        if analysis_id:
            events_raw = _db_client.execute("SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?", (analysis_id,)).rows
            events = {etype: angle for etype, angle in events_raw}
            if 'open' in events and 'close' in events:
                fig.add_vrect(x0=events['open'], x1=events['close'], fillcolor=color_rgba.replace('0.4','0.2'), layer="below", line_width=0)
            for event_type, crank_angle in events.items():
                fig.add_vline(x=crank_angle, line_width=2, line_dash="dash", line_color='green' if event_type == 'open' else 'red')
        
        current_offset += vertical_offset
    
    fig.update_layout(
        height=700, 
        title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}", 
        xaxis_title="Crank Angle (deg)", 
        template="ggplot2", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Pressure (PSI)</b>", color="black", secondary_y=False)
    fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", color="blue", secondary_y=True)
    
    return fig, report_data
    # --- END OF FULLY CORRECTED FUNCTION ---
    # --- END OF CORRECTED FUNCTION ---

# --- Main Application ---
db_client = init_db()

if 'active_session_id' not in st.session_state: st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state: st.session_state.file_uploader_key = 0

st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")
st.markdown("Upload your machine's XML data files. The configuration will be discovered automatically.")

with st.sidebar:
    st.header("1. Upload Data Files")
    uploaded_files = st.file_uploader("Upload Curves, Levels, Source XML files", type=["xml"], accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
    if st.button("Start New Analysis / Clear Files"):
        st.session_state.file_uploader_key += 1
        st.session_state.active_session_id = None
        st.rerun()

    st.header("2. View Options")
    envelope_view = st.checkbox("Enable Envelope View", value=True)
    vertical_offset = st.slider("Vertical Offset", 0.0, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("AI Model Tuning")
    contamination_level = st.slider(
        "Anomaly Detection Sensitivity", 
        min_value=0.01, 
        max_value=0.20, 
        value=0.05,
        step=0.01,
        help="Adjust the proportion of data points considered as anomalies. Higher values mean more sensitive detection."
    )

if uploaded_files and len(uploaded_files) == 3:
    files_content = {}
    for f in uploaded_files:
        if 'curves' in f.name.lower(): files_content['curves'] = f.getvalue().decode('utf-8')
        elif 'levels' in f.name.lower(): files_content['levels'] = f.getvalue().decode('utf-8')
        elif 'source' in f.name.lower(): files_content['source'] = f.getvalue().decode('utf-8')

    if 'curves' in files_content and 'source' in files_content and 'levels' in files_content:
        df, actual_curve_names = load_all_curves_data(files_content['curves'])
        if df is not None:
            discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
            if discovered_config:
               st.session_state['auto_discover_config'] = discovered_config
               rpm = extract_rpm(files_content['levels'])
               machine_id = discovered_config.get('machine_id', 'N/A')
               if st.session_state.active_session_id is None:
                    db_client.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (machine_id, rpm))
                    st.session_state.active_session_id = get_last_row_id(db_client)
                    st.success(f"✅ New analysis session #{st.session_state.active_session_id} created.")

               cylinders = discovered_config.get("cylinders", [])
               cylinder_names = [c.get("cylinder_name") for c in cylinders]
               with st.sidebar:
                    selected_cylinder_name = st.selectbox("Select Cylinder for Detailed View", cylinder_names)
                
               selected_cylinder_config = next((c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), None)

               if selected_cylinder_config:
                    # Generate plot and initial data
                    _, temp_report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, {}, contamination_level)
                   
                    # Create or update analysis records in DB
                    analysis_ids = {}
                    for item in temp_report_data:
                        rs = db_client.execute("SELECT id FROM analyses WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?", (st.session_state.active_session_id, selected_cylinder_name, item['curve_name']))
                        existing_id_row = rs.rows[0] if rs.rows else None
                        if existing_id_row:
                            analysis_id = existing_id_row[0]
                            db_client.execute("UPDATE analyses SET anomaly_count = ?, threshold = ? WHERE id = ?", (item['count'], item['threshold'], analysis_id))
                        else:
                            db_client.execute("INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)", (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold']))
                            analysis_id = get_last_row_id(db_client)
                        analysis_ids[item['name']] = analysis_id
                    
                    # Regenerate plot with correct analysis_ids
                    fig, report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, analysis_ids, contamination_level)
                   # Run rule-based diagnostics on the report data
                   # Get a dictionary of suggested labels keyed by the report item name
                    suggestions = run_rule_based_diagnostics(report_data)
                    if suggestions:
                        st.subheader("🛠 Rule‑Based Diagnostics")
                        for name, suggestion in suggestions.items():
                            st.warning(f"{name}: {suggestion}")

                   # Compute and display a health score
                    health_score = compute_health_score(report_data, diagnostics)
                    st.metric("Health Score", f"{health_score:.1f}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Display health report
                    st.subheader("📋 Compressor Health Report")
                    cylinder_index = int(re.search(r'\d+', selected_cylinder_name).group())
                    health_report_df = generate_health_report_table(files_content['source'], files_content['levels'], cylinder_index)
                    if not health_report_df.empty:
                        st.dataframe(health_report_df, use_container_width=True, hide_index=True)

                    # Labeling and event marking
                    with st.expander("Add labels and mark valve events"):
                        st.subheader("Fault Labels")
                        for item in report_data:
                            if item['count'] > 0 and item['name'] != 'Pressure':
                                with st.form(key=f"label_form_{analysis_ids[item['name']]}"):
                                    st.write(f"**{item['name']} Anomaly**")
                                    default_label = suggestions.get(item['name'], None)
                                    if default_label and default_label in FAULT_LABELS:
                                        default_index = FAULT_LABELS.index(default_label)
                                        selected_label = st.selectbox(
                                            "Select fault label:",
                                            options=FAULT_LABELS,
                                            index=default_index,
                                            key=f"sel_{analysis_ids[item['name']]}"
                                        )
                                    else:
                                        selected_label = st.selectbox(
                                            "Select fault label:",
                                            options=FAULT_LABELS,
                                            key=f"sel_{analysis_ids[item['name']]}"
                                        )

                                    custom_label = st.text_input(
                                        "Or enter custom label if 'Other':",
                                        key=f"txt_{analysis_ids[item['name']]}"
                                    )
                                    if st.form_submit_button("Save Label"):
                                        final_label = custom_label if selected_label == "Other" and custom_label else selected_label
                                        if final_label and final_label != "Other":
                                            db_client.execute(
                                                "INSERT INTO labels (analysis_id, label_text) VALUES (?, ?)",
                                                (analysis_ids[item['name']], final_label)
                                            )
                                            st.success(f"Label '{final_label}' saved for {item['name']}.")

                        
                        st.subheader("Mark Valve Open/Close Events")
                        for item in report_data:
                            if item['name'] != 'Pressure':
                                with st.form(key=f"valve_form_{analysis_ids[item['name']]}"):
                                    st.write(f"**{item['name']} Valve Events:**")
                                    cols = st.columns(2)
                                    open_angle = cols[0].number_input("Open Angle", key=f"open_{analysis_ids[item['name']]}", value=None, format="%.2f")
                                    close_angle = cols[1].number_input("Close Angle", key=f"close_{analysis_ids[item['name']]}", value=None, format="%.2f")
                                    if st.form_submit_button(f"Save Events for {item['name']}"):
                                        db_client.execute("DELETE FROM valve_events WHERE analysis_id = ?", (analysis_ids[item['name']],))
                                        if open_angle is not None: db_client.execute("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", (analysis_ids[item['name']], 'open', open_angle))
                                        if close_angle is not None: db_client.execute("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", (analysis_ids[item['name']], 'close', close_angle))
                                        st.success(f"Events updated for {item['name']}.")
                                        st.rerun()

                    # Export and Cylinder Details
                    st.header("📄 Export Report")
                    if st.button("🔄 Generate Report for this Cylinder", type="primary"):
                        pdf_buffer = generate_pdf_report(machine_id, rpm, selected_cylinder_name, report_data, health_report_df, fig)
                        if pdf_buffer:
                            st.download_button("📥 Download PDF Report", pdf_buffer, f"report_{machine_id}_{selected_cylinder_name}.pdf", "application/pdf")

                    st.markdown("---")
                    # 🆕 Machine Info Block
                    cfg = st.session_state.get('auto_discover_config', {})
                    st.markdown(f"""
                    <div style='border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0;'>
                      <strong>Machine ID:</strong> {cfg.get('machine_id','N/A')} &nbsp;|&nbsp;
                      <strong>Model:</strong> {cfg.get('model','N/A')} &nbsp;|&nbsp;
                      <strong>Serial:</strong> {cfg.get('serial_number','N/A')} &nbsp;|&nbsp;
                      <strong>Rated RPM:</strong> {cfg.get('rated_rpm','N/A')} &nbsp;|&nbsp;
                      <strong>Rated HP:</strong> {cfg.get('rated_hp','N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.header("🔧 All Cylinder Details")
                    all_details = get_all_cylinder_details(files_content['source'], files_content['levels'], len(cylinders))
                    if all_details:
                        cols = st.columns(len(all_details) or 1)
                        for i, detail in enumerate(all_details):
                            with cols[i]:
                                st.markdown(f"""
                                <div style='border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;'>
                                <h5>{detail['name']}</h5>
                                <small>Bore: <strong>{detail['bore']}</strong></small><br>
                                <small>Rod Dia.: <strong>{detail.get('rod_diameter', 'N/A')}</strong></small><br>
                                <small>Stroke: <strong>{detail.get('stroke', 'N/A')}</strong></small><br>
                                <small>Volume: <strong>{detail.get('volume', 'N/A')}</strong></small><br>
                                <small>Temps (S/D): <strong>{detail['suction_temp']} / {detail['discharge_temp']}</strong></small><br>
                                <small>Pressures (S/D): <strong>{detail['suction_pressure']} / {detail['discharge_pressure']}</strong></small><br>
                                <small>Flow Balance (CE/HE): <strong>{detail['flow_balance_ce']} / {detail['flow_balance_he']}</strong></small>
                                </div>
                                """, unsafe_allow_html=True)

            else:
                st.error("Could not discover a valid machine configuration.")
        # ... previous code for file processing ...
    else:
        st.error("Failed to process curve data.")
else:
    st.warning("Please upload your XML data files to begin analysis.", icon="⚠️")


# ===================================================================
# --- THIS IS THE FULL BLOCK FOR THE UI INTEGRATION ---

st.markdown("---")
st.header("📈 Historical Trend Analysis")
display_historical_analysis(db_client)

# ===================================================================


# Display All Saved Labels at the bottom
with st.sidebar:
    st.header("3. View All Saved Labels")
    rs = db_client.execute("SELECT DISTINCT machine_id FROM sessions ORDER BY machine_id ASC")
    #... rest of your script
    machine_id_options = [row[0] for row in rs.rows]
    selected_machine_id_filter = st.selectbox("Filter labels by Machine ID", options=["All"] + machine_id_options)

st.header("📋 All Saved Labels")
query = "SELECT s.timestamp, s.machine_id, a.cylinder_name, a.curve_name, l.label_text FROM labels l JOIN analyses a ON l.analysis_id = a.id JOIN sessions s ON a.session_id = s.id"
params = []
if selected_machine_id_filter != "All":
    query += " WHERE s.machine_id = ?"
    params.append(selected_machine_id_filter)
query += " ORDER BY s.timestamp DESC"
rs = db_client.execute(query, tuple(params))
if rs.rows:
    labels_df = pd.DataFrame(rs.rows, columns=['Timestamp', 'Machine ID', 'Cylinder', 'Curve', 'Label'])
    st.dataframe(labels_df, use_container_width=True)
    # Download buttons for labels
    csv_data = labels_df.to_csv(index=False).encode('utf-8')
    st.download_button("📊 Download Labels as CSV", csv_data, "anomaly_labels.csv", "text/csv")
else:
    st.info("📝 No labels found.")
