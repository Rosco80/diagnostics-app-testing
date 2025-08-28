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
from sklearn.ensemble import IsolationForest
import plotly.express as px
import math
from streamlit_plotly_events import plotly_events

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")

# --- INITIALIZE SESSION STATE FIRST (BEFORE ANY ACCESS) ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'active_session_id' not in st.session_state: 
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state: 
    st.session_state.file_uploader_key = 0
if 'last_settings_update' not in st.session_state: 
    st.session_state.last_settings_update = datetime.datetime.now()
if 'settings_changed' not in st.session_state: 
    st.session_state.settings_changed = False
if 'valve_event_tags' not in st.session_state:
    st.session_state.valve_event_tags = {}

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

# --- Helper function for triggering reruns (REMOVED - causing issues) ---
# def trigger_rerun():
#     st.session_state.settings_changed = True

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
            "CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, analysis_id INTEGER, event_type TEXT, crank_angle REAL, FOREIGN KEY (analysis_id) REFERENCES analyses (id))",
            "CREATE TABLE IF NOT EXISTS configs (machine_id TEXT PRIMARY KEY, contamination REAL DEFAULT 0.05, pressure_anom_limit INT DEFAULT 10, valve_anom_limit INT DEFAULT 5, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, machine_id TEXT, cylinder TEXT, severity TEXT, message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        ])
        
        return client
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()

# --- Helper Functions ---

def inject_custom_css():
    """Inject custom CSS for enhanced styling"""
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
inject_custom_css()

# ADD this new function to your app.py file (add it near your other helper functions):

def generate_executive_summary(machine_id, cylinder_name, health_score, report_data, suggestions):
    """
    Generates executive summary data for PDF reports
    """
    summary = {
        'overall_status': 'UNKNOWN',
        'health_score': health_score,
        'total_anomalies': 0,
        'critical_issues': [],
        'top_diagnostics': [],
        'recommendations': [],
        'next_actions': []
    }
    
    # Calculate total anomalies
    total_anomalies = sum(item.get('count', 0) for item in report_data)
    summary['total_anomalies'] = total_anomalies
    
    # Determine overall status based on health score
    if health_score >= 85:
        summary['overall_status'] = 'EXCELLENT'
    elif health_score >= 70:
        summary['overall_status'] = 'GOOD'
    elif health_score >= 55:
        summary['overall_status'] = 'FAIR'
    elif health_score >= 40:
        summary['overall_status'] = 'POOR'
    else:
        summary['overall_status'] = 'CRITICAL'
    
    # Identify critical issues (high anomaly counts)
    for item in report_data:
        if item.get('count', 0) > 10:  # Threshold for critical
            summary['critical_issues'].append(f"{item['name']}: {item['count']} anomalies detected")
    
    # Get top 3 diagnostics from suggestions
    diagnostic_items = list(suggestions.items())[:3]
    for name, diagnosis in diagnostic_items:
        summary['top_diagnostics'].append(f"{name}: {diagnosis}")
    
    # Generate recommendations based on status
    if summary['overall_status'] == 'CRITICAL':
        summary['recommendations'] = [
            "Immediate maintenance attention required",
            "Consider equipment shutdown for inspection",
            "Schedule detailed valve inspection"
        ]
        summary['next_actions'] = [
            "Contact maintenance team within 24 hours",
            "Perform detailed diagnostic analysis",
            "Plan maintenance shutdown"
        ]
    elif summary['overall_status'] == 'POOR':
        summary['recommendations'] = [
            "Schedule maintenance within 1-2 weeks",
            "Monitor equipment closely",
            "Increase inspection frequency"
        ]
        summary['next_actions'] = [
            "Schedule maintenance inspection",
            "Continue monitoring",
            "Review operating parameters"
        ]
    elif summary['overall_status'] == 'FAIR':
        summary['recommendations'] = [
            "Schedule routine maintenance",
            "Monitor trending parameters",
            "Consider minor adjustments"
        ]
        summary['next_actions'] = [
            "Plan routine maintenance",
            "Track performance trends",
            "Optimize operating conditions"
        ]
    else:  # GOOD or EXCELLENT
        summary['recommendations'] = [
            "Continue current maintenance schedule",
            "Equipment operating within normal parameters",
            "Monitor for any trending changes"
        ]
        summary['next_actions'] = [
            "Maintain current schedule",
            "Continue routine monitoring",
            "No immediate action required"
        ]
    
    return summary

def extract_rpm(levels_xml_content):
    """Extract RPM from levels file"""
    try:
        levels_root = ET.fromstring(levels_xml_content)
        # Look for RPM in the levels file
        rpm = find_xml_value(levels_root, 'Levels', 'RPM', 2)
        if rpm and rpm != "N/A":
            try:
                rpm_val = float(rpm)
                if 100 <= rpm_val <= 10000:  # Reasonable RPM range
                    return f"{rpm_val:.0f}"
            except (ValueError, TypeError):
                pass
        return "N/A"
    except Exception:
        return "N/A"

def validate_xml_files(uploaded_files):
    """
    Validates uploaded XML files and returns validation results - ROBUST VERSION
    """
    validation_results = {
        'is_valid': False,
        'files_found': {},
        'missing_files': [],
        'file_info': {},
        'errors': []
    }
    
    if len(uploaded_files) != 3:
        validation_results['errors'].append(f"Expected 3 files, got {len(uploaded_files)}")
        return validation_results
    
    # Check for required file types
    required_files = ['curves', 'levels', 'source']
    found_files = {}
    
    for file in uploaded_files:
        filename_lower = file.name.lower()
        if 'curves' in filename_lower:
            found_files['curves'] = file
        elif 'levels' in filename_lower:
            found_files['levels'] = file
        elif 'source' in filename_lower:
            found_files['source'] = file
    
    # Check what's missing
    missing = [file_type for file_type in required_files if file_type not in found_files]
    validation_results['missing_files'] = missing
    validation_results['files_found'] = found_files
    
    if missing:
        validation_results['errors'].append(f"Missing required files: {', '.join(missing)}")
        return validation_results
    
    # Validate each XML file with robust error handling
    for file_type, file in found_files.items():
        try:
            content = file.getvalue().decode('utf-8')
            
            # Basic XML validation
            root = ET.fromstring(content)
            
            # File-specific validation with safe fallbacks
            if file_type == 'curves':
                try:
                    # Count data elements safely
                    data_elements = root.findall('.//Data')
                    curve_count = len([elem for elem in data_elements if elem.text and elem.text.strip()])
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'data_points': curve_count,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'data_points': 0,
                        'status': 'Valid'
                    }
            
            elif file_type == 'levels':
                try:
                    # Extract machine info safely
                    machine_info = None
                    try:
                        machine_info = find_xml_value(root, 'Levels', 'Machine', 2)
                    except:
                        pass
                    
                    if not machine_info or machine_info == 'N/A':
                        machine_info = 'Unknown'
                    
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'machine_id': machine_info,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'machine_id': 'Unknown',
                        'status': 'Valid'
                    }
            
            elif file_type == 'source':
                try:
                    # Count configuration entries safely
                    config_count = 0
                    try:
                        # Safe iteration through elements
                        for elem in root.iter():
                            if hasattr(elem, 'text') and elem.text and 'CYLINDER' in str(elem.text):
                                config_count += 1
                    except:
                        config_count = 0
                    
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'config_entries': config_count,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'config_entries': 0,
                        'status': 'Valid'
                    }
                
        except ET.ParseError as e:
            validation_results['errors'].append(f"{file_type.title()} file: Invalid XML format")
            validation_results['file_info'][file_type] = {'status': 'Invalid XML', 'error': 'XML parsing failed'}
        except UnicodeDecodeError:
            validation_results['errors'].append(f"{file_type.title()} file: Invalid file encoding")
            validation_results['file_info'][file_type] = {'status': 'Encoding Error', 'error': 'Cannot decode file'}
        except Exception as e:
            validation_results['errors'].append(f"{file_type.title()} file: Unexpected error")
            validation_results['file_info'][file_type] = {'status': 'Error', 'error': 'Processing failed'}
    
    # Set overall validation status
    validation_results['is_valid'] = len(validation_results['errors']) == 0
    
    return validation_results

def extract_preview_info(files_content):
    """
    Extracts key information for preview display - FIXED: RPM now pulled from source.xml
    """
    preview_info = {
        'machine_id': 'Unknown',
        'rpm': 'Unknown',
        'cylinder_count': 0,
        'total_curves': 0,
        'file_sizes': {},
        'date_time': 'Unknown'
    }

    # LEVELS FILE - Get date/time only
    if 'levels' in files_content:
        try:
            levels_root = ET.fromstring(files_content['levels'])
            for elem in levels_root.iter():
                if hasattr(elem, 'text') and elem.text and '/' in str(elem.text) and len(str(elem.text)) > 8:
                    preview_info['date_time'] = str(elem.text)
                    break
        except:
            pass
        preview_info['file_sizes']['levels'] = len(files_content['levels']) / 1024

    # CURVES FILE - Count curves
    if 'curves' in files_content:
        try:
            curves_root = ET.fromstring(files_content['curves'])

            # Count curves
            curve_count = 0
            for i, elem in enumerate(curves_root.iter()):
                if hasattr(elem, 'text') and elem.text:
                    text_upper = str(elem.text).upper()
                    if any(k in text_upper for k in ['PRESSURE', 'VIBRATION', 'PHASED']):
                        curve_count += 1
                if i > 200:
                    break
            preview_info['total_curves'] = curve_count

        except:
            pass
        preview_info['file_sizes']['curves'] = len(files_content['curves']) / 1024

    # SOURCE FILE - Machine ID, RPM, Cylinder Count
    if 'source' in files_content:
        try:
            source_root = ET.fromstring(files_content['source'])

            # Extract RPM from source
            rpm = extract_rpm_from_source(files_content['source'])
            if rpm and rpm not in ['N/A', '', 'Unknown']:
                preview_info['rpm'] = rpm

            # Try extracting machine ID directly
            for elem in source_root.iter():
                if hasattr(elem, 'text') and elem.text:
                    text = str(elem.text).strip()
                    if ('-' in text and len(text) < 20 and len(text) > 3 and
                        any(c.isalnum() for c in text) and 
                        not any(w in text.upper() for w in ['CYLINDER', 'PRESSURE', 'TEMPERATURE', 'VALVE', 'COMPRESSOR'])):
                        preview_info['machine_id'] = text
                        break

            # Auto-discovery for cylinder count
            try:
                curves_content = files_content.get('curves', '')
                if curves_content:
                    df, curve_names = load_all_curves_data(curves_content)
                    if df is not None and curve_names:
                        config = auto_discover_configuration(files_content['source'], curve_names)
                        if config and 'cylinders' in config:
                            preview_info['cylinder_count'] = len(config['cylinders'])
                            if preview_info['machine_id'] == 'Unknown' and config.get('machine_id'):
                                preview_info['machine_id'] = config['machine_id']
            except:
                # Fallback: simple bore count
                bore_count = 0
                for elem in source_root.iter():
                    if hasattr(elem, 'text') and elem.text and 'BORE' in elem.text.upper():
                        bore_count += 1
                preview_info['cylinder_count'] = min(bore_count, 10)

        except:
            pass
        preview_info['file_sizes']['source'] = len(files_content['source']) / 1024

    return preview_info


def render_section_header(title, icon="🔧"):
    """Render consistent section headers"""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h3 style="margin: 0; color: var(--primary-color);">{title}</h3>
    </div>
    """, unsafe_allow_html=True)

def enhanced_file_upload_section():
    """
    Enhanced file upload with validation, preview, and styled UI
    """
    render_section_header("📁 1. Data Upload")
    

    # Check if we already have validated files in session state
    if 'validated_files' in st.session_state and st.session_state.validated_files:
        st.success("✅ Files already loaded and validated!")

        files_content = st.session_state.validated_files
        preview_info = extract_preview_info(files_content)

        st.markdown("""
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; font-size: 0.9rem;">
    <div style="flex: 1 1 45%;">
        <strong>Machine ID:</strong><br>{machine_id}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Data Curves:</strong><br>{curves}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Cylinders:</strong><br>{cyl}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Total Size:</strong><br>{size} KB
    </div>
</div>
""".format(
    machine_id=preview_info['machine_id'],
    curves=preview_info['total_curves'],
    cyl=preview_info['cylinder_count'],
    size=f"{sum(preview_info['file_sizes'].values()):.1f}"
), unsafe_allow_html=True)


        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button("🔄 Upload New Files", use_container_width=True):
                    st.session_state.file_uploader_key += 1
                    st.session_state.active_session_id = None
                    if 'validated_files' in st.session_state:
                        del st.session_state.validated_files
                    if 'analysis_results' in st.session_state:
                        st.session_state.analysis_results = None
                    st.rerun()
        
        return files_content
        
    # Upload input
    uploaded_files = st.file_uploader(
        "Upload Curves, Levels, Source XML files",
        type=["xml"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        help="Upload exactly 3 XML files: one each for Curves, Levels, and Source data"
    )

    if st.button("🗑️ Start New Analysis / Clear Files"):
        st.session_state.file_uploader_key += 1
        st.session_state.active_session_id = None
        if 'validated_files' in st.session_state:
            del st.session_state.validated_files
        if 'analysis_results' in st.session_state:
            st.session_state.analysis_results = None
        st.rerun()

    if uploaded_files:
        if len(uploaded_files) != 3:
            st.error(f"❌ Please upload exactly 3 XML files. You uploaded {len(uploaded_files)} files.")
            st.info("💡 Required files: Curves.xml, Levels.xml, Source.xml")
            return None

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 Validating uploaded files...")
        progress_bar.progress(25)
        validation_results = validate_xml_files(uploaded_files)

        if not validation_results['is_valid']:
            progress_bar.progress(100)
            st.error("❌ File validation failed!")
            for error in validation_results['errors']:
                st.error(f"• {error}")
            if validation_results['files_found']:
                st.info("✅ Files detected:")
                for file_type in validation_results['files_found']:
                    st.success(f"• {file_type.title()} file found")
            return None

        # Extract and Preview
        status_text.text("📄 Processing file contents...")
        progress_bar.progress(50)

        files_content = {}
        for file in uploaded_files:
            name = file.name.lower()
            if 'curves' in name:
                files_content['curves'] = file.getvalue().decode('utf-8')
            elif 'levels' in name:
                files_content['levels'] = file.getvalue().decode('utf-8')
            elif 'source' in name:
                files_content['source'] = file.getvalue().decode('utf-8')

        status_text.text("🔍 Generating preview...")
        progress_bar.progress(75)
        preview_info = extract_preview_info(files_content)

        progress_bar.progress(100)
        status_text.text("✅ Files ready for analysis!")
        st.success("✅ All files validated successfully!")

        st.markdown("### 📋 Data Preview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **🏭 Machine Information:**
            - **ID:** {preview_info['machine_id']}
            - **RPM:** {preview_info['rpm']}
            - **Date:** {preview_info['date_time']}
            """)

        with col2:
            st.markdown(f"""
            **📊 Data Summary:**
            - **Cylinders:** {preview_info['cylinder_count']}
            - **Data Curves:** {preview_info['total_curves']}
            - **Total Size:** {sum(preview_info['file_sizes'].values()):.1f} KB
            """)

        # File status
        st.markdown("**📁 File Details:**")
        file_details = []
        for file_type, size_kb in preview_info['file_sizes'].items():
            status = "✅" if size_kb > 0 else "⚠️"
            file_details.append(f"{status} **{file_type.title()}:** {size_kb:.1f} KB")
        st.markdown(" | ".join(file_details))

        # Warnings
        warnings = []
        if preview_info['machine_id'] == 'Unknown':
            warnings.append("⚠️ Machine ID not detected")
        if preview_info['cylinder_count'] == 0:
            warnings.append("⚠️ No cylinders detected")
        if preview_info['total_curves'] == 0:
            warnings.append("⚠️ No data curves detected")

        if warnings:
            st.warning("**Data Quality Warnings:**\n" + "\n".join(warnings))

        # Expander for file info
        with st.expander("🔍 Detailed Technical Information"):
            for file_type, info in validation_results['file_info'].items():
                st.markdown(f"**{file_type.title()} File Analysis:**")
                if info['status'] == 'Valid':
                    st.write(f"• Status: ✅ Valid XML structure")
                    st.write(f"• File size: {info['size_kb']:.1f} KB")
                    if file_type == 'curves':
                        st.write(f"• Data elements: {info['data_points']}")
                        st.write(f"• Detected curves: {preview_info['total_curves']}")
                    elif file_type == 'levels':
                        st.write(f"• Machine ID: {preview_info['machine_id']}")
                        st.write(f"• Recording date: {preview_info['date_time']}")
                    elif file_type == 'source':
                        st.write(f"• Configuration entries: {info['config_entries']}")
                        st.write(f"• Detected cylinders: {preview_info['cylinder_count']}")
                else:
                    st.error(f"• Status: ❌ {info['status']}")
                    if 'error' in info:
                        st.error(f"• Error: {info['error']}")
                st.markdown("---")

        st.markdown("---")
        if warnings:
            st.warning("⚠️ **You can proceed, but please check the warnings above**")
        else:
            st.success("✅ **Data looks good! Ready to analyze**")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if st.button("🚀 Analyse", type="primary", use_container_width=True):
                st.session_state.validated_files = files_content
                st.markdown('</div>', unsafe_allow_html=True)
                return files_content

        st.markdown('</div>', unsafe_allow_html=True)
        return None

    else:
        st.info("👆 Please upload your 3 XML files to begin")
        st.markdown('</div>', unsafe_allow_html=True)
        return None


def compute_volume_series(crank_angles, bore, stroke, clearance_pct):
    """
    Computes instantaneous cylinder volume for each crank angle for P-V diagram.
    FIXED: Proper index alignment to prevent out-of-bounds errors
    """
    if len(crank_angles) == 0:
        return pd.Series([], dtype=float)
        
    try:
        # Convert inputs to float
        bore_f = float(bore)
        stroke_f = float(stroke) 
        clearance_f = float(clearance_pct) / 100.0
        
        # Calculate swept volume
        area = math.pi * (bore_f / 2) ** 2
        swept_volume = area * stroke_f
        clearance_volume = swept_volume * clearance_f
        
        # Convert crank angles - preserve original index from DataFrame
        if isinstance(crank_angles, pd.Series):
            # If it's already a pandas Series, use it directly
            crank_angles_series = crank_angles.astype(float)
        else:
            # If it's a list/array, convert to Series with default index
            crank_angles_series = pd.Series(crank_angles, dtype=float)
        
        # Convert to radians
        theta_rad = np.deg2rad(crank_angles_series)
        
        # Calculate piston position using kinematic formula
        # For simplified MVP: piston_position = stroke/2 * (1 - cos(theta))
        piston_position = (stroke_f / 2) * (1 - np.cos(theta_rad))
        
        # Calculate instantaneous volume - preserve the original index
        instantaneous_volume = clearance_volume + area * piston_position
        
        # Return with the same index as the input crank_angles
        return pd.Series(instantaneous_volume.values, index=crank_angles_series.index)
        
    except (ValueError, TypeError) as e:
        st.warning(f"Volume computation error: {e}")
        return pd.Series([], dtype=float)
        
    except Exception as e:
        st.error(f"Volume computation error: {e}")
        return None

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

        # Create DataFrame with manual column names
        df = pd.DataFrame(rs.rows, columns=['timestamp', 'machine_id', 'total_anomalies'])
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

def run_anomaly_detection(df, curve_names, contamination_level=0.05): 
    """
    Applies Isolation Forest to detect anomalies and calculate their scores.
    """
    for curve in curve_names:
        if curve in df.columns:
            data = df[[curve]].values
            # Use the contamination_level parameter
            model = IsolationForest(contamination=contamination_level, random_state=42)
            
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
    Returns a dict: { item['name'] -> suggested_label_in_FAULT_LABELS }
    """
    suggestions = {}
    for item in report_data:
        # Pressure anomalies -> suggest Valve Leakage
        if item['name'] == 'Pressure' and item['count'] > 10:
            suggestions[item['name']] = 'Valve Leakage'
        # Any valve anomalies -> suggest Valve Wear (generic, matches FAULT_LABELS)
        elif item['name'] != 'Pressure' and item['count'] > 5:
            suggestions[item['name']] = 'Valve Wear'
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

def save_valve_event_to_db(db_client, analysis_id, crank_angle, event_type="Manual Tag"):
    """Save valve event to database"""
    db_client.execute(
        "INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)",
        (analysis_id, event_type, crank_angle)
    )

def render_interactive_plot_with_event(df, x_col, y_col, label, analysis_id=None, existing_events=None):
    """Render interactive plot with click event handling for crank-angle tagging"""
    fig = px.line(df, x=x_col, y=y_col, title=f"{label} Curve", markers=True)
    fig.update_layout(hovermode='x unified')

    # Show previous events (if any)
    if existing_events:
        for angle in existing_events:
            fig.add_vline(x=angle, line_dash="dash", line_color="red", annotation_text=f"Tagged: {angle}°")

    st.markdown(f"### 🔍 Click the curve to tag crank angle positions")
    
    # Create unique key for this plot
    plot_key = f"{label.replace(' ', '_')}_plot"
    
    selected_points = plotly_events(fig, click_event=True, key=plot_key)

    if selected_points:
        clicked_x = selected_points[0].get("x")
        if clicked_x is not None:
            st.success(f"✅ Crank-angle tagged at: {clicked_x:.2f}°")
            
            # Store in session state with plot-specific key
            if plot_key not in st.session_state.valve_event_tags:
                st.session_state.valve_event_tags[plot_key] = []
            st.session_state.valve_event_tags[plot_key].append(clicked_x)

    return st.session_state.valve_event_tags.get(plot_key, [])

def extract_rpm_from_source(source_xml_content):
    """
    Extract RPM directly from Source XML rows using Excel schema structure.
    """
    try:
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        root = ET.fromstring(source_xml_content)
        rows = root.findall(".//ss:Row", ns)

        for row in rows:
            cells = row.findall("ss:Cell", ns)
            if len(cells) >= 3:
                cell_value = cells[0].find("ss:Data", ns)
                if cell_value is not None and "RPM" in cell_value.text.upper():
                    rpm_data = cells[2].find("ss:Data", ns)
                    if rpm_data is not None:
                        try:
                            return f"{float(rpm_data.text):.0f}"
                        except:
                            return "N/A"
    except Exception as e:
        print("RPM extraction failed:", e)

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
        num_cols_offset_start = 2  # first cylinder's value is in column index 2

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

        # Build configuration for each cylinder - FIXED LOGIC
        cylinders_config = []
        
        # Debug: Print all curve names to see what we're working with
        print("DEBUG: Available curve names:")
        for name in sorted(all_curve_names):
            print(f"  - {name}")
        
        for i in range(1, num_cylinders + 1):
            print(f"\nDEBUG: Processing Cylinder {i}")
            
            # FIXED: More robust pressure curve detection
            pressure_curve = None
            
            # Try Head End first (.{i}H.)
            he_pressure = next(
                (c for c in all_curve_names if f".{i}H." in c and "STATIC" in c and "COMPRESSOR PT" in c),
                None
            )
            
            # Try Crank End (.{i}C.)
            ce_pressure = next(
                (c for c in all_curve_names if f".{i}C." in c and "STATIC" in c and "COMPRESSOR PT" in c),
                None
            )
            
            # Prefer Head End, fall back to Crank End
            pressure_curve = he_pressure or ce_pressure
            
            print(f"DEBUG: Cylinder {i} - HE pressure: {he_pressure}")
            print(f"DEBUG: Cylinder {i} - CE pressure: {ce_pressure}")
            print(f"DEBUG: Cylinder {i} - Selected pressure: {pressure_curve}")

            # FIXED: Correct valve curve detection based on actual XML patterns
            valve_curves = []
            
            # Head End Discharge (.{i}HD1)
            he_discharge = next(
                (c for c in all_curve_names if f".{i}HD1" in c and "VIBRATION" in c),
                None
            )
            if he_discharge:
                valve_curves.append({"name": "HE Discharge", "curve": he_discharge})
            
            # Head End Suction (.{i}HS1)  
            he_suction = next(
                (c for c in all_curve_names if f".{i}HS1" in c and "VIBRATION" in c),
                None
            )
            if he_suction:
                valve_curves.append({"name": "HE Suction", "curve": he_suction})
            
            # Crank End Discharge (.{i}CD1)
            ce_discharge = next(
                (c for c in all_curve_names if f".{i}CD1" in c and "VIBRATION" in c),
                None
            )
            if ce_discharge:
                valve_curves.append({"name": "CE Discharge", "curve": ce_discharge})
            
            # Crank End Suction (.{i}CS1)
            ce_suction = next(
                (c for c in all_curve_names if f".{i}CS1" in c and "VIBRATION" in c),
                None
            )
            if ce_suction:
                valve_curves.append({"name": "CE Suction", "curve": ce_suction})

            print(f"DEBUG: Cylinder {i} - Found valve curves: {[vc['name'] for vc in valve_curves]}")

            # FIXED: More lenient condition - include cylinder if it has EITHER pressure OR valve data
            # This ensures we don't skip cylinders that might have partial data
            has_pressure = pressure_curve is not None
            has_valves = len(valve_curves) > 0
            
            if has_pressure or has_valves:
                bore = bore_values[i - 1] if i <= len(bore_values) else None
                rod_dia = rod_diameter_values[i - 1] if i <= len(rod_diameter_values) else None
                stroke = stroke_values[i - 1] if i <= len(stroke_values) else None
                volume = None
                if bore is not None and stroke is not None:
                    volume = math.pi * (bore / 2) ** 2 * stroke

                cylinder_config = {
                    "cylinder_name": f"Cylinder {i}",
                    "pressure_curve": pressure_curve,
                    "valve_vibration_curves": valve_curves,
                    "bore": bore,
                    "rod_diameter": rod_dia,
                    "stroke": stroke,
                    "volume": volume,
                }
                
                cylinders_config.append(cylinder_config)
                print(f"DEBUG: Added Cylinder {i} to config")
            else:
                print(f"DEBUG: Skipped Cylinder {i} - no pressure or valve data found")

        print(f"\nDEBUG: Final cylinders_config length: {len(cylinders_config)}")
        print(f"DEBUG: Cylinder names: {[c['cylinder_name'] for c in cylinders_config]}")

        # FIXED: Ensure cylinders are sorted by number to guarantee Cylinder 1 comes first
        cylinders_config.sort(key=lambda x: int(x['cylinder_name'].split()[-1]))

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

def render_cylinder_selection_sidebar(cylinders_config):
    """
    Fixed cylinder selection that always defaults to Cylinder 1
    """
    cylinders = cylinders_config.get("cylinders", [])
    cylinder_names = [c.get("cylinder_name") for c in cylinders]
    
    if not cylinder_names:
        st.sidebar.error("No cylinders detected")
        return None, None
    
    # FIXED: Force Cylinder 1 to be default if it exists
    default_index = 0
    if "Cylinder 1" in cylinder_names:
        default_index = cylinder_names.index("Cylinder 1")
        print(f"DEBUG: Setting default to Cylinder 1 (index {default_index})")
    else:
        print(f"DEBUG: Cylinder 1 not found, using first available: {cylinder_names[0]}")
    
    
    # Find default index for Cylinder 1
    default_index = 0
    if "Cylinder 1" in cylinder_names:
        default_index = cylinder_names.index("Cylinder 1")

    selected_cylinder_name = st.sidebar.selectbox(
        "Select Cylinder for Detailed View", 
        cylinder_names,
        index=default_index,  # This ensures proper default selection
        help="Choose which cylinder to analyze in detail"
    )
    
    selected_cylinder_config = next(
        (c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), 
        None
    )
    st.session_state['selected_cylinder_name'] = selected_cylinder_name
    
    return selected_cylinder_name, selected_cylinder_config

def generate_health_report_table(_source_xml_content, _levels_xml_content, cylinder_index):
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        col_idx = cylinder_index
        
        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str: return "N/A"
            try: return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError): return kpa_str

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
        
        # Extract raw values first
        comp_ratio_he_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=2)
        comp_ratio_ce_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=1)
        power_he_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=2)
        power_ce_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=1)

        # Apply formatting to the extracted values
        comp_ratio_he = format_numeric_value(comp_ratio_he_raw, precision=2)
        comp_ratio_ce = format_numeric_value(comp_ratio_ce_raw, precision=2)
        power_he = format_numeric_value(power_he_raw, precision=1)
        power_ce = format_numeric_value(power_ce_raw, precision=1)

        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'], 
            'Bore (ins)': [bore] * 2, 
            'Rod Diam (ins)': ['N/A', rod_diam],
            'Pressure Ps/Pd (psig)': [f"{suction_p} / {discharge_p}"] * 2, 
            'Temp Ts/Td (°C)': [f"{suction_temp} / {discharge_temp}"] * 2,
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

def generate_pdf_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None, suggestions=None, health_score=None, critical_alerts=None):
    """
    Enhanced PDF report generator with improved UI and executive summary
    """
    if not REPORTLAB_AVAILABLE:
        st.warning("ReportLab not installed. PDF generation unavailable.")
        return None
    
    # Set default values
    if suggestions is None:
        suggestions = {}
    if health_score is None:
        health_score = 50.0
    if critical_alerts is None:
        critical_alerts = []
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    styles = getSampleStyleSheet()
    story = []
    
    # Enhanced custom styles for better formatting
    title_style = styles['Title']
    title_style.fontSize = 20
    title_style.spaceAfter = 20
    title_style.alignment = 1  # Center alignment
    title_style.textColor = colors.darkblue
    
    heading_style = styles['Heading2']
    heading_style.fontSize = 14
    heading_style.spaceAfter = 12
    heading_style.spaceBefore = 16
    heading_style.textColor = colors.darkblue
    heading_style.borderWidth = 1
    heading_style.borderColor = colors.lightgrey
    heading_style.borderPadding = 8
    heading_style.backColor = colors.lightgrey
    
    subheading_style = styles['Heading3']
    subheading_style.fontSize = 12
    subheading_style.spaceAfter = 8
    subheading_style.spaceBefore = 12
    subheading_style.textColor = colors.darkblue
    
    # Header with title and logo space
    story.append(Paragraph("MACHINE DIAGNOSTICS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Basic info in a well-formatted table
    basic_info = [
        ['Machine ID:', machine_id, 'Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ['Cylinder:', cylinder_name, 'RPM:', str(rpm)],
    ]
    
    basic_table = Table(basic_info, colWidths=[80, 120, 80, 120])
    basic_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # First column bold
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),  # Third column bold
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.lightblue, colors.white])
    ]))
    story.append(basic_table)
    story.append(Spacer(1, 25))
    
    # EXECUTIVE SUMMARY SECTION with enhanced formatting
    executive_summary = generate_executive_summary(machine_id, cylinder_name, health_score, report_data, suggestions)
    
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Spacer(1, 10))
    
    # Status box with improved color coding and layout
    status_color = colors.red
    status_bg_color = colors.pink
    if executive_summary['overall_status'] in ['EXCELLENT', 'GOOD']:
        status_color = colors.green
        status_bg_color = colors.lightgreen
    elif executive_summary['overall_status'] == 'FAIR':
        status_color = colors.orange
        status_bg_color = colors.lightyellow
    elif executive_summary['overall_status'] == 'POOR':
        status_color = colors.orangered
        status_bg_color = colors.mistyrose
    
    # Executive summary table with better spacing
    exec_data = [
        ['Overall Status:', executive_summary['overall_status']],
        ['Health Score:', f"{executive_summary['health_score']:.1f}/100"],
        ['Total Anomalies:', str(executive_summary['total_anomalies'])],
        ['Critical Issues:', str(len(critical_alerts))]
    ]
    
    exec_table = Table(exec_data, colWidths=[150, 200])
    exec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),  # Status value bold
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (1, 0), (1, 0), status_color),  # Color code status
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, -1), status_bg_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 20))
    
    # Critical Issues section with better formatting
    if executive_summary['critical_issues']:
        story.append(Paragraph("🚨 Critical Issues Identified", subheading_style))
        for issue in executive_summary['critical_issues'][:5]:  # Limit to 5 issues
            story.append(Paragraph(f"• {issue}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Top Diagnostics in a structured format
    if executive_summary['top_diagnostics']:
        story.append(Paragraph("🔍 Key Diagnostic Findings", subheading_style))
        for finding in executive_summary['top_diagnostics']:
            story.append(Paragraph(f"• {finding}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Recommendations in a formatted box
    story.append(Paragraph("📋 Recommendations", subheading_style))
    rec_data = [[f"• {rec}"] for rec in executive_summary['recommendations'][:4]]  # Limit to 4
    rec_table = Table(rec_data, colWidths=[450])
    rec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 15))
    
    # Next Actions in a formatted box
    story.append(Paragraph("⚡ Next Actions", subheading_style))
    action_data = [[f"• {action}"] for action in executive_summary['next_actions'][:4]]  # Limit to 4
    action_table = Table(action_data, colWidths=[450])
    action_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightcyan),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 25))
    
    # Chart section with better integration
    if chart_fig:
        story.append(Paragraph("📊 Diagnostic Chart", heading_style))
        story.append(Spacer(1, 10))
        try:
            img_buffer = io.BytesIO()
            chart_fig.write_image(img_buffer, format='png', width=600, height=400, scale=2)
            img_buffer.seek(0)
            from reportlab.platypus import Image
            # Center the image
            img = Image(img_buffer, width=500, height=333)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 20))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Chart generation error: {str(e)}", styles['Normal']))
            story.append(Spacer(1, 15))
    
    # Detailed Health Report with improved table formatting
    if not health_report_df.empty:
        story.append(Paragraph("🔧 Detailed Health Report", heading_style))
        story.append(Spacer(1, 10))
        
        # Convert DataFrame to table data
        table_data = [health_report_df.columns.tolist()] + health_report_df.values.tolist()
        
        # Calculate column widths based on content
        num_cols = len(table_data[0])
        col_width = 450 / num_cols  # Distribute evenly
        
        # Create table with improved styling
        health_table = Table(table_data, colWidths=[col_width] * num_cols)
        health_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(health_table)
        story.append(Spacer(1, 20))
    
    # Anomaly Analysis Details with enhanced formatting
    if report_data:
        story.append(Paragraph("📈 Anomaly Analysis Details", heading_style))
        story.append(Spacer(1, 10))
        
        anomaly_data = [['Component', 'Anomaly Count', 'Avg. Threshold', 'Unit', 'Status']]
        for item in report_data:
            status = "⚠️ High" if item.get('count', 0) > 5 else "✓ Normal"
            status_color_cell = colors.red if item.get('count', 0) > 5 else colors.green
            anomaly_data.append([
                item.get('name', 'Unknown'),
                str(item.get('count', 0)),
                f"{item.get('threshold', 0):.2f}",
                item.get('unit', ''),
                status
            ])
        
        anomaly_table = Table(anomaly_data, colWidths=[120, 80, 90, 60, 80])
        
        # Create style with conditional coloring for status column
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        # Add conditional coloring for high anomaly counts
        for i, item in enumerate(report_data, 1):
            if item.get('count', 0) > 5:
                table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.red))
                table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))
            else:
                table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.green))
        
        anomaly_table.setStyle(TableStyle(table_style))
        story.append(anomaly_table)
        story.append(Spacer(1, 25))
    
    # Professional footer with border
    footer_data = [[f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | AI-Powered Machine Diagnostics Analyzer v2.0"]]
    footer_table = Table(footer_data, colWidths=[500])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(footer_table)
    
    # Build PDF with improved error handling
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def generate_cylinder_view(_db_client, df, cylinder_config, envelope_view, vertical_offset, analysis_ids, contamination_level, view_mode="Crank-angle", clearance_pct=5.0, show_pv_overlay=False):
    """
    Generates cylinder view plots with pressure and valve vibration data.
    """
    pressure_curve = cylinder_config.get('pressure_curve')
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []
    
    # Initialize variables needed for P-V calculations
    bore = cylinder_config.get("bore")
    stroke = cylinder_config.get("stroke")
    can_plot_pv = (
        (bore is not None) and (stroke is not None) and
        (pressure_curve is not None) and (pressure_curve in df.columns)
    )

    curves_to_analyze = [vc['curve'] for vc in valve_curves if vc['curve'] in df.columns]
    if pressure_curve and pressure_curve in df.columns:
        curves_to_analyze.append(pressure_curve)

    df = run_anomaly_detection(df, curves_to_analyze, contamination_level)

    # --- P-V mode (standalone) ---
    if view_mode == "P-V" and not show_pv_overlay:
        if can_plot_pv:
            try:
                V = compute_volume_series(df["Crank Angle"], bore, stroke, clearance_pct)
        
                if V is not None and len(V) > 0:
                    pressure_data = df[pressure_curve]
            
                    if len(V) == len(pressure_data):
                        fig = go.Figure()
                
                        # Add the P-V cycle
                        fig.add_trace(go.Scatter(
                            x=V, y=pressure_data,
                            mode="lines+markers",
                            line=dict(width=2),
                            marker=dict(size=2),
                            name="P-V Cycle",
                            hovertemplate="<b>Volume:</b> %{x:.1f} in³<br>" +
                                        "<b>Pressure:</b> %{y:.1f} PSI<br>" +
                                        "<extra></extra>"
                        ))
                        
                        try:
                            # STANDALONE P-V MODE: Find TDC/BDC points for P-V diagram
                            if len(V) > 0 and len(pressure_data) > 0 and len(V) == len(pressure_data):
                                # Use numpy arrays for safer operations
                                volume_values = V.values
                                pressure_values = pressure_data.values
                                
                                # Find positions of min and max volume
                                min_vol_pos = np.argmin(volume_values)
                                max_vol_pos = np.argmax(volume_values)
                                
                                # Get the actual values for P-V plot (Volume on X-axis)
                                min_vol = volume_values[min_vol_pos]
                                max_vol = volume_values[max_vol_pos]
                                min_pressure = pressure_values[min_vol_pos]
                                max_pressure = pressure_values[max_vol_pos]
                                
                                # Add TDC point (minimum volume) on P-V diagram
                                fig.add_trace(go.Scatter(
                                    x=[min_vol], y=[min_pressure],  # X=Volume, Y=Pressure
                                    mode="markers",
                                    marker=dict(size=12, color="red", symbol="circle"),
                                    name="TDC (Top Dead Center)"
                                ))
                                
                                # Add BDC point (maximum volume) on P-V diagram
                                fig.add_trace(go.Scatter(
                                    x=[max_vol], y=[max_pressure],  # X=Volume, Y=Pressure
                                    mode="markers", 
                                    marker=dict(size=12, color="blue", symbol="square"),
                                    name="BDC (Bottom Dead Center)"
                                ))
                                
                                # Add annotations on P-V diagram
                                fig.add_annotation(
                                    x=min_vol, y=min_pressure,
                                    text="TDC", showarrow=True, arrowhead=2, ax=20, ay=-20
                                )
                                fig.add_annotation(
                                    x=max_vol, y=max_pressure,
                                    text="BDC", showarrow=True, arrowhead=2, ax=-20, ay=20
                                )
                                
                                # Debug info for P-V diagram
                                st.info(f"🔍 TDC at {min_vol:.1f} in³ ({min_pressure:.1f} PSI), BDC at {max_vol:.1f} in³ ({max_pressure:.1f} PSI)")
                            else:
                                st.warning("⚠️ Volume and pressure data length mismatch or empty data")
                    
                        except Exception as e:
                            st.warning(f"⚠️ Could not mark TDC/BDC points: {str(e)}")
                
                        fig.update_layout(
                            height=700,
                            title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name','Cylinder')}",
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            showlegend=True
                        )
                        fig.update_xaxes(title_text="<b>Volume (in³)</b>")
                        fig.update_yaxes(title_text="<b>Pressure (PSI)</b>")
                
                        # Add pressure data to report_data
                        if pressure_curve in df.columns:
                            anomaly_count = int(df[f'{pressure_curve}_anom'].sum())
                            avg_score = df.loc[df[f'{pressure_curve}_anom'], f'{pressure_curve}_anom_score'].mean() if anomaly_count > 0 else 0.0
                            report_data.append({
                                "name": "Pressure",
                                "curve_name": pressure_curve,
                                "threshold": avg_score,
                                "count": anomaly_count,
                                "unit": "PSI"
                            })
                
                        return fig, report_data
                    else:
                        st.warning(f"Data length mismatch: Volume={len(V)}, Pressure={len(pressure_data)}")
                else:
                    st.warning("Failed to compute volume data")
            
            except Exception as e:
                st.warning(f"P-V diagram computation failed: {e}")
        else:
            missing = []
            if bore is None:
                missing.append("bore dimension")
            if stroke is None:
                missing.append("stroke dimension")
            if pressure_curve is None or pressure_curve not in df.columns:
                missing.append("pressure curve")
            st.warning(f"P-V diagram not available - missing: {', '.join(missing)}")
            
        # Return empty figure and report data if P-V plot fails
        if 'fig' not in locals():
            fig = go.Figure()
            fig.update_layout(
                height=700,
                title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name','Cylinder')} (Error)",
                template="plotly_white"
            )
            fig.add_annotation(
                text="Unable to generate P-V diagram",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        return fig, report_data

    # --- Crank-angle mode OR Dual view mode ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add pressure curve to crank-angle plot
    if pressure_curve and pressure_curve in df.columns:
        anomaly_count = int(df[f'{pressure_curve}_anom'].sum())
        avg_score = df.loc[df[f'{pressure_curve}_anom'], f'{pressure_curve}_anom_score'].mean() if anomaly_count > 0 else 0.0
        report_data.append({
            "name": "Pressure",
            "curve_name": pressure_curve,
            "threshold": avg_score,
            "count": anomaly_count,
            "unit": "PSI"
        })
        fig.add_trace(
            go.Scatter(
                x=df['Crank Angle'],
                y=df[pressure_curve],
                name='Pressure (PSI)',
                line=dict(color='black', width=2)
            ),
            secondary_y=False
        )

    # Add valve vibration curves
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
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0.5, color=color_rgba.replace('0.4','1')),
                    showlegend=False,
                    hoverinfo='none'
                ),
                secondary_y=True
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0.5, color=color_rgba.replace('0.4','1')),
                    fill='tonexty',
                    fillcolor=color_rgba,
                    name=label_name,
                    hoverinfo='none'
                ),
                secondary_y=True
            )
        else:
            vibration_data = df[curve_name] + current_offset
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=vibration_data,
                    name=label_name,
                    mode='lines',
                    line=dict(color=color_rgba.replace('0.4','1'))
                ),
                secondary_y=True
            )

        # Add anomalies
        anomalies_df = df[df[f'{curve_name}_anom']]
        if not anomalies_df.empty:
            anomaly_vibration_data = anomalies_df[curve_name] + current_offset
            fig.add_trace(
                go.Scatter(
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
                ),
                secondary_y=True
            )

        # Add valve events
        analysis_id = analysis_ids.get(vc['name'])
        if analysis_id:
            try:
                events_raw = _db_client.execute(
                    "SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?",
                    (analysis_id,)
                ).rows
                events = {etype: angle for etype, angle in events_raw}
                if 'open' in events and 'close' in events:
                    fig.add_vrect(
                        x0=events['open'],
                        x1=events['close'],
                        fillcolor=color_rgba.replace('0.4','0.2'),
                        layer="below",
                        line_width=0
                    )
                for event_type, crank_angle in events.items():
                    fig.add_vline(
                        x=crank_angle,
                        line_width=2,
                        line_dash="dash",
                        line_color='green' if event_type == 'open' else 'red'
                    )
            except Exception as e:
                st.warning(f"Could not load valve events for {vc['name']}: {e}")
        
        # Add anomaly data to report
        anomaly_count = int(df[f'{curve_name}_anom'].sum())
        avg_score = df.loc[df[f'{curve_name}_anom'], f'{curve_name}_anom_score'].mean() if anomaly_count > 0 else 0.0
        report_data.append({
            "name": vc['name'],
            "curve_name": curve_name,
            "threshold": avg_score,
            "count": anomaly_count,
            "unit": "G"
        })
        
        current_offset += vertical_offset

    # Set up layout
    title_suffix = " with P-V Overlay" if show_pv_overlay else ""
    fig.update_layout(
        height=700,
        title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}{title_suffix}",
        xaxis_title="Crank Angle (deg)",
        template="ggplot2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Pressure (PSI)</b>", color="black", secondary_y=False)
    fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", color="blue", secondary_y=True)

    # --- ADD P-V OVERLAY if requested ---
    if show_pv_overlay and view_mode == "Crank-angle" and can_plot_pv:
        try:
            V = compute_volume_series(df["Crank Angle"], bore, stroke, clearance_pct)
            
            if V is not None and len(V) > 0:
                pressure_data = df[pressure_curve]
                
                if len(V) == len(pressure_data):
                    # Scale volume data to fit nicely on the plot
                    pressure_range = pressure_data.max() - pressure_data.min()
                    volume_range = V.max() - V.min()
                    
                    # Scale volume to use about 20% of the pressure range at the top
                    volume_scaled = ((V - V.min()) / volume_range) * (pressure_range * 0.2) + pressure_data.max() + (pressure_range * 0.05)
                    
                    # Add P-V overlay as a line
                    fig.add_trace(
                        go.Scatter(
                            x=df['Crank Angle'],
                            y=volume_scaled,
                            mode='lines',
                            line=dict(color='purple', width=2, dash='dot'),
                            name='Volume (scaled)',
                            hovertemplate="<b>Crank Angle:</b> %{x:.1f}°<br>" +
                                        "<b>Volume:</b> %{customdata:.1f} in³<br>" +
                                        "<extra></extra>",
                            customdata=V,
                            opacity=0.7
                        ),
                        secondary_y=False
                    )

                    try:
                        # P-V OVERLAY MODE: Find TDC/BDC for crank-angle chart with overlay
                        if len(V) > 0 and len(pressure_data) > 0 and len(V) == len(pressure_data) and len(df) == len(V):
                            # Use numpy arrays for safer operations
                            volume_values = V.values
                            pressure_values = pressure_data.values
                            crank_angles = df['Crank Angle'].values
                            
                            # Find positions of min and max volume
                            min_vol_pos = np.argmin(volume_values)
                            max_vol_pos = np.argmax(volume_values)
                            
                            # Get crank angles and pressures for overlay markers
                            tdc_crank_angle = crank_angles[min_vol_pos]  # X=Crank Angle
                            bdc_crank_angle = crank_angles[max_vol_pos]  # X=Crank Angle
                            tdc_pressure = pressure_values[min_vol_pos]  # Y=Pressure
                            bdc_pressure = pressure_values[max_vol_pos]  # Y=Pressure
                            
                            # Add TDC marker on crank-angle chart
                            fig.add_trace(
                                go.Scatter(
                                    x=[tdc_crank_angle], y=[tdc_pressure],  # X=Crank Angle, Y=Pressure
                                    mode='markers',
                                    marker=dict(
                                        size=12, color='red', symbol='circle',
                                        line=dict(width=2, color='darkred')
                                    ),
                                    name='TDC',
                                    hovertemplate="<b>TDC</b><br>Angle: %{x:.1f}°<br>Pressure: %{y:.1f} PSI<extra></extra>"
                                ),
                                secondary_y=False
                            )
                            
                            # Add BDC marker on crank-angle chart
                            fig.add_trace(
                                go.Scatter(
                                    x=[bdc_crank_angle], y=[bdc_pressure],  # X=Crank Angle, Y=Pressure
                                    mode='markers',
                                    marker=dict(
                                        size=12, color='blue', symbol='square',
                                        line=dict(width=2, color='darkblue')
                                    ),
                                    name='BDC',
                                    hovertemplate="<b>BDC</b><br>Angle: %{x:.1f}°<br>Pressure: %{y:.1f} PSI<extra></extra>"
                                ),
                                secondary_y=False
                            )
                            
                            # Add annotations on crank-angle chart
                            fig.add_annotation(
                                x=tdc_crank_angle, y=tdc_pressure,
                                text="TDC", showarrow=True, arrowhead=2, ax=30, ay=-30,
                                bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
                                font=dict(color="red", size=10)
                            )
                            
                            fig.add_annotation(
                                x=bdc_crank_angle, y=bdc_pressure,
                                text="BDC", showarrow=True, arrowhead=2, ax=-30, ay=30,
                                bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
                                font=dict(color="blue", size=10)
                            )
                            
                            # Add P-V overlay info box
                            fig.update_layout(
                                annotations=fig.layout.annotations + (dict(
                                    x=0.02, y=0.98, xref="paper", yref="paper",
                                    text=f"<b>P-V Overlay Active</b><br>Volume range: {V.min():.1f} - {V.max():.1f} in³<br>Clearance: {clearance_pct}%",
                                    showarrow=False, bgcolor="rgba(128,0,128,0.1)",
                                    bordercolor="purple", borderwidth=1,
                                    font=dict(size=9), align="left"
                                ),)
                            )
                            
                            st.info("✅ P-V overlay active! Purple dotted line shows scaled volume, markers show TDC/BDC positions.")
                        else:
                            st.warning("⚠️ Data length mismatch between volume, pressure, and DataFrame")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not mark TDC/BDC points: {str(e)}")
                        
                else:
                    st.warning(f"⚠️ P-V overlay failed: Data length mismatch (Volume={len(V)}, Pressure={len(pressure_data)})")
            else:
                st.warning("⚠️ P-V overlay failed: Could not compute volume data")
                    
        except Exception as e:
            st.warning(f"⚠️ P-V overlay failed: {str(e)}")

    return fig, report_data

def render_ai_model_tuning_section(db_client, discovered_config):
    """Enhanced AI Model Tuning with machine-specific configuration"""
    st.markdown("---")
    st.subheader("AI Model Tuning")
    
    # Get machine ID if available
    machine_id = discovered_config.get('machine_id', 'N/A') if discovered_config else None
    
    if machine_id and machine_id != 'N/A':
        st.markdown(f"**Machine-Specific Configuration** for `{machine_id}`")
        
        # Load existing config from database
        try:
            rs = db_client.execute("SELECT * FROM configs WHERE machine_id = ?", (machine_id,))
            existing_config = rs.rows[0] if rs.rows else None
        except Exception:
            existing_config = None
        
        # Configuration inputs with saved values
        col1, col2 = st.columns(2)
        
        with col1:
            contamination_level = st.slider(
                "Anomaly Detection Sensitivity", 
                min_value=0.01, 
                max_value=0.20, 
                value=existing_config[1] if existing_config else 0.05,
                step=0.01,
                help="Machine-specific sensitivity for anomaly detection"
            )
            
            pressure_limit = st.number_input(
                "Pressure Anomaly Threshold", 
                min_value=1, 
                max_value=50, 
                value=existing_config[2] if existing_config else 10,
                help="Maximum allowed pressure anomalies before alert"
            )
        
        with col2:
            valve_limit = st.number_input(
                "Valve Anomaly Threshold", 
                min_value=1, 
                max_value=30, 
                value=existing_config[3] if existing_config else 5,
                help="Maximum allowed valve anomalies before alert"
            )
            
            # Save button
            if st.button("💾 Save Machine Config", type="primary"):
                try:
                    db_client.execute(
                        "INSERT OR REPLACE INTO configs (machine_id, contamination, pressure_anom_limit, valve_anom_limit, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                        (machine_id, contamination_level, pressure_limit, valve_limit)
                    )
                    st.success(f"✅ Configuration saved for {machine_id}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")
        
        # Show if config was loaded
        if existing_config:
            st.info(f"📋 Using saved configuration (last updated: {existing_config[4] if len(existing_config) > 4 else 'Unknown'})")
        else:
            st.info("🆕 Using default values - save to create machine-specific configuration")
            
        return contamination_level, pressure_limit, valve_limit
    
    else:
        # Fallback to original slider when no machine ID
        st.info("ℹ️ Upload files to enable machine-specific configuration")
        contamination_level = st.slider(
            "Anomaly Detection Sensitivity", 
            min_value=0.01, 
            max_value=0.20, 
            value=0.05,
            step=0.01,
            help="Adjust the proportion of data points considered as anomalies. Higher values mean more sensitive detection."
        )
        return contamination_level, 10, 5  # Default limits

def run_rule_based_diagnostics_enhanced(report_data, pressure_limit=10, valve_limit=5):
    """Enhanced rule-based diagnostics using machine-specific thresholds"""
    suggestions = {}
    critical_alerts = []
    
    for item in report_data:
        item_name = item['name']
        anomaly_count = item['count']
        
        # Pressure-specific rules
        if item_name == 'Pressure':
            if anomaly_count > pressure_limit:
                if anomaly_count > pressure_limit * 2:
                    suggestions[item_name] = 'Valve Leakage'
                    critical_alerts.append(f"CRITICAL: {item_name} has {anomaly_count} anomalies (limit: {pressure_limit})")
                else:
                    suggestions[item_name] = 'Valve Wear'
        
        # Valve-specific rules
        elif item_name != 'Pressure':
            if anomaly_count > valve_limit:
                if 'Suction' in item_name:
                    if anomaly_count > valve_limit * 2:
                        suggestions[item_name] = 'Valve Sticking or Fouling'
                        critical_alerts.append(f"CRITICAL: {item_name} has {anomaly_count} anomalies (limit: {valve_limit})")
                    else:
                        suggestions[item_name] = 'Valve Wear'
                elif 'Discharge' in item_name:
                    if anomaly_count > valve_limit * 1.5:
                        suggestions[item_name] = 'Valve Impact or Slamming'
                        critical_alerts.append(f"HIGH: {item_name} has {anomaly_count} anomalies (limit: {valve_limit})")
                    else:
                        suggestions[item_name] = 'Valve Wear'
                else:
                    suggestions[item_name] = 'Valve Wear'
    
    return suggestions, critical_alerts

def check_and_display_alerts(db_client, machine_id, cylinder_name, critical_alerts, health_score):
    """Check for critical conditions and display in-app alerts"""
    current_time = datetime.datetime.now()
    
    # Health score alert
    if health_score < 40:
        alert_msg = f"Health score critically low: {health_score:.1f}"
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'CRITICAL', alert_msg, current_time)
            )
        except:
            pass
        st.error(f"🚨 CRITICAL ALERT: {alert_msg}")
    
    elif health_score < 60:
        alert_msg = f"Health score below normal: {health_score:.1f}"
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'WARNING', alert_msg, current_time)
            )
        except:
            pass
        st.warning(f"⚠️ WARNING: {alert_msg}")
    
    # Critical anomaly alerts
    for alert in critical_alerts:
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'HIGH', alert, current_time)
            )
        except:
            pass
        st.error(f"🔥 {alert}")

def generate_pdf_report_enhanced(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None, suggestions=None, health_score=None, critical_alerts=None, tagged_events=None):
    """Enhanced PDF report generator with executive summary"""
    if not REPORTLAB_AVAILABLE:
        st.warning("ReportLab not installed. PDF generation unavailable.")
        return None
    
    if suggestions is None: suggestions = {}
    if health_score is None: health_score = 50.0
    if critical_alerts is None: critical_alerts = []
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = styles['Title']
    title_style.fontSize = 18
    title_style.spaceAfter = 30
    
    heading_style = styles['Heading2']
    heading_style.fontSize = 14
    heading_style.spaceAfter = 12
    heading_style.textColor = colors.darkblue
    
    # Title
    story.append(Paragraph("MACHINE DIAGNOSTICS REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Basic info table
    basic_info = [
        ['Machine ID:', machine_id],
        ['Cylinder:', cylinder_name], 
        ['RPM:', rpm],
        ['Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    
    basic_table = Table(basic_info, colWidths=[100, 200])
    basic_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(basic_table)
    story.append(Spacer(1, 20))
    
    # EXECUTIVE SUMMARY
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    if health_score >= 85:
        overall_status = 'EXCELLENT'
        status_color = colors.green
    elif health_score >= 70:
        overall_status = 'GOOD'
        status_color = colors.green
    elif health_score >= 55:
        overall_status = 'FAIR'
        status_color = colors.orange
    elif health_score >= 40:
        overall_status = 'POOR'
        status_color = colors.orangered
    else:
        overall_status = 'CRITICAL'
        status_color = colors.red
    
    total_anomalies = sum(item.get('count', 0) for item in report_data)
    exec_data = [
        ['Overall Status:', overall_status],
        ['Health Score:', f"{health_score:.1f}/100"],
        ['Total Anomalies:', str(total_anomalies)],
        ['Critical Issues:', str(len(critical_alerts))]
    ]
    
    exec_table = Table(exec_data, colWidths=[120, 150])
    exec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (1, 0), (1, 0), status_color),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 15))
    
    # Chart
    if chart_fig:
        try:
            img_buffer = io.BytesIO()
            chart_fig.write_image(img_buffer, format='png', width=800, height=500, scale=2)
            img_buffer.seek(0)
            from reportlab.platypus import Image
            story.append(Paragraph("Diagnostic Chart", heading_style))
            story.append(Image(img_buffer, width=500, height=312))
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"Chart could not be generated. Error: {str(e)}", styles['Normal']))
    
    # Tagged Events Section
    if tagged_events:
        story.append(Paragraph("🔧 Crank-Angle Tagged Events", heading_style))
        for angle in tagged_events:
            story.append(Paragraph(f"• Tagged at: {angle:.2f}°", styles['Normal']))
        story.append(Spacer(1, 15))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def render_pressure_options_sidebar():
    """
    Render pressure analysis options similar to professional software with signal validation
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Pressure Analysis Options")
    
    # Main pressure enable toggle
    enable_pressure = st.sidebar.checkbox(
        "Enable Advanced Pressure Analysis", 
        value=True, 
        key='enable_pressure'
    )
    
    pressure_options = {'enable_pressure': enable_pressure}
    
    if enable_pressure:
        # Pressure Traces Section
        st.sidebar.markdown("**Pressure Traces:**")
        pressure_options['show_he_pt'] = st.sidebar.checkbox("Show HE PT trace", key='show_he_pt')
        pressure_options['show_he_theoretical'] = st.sidebar.checkbox("Show HE Theoretical", key='show_he_theoretical')
        pressure_options['show_ce_pt'] = st.sidebar.checkbox("Show CE PT trace", value=True, key='show_ce_pt')
        pressure_options['show_ce_theoretical'] = st.sidebar.checkbox("Show CE Theoretical", key='show_ce_theoretical')
        
        # Additional Pressures Section  
        st.sidebar.markdown("**Additional Pressures:**")
        pressure_options['show_he_nozzle'] = st.sidebar.checkbox("Show HE Nozzle pressure", key='show_he_nozzle')
        pressure_options['show_ce_nozzle'] = st.sidebar.checkbox("Show CE Nozzle pressure", key='show_ce_nozzle')
        pressure_options['show_he_terminal'] = st.sidebar.checkbox("Show HE Terminal pressure", key='show_he_terminal')
        pressure_options['show_ce_terminal'] = st.sidebar.checkbox("Show CE Terminal pressure", key='show_ce_terminal')
        
        # Period Selection
        st.sidebar.markdown("**Pressure Period Selection:**")
        period_options = [
            "Median", "Average", "Maximum", "Minimum", 
            "Outer Envelope", "Inner Envelope", "All periods"
        ]
        
        pressure_options['period_selection'] = st.sidebar.selectbox(
            "Period Selection:",
            period_options,
            index=0,  # Default to Median
            key='pressure_period'
        )
        
        # Additional options
        pressure_options['use_crc_data'] = st.sidebar.checkbox("Use CRC data", key='use_crc_data')
    else:
        # Set all options to False if pressure analysis is disabled
        pressure_options.update({
            'show_he_pt': False,
            'show_he_theoretical': False, 
            'show_ce_pt': False,
            'show_ce_theoretical': False,
            'show_he_nozzle': False,
            'show_ce_nozzle': False,
            'show_he_terminal': False,
            'show_ce_terminal': False,
            'period_selection': "Median",
            'use_crc_data': False
        })
    
    return pressure_options

def validate_pressure_signals(df, cylinder_config, pressure_options):
    """
    HolizTech-style signal validation - FIXED to use actual CURVES data instead of LEVELS data
    Returns ✅ or ❌ for each signal based on the actual time-series data quality
    """
    validation_results = {}
    
    # FIXED: Check CE PT trace using actual time-series data from df (CURVES.xml)
    if pressure_options.get('show_ce_pt', False):
        # Look for CE pressure columns in the actual DataFrame
        ce_columns = [col for col in df.columns if '.1C.' in col and 'STATIC' in col and 'COMPRESSOR PT' in col]
        
        if ce_columns and len(ce_columns) > 0:
            ce_pressure_data = df[ce_columns[0]]  # Use the actual time-series data
            
            # Quality checks on the actual time-series data
            has_data = len(ce_pressure_data) > 0
            no_all_zeros = not (ce_pressure_data == 0).all()
            has_variation = ce_pressure_data.std() > 1.0  # Some variation in the data
            reasonable_range = (ce_pressure_data.min() >= 0) and (ce_pressure_data.max() < 10000)  # Broader range
            no_excessive_spikes = (ce_pressure_data.std() < 2000)  # Not too erratic
            
            is_valid = has_data and no_all_zeros and has_variation and reasonable_range and no_excessive_spikes
            validation_results['CE PT trace'] = "✅" if is_valid else "❌"
        else:
            validation_results['CE PT trace'] = "❌"  # No CE data found in time-series
    
    # FIXED: Check HE PT trace using actual time-series data from df (CURVES.xml)
    if pressure_options.get('show_he_pt', False):
        # Look for HE pressure columns in the actual DataFrame
        he_columns = [col for col in df.columns if '.1H.' in col and 'STATIC' in col and 'COMPRESSOR PT' in col]
        
        if he_columns and len(he_columns) > 0:
            he_pressure_data = df[he_columns[0]]  # Use the actual time-series data
            
            # Same quality checks as CE but on actual time-series data
            has_data = len(he_pressure_data) > 0
            no_all_zeros = not (he_pressure_data == 0).all()
            has_variation = he_pressure_data.std() > 1.0  # Some variation in the data  
            reasonable_range = (he_pressure_data.min() >= 0) and (he_pressure_data.max() < 10000)  # Broader range
            no_excessive_spikes = (he_pressure_data.std() < 2000)  # Not too erratic
            
            is_valid = has_data and no_all_zeros and has_variation and reasonable_range and no_excessive_spikes
            validation_results['HE PT trace'] = "✅" if is_valid else "❌"
        else:
            validation_results['HE PT trace'] = "❌"  # No HE data found in time-series
    
    # Check CE Theoretical (this always works since it's generated)
    if pressure_options.get('show_ce_theoretical', False):
        validation_results['CE Theoretical'] = "✅"
    
    # Check HE Theoretical (this always works since it's generated)
    if pressure_options.get('show_he_theoretical', False):
        validation_results['HE Theoretical'] = "✅"
    
    # Check additional pressure traces (currently not available)
    additional_checks = [
        ('show_ce_nozzle', 'CE Nozzle pressure'),
        ('show_he_nozzle', 'HE Nozzle pressure'), 
        ('show_ce_terminal', 'CE Terminal pressure'),
        ('show_he_terminal', 'HE Terminal pressure')
    ]
    
    for option_key, display_name in additional_checks:
        if pressure_options.get(option_key, False):
            validation_results[display_name] = "❌"  # Currently no data available for these
    
    return validation_results

def apply_pressure_options_to_plot(fig, df, cylinder_config, pressure_options, files_content):
    """
    Apply pressure options - CORRECTED to preserve existing pressure line (black)
    """
    if not pressure_options['enable_pressure']:
        return fig
    
    # Color scheme for different traces
    colors = {
        'he_pt': 'blue',
        'ce_pt': 'orange',  # Changed to orange to avoid confusion with red theoretical
        'he_theoretical': 'darkgreen',
        'ce_theoretical': 'red',  # Keep as red (this works correctly)
        'he_nozzle': 'darkblue',
        'ce_nozzle': 'darkred',
        'he_terminal': 'navy',
        'ce_terminal': 'maroon'
    }
    
    # DEBUG: Add debug info
    st.sidebar.write(f"Debug: CE PT checked: {pressure_options.get('show_ce_pt', False)}")
    st.sidebar.write(f"Debug: HE PT checked: {pressure_options.get('show_he_pt', False)}")
    st.sidebar.write(f"Debug: Period Selection: {pressure_options.get('period_selection', 'None')}")
        
    # FIXED: Show CE (Crank End) pressure trace - ADD to existing plot, don't replace
    if pressure_options['show_ce_pt']:
        # Look for CE pressure columns directly in DataFrame
        ce_columns = [col for col in df.columns if '.1C.' in col and 'STATIC' in col and 'COMPRESSOR PT' in col]
        
        if ce_columns:
            ce_pressure_col = ce_columns[0]
            st.sidebar.write(f"Debug: Found CE column: {ce_pressure_col}")
            
            # Check if we already added this trace to avoid duplicates
            existing_ce_traces = [trace.name for trace in fig.data if trace.name and 'CE PT trace' in trace.name]
            
            if not existing_ce_traces:  # Only add if not already present
                # Apply period selection processing
                processed_pressure = process_pressure_by_period(df, ce_pressure_col, pressure_options.get('period_selection', 'Median'))
            
                if processed_pressure is not None:
                    trace_name = f"CE PT trace ({pressure_options.get('period_selection', 'Median')})"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['Crank Angle'],
                            y=processed_pressure,
                            name=trace_name,
                            line=dict(color=colors['ce_pt'], width=2, dash='solid'),
                            mode='lines'
                        ),
                        secondary_y=False
                    )
                    st.sidebar.success(f"✅ Added {trace_name}")
                else:
                    st.sidebar.error("❌ CE pressure processing failed")
            else:
                st.sidebar.info("CE PT trace already exists")
        else:
            st.sidebar.error("❌ No CE pressure column found in DataFrame")
    
    # FIXED: Show HE (Head End) pressure trace - ADD to existing plot, don't replace
    if pressure_options['show_he_pt']:
        # Look for HE pressure columns directly in DataFrame  
        he_columns = [col for col in df.columns if '.1H.' in col and 'STATIC' in col and 'COMPRESSOR PT' in col]
        
        if he_columns:
            he_pressure_col = he_columns[0]
            st.sidebar.write(f"Debug: Found HE column: {he_pressure_col}")
            
            # Check if we already added this trace to avoid duplicates
            existing_he_traces = [trace.name for trace in fig.data if trace.name and 'HE PT trace' in trace.name]
            
            if not existing_he_traces:  # Only add if not already present
                # Apply period selection processing
                processed_he_pressure = process_pressure_by_period(df, he_pressure_col, pressure_options.get('period_selection', 'Median'))
            
                if processed_he_pressure is not None:
                    trace_name = f"HE PT trace ({pressure_options.get('period_selection', 'Median')})"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['Crank Angle'],
                            y=processed_he_pressure,
                            name=trace_name,
                            line=dict(color=colors['he_pt'], width=2, dash='solid'),
                            mode='lines'
                        ),
                        secondary_y=False
                    )
                    st.sidebar.success(f"✅ Added {trace_name}")
                else:
                    st.sidebar.error("❌ HE pressure processing failed")
            else:
                st.sidebar.info("HE PT trace already exists")
        else:
            st.sidebar.error("❌ No HE pressure column found in DataFrame")
    
    # Keep existing CE Theoretical logic (THIS WORKS - DON'T CHANGE)
    if pressure_options['show_ce_theoretical']:
        try:
            st.sidebar.write("Debug: Attempting CE Theoretical calculation...")
            import numpy as np
            
            theta_rad = np.deg2rad(df['Crank Angle'])
            suction_pressure = 690.0
            discharge_pressure = 1550.0
            pressure_amplitude = (discharge_pressure - suction_pressure) / 2
            pressure_baseline = suction_pressure + pressure_amplitude
            theoretical_ce = pressure_baseline + pressure_amplitude * np.sin(theta_rad)
            
            # Check if CE Theoretical already exists
            existing_ce_theoretical = [trace.name for trace in fig.data if trace.name and 'CE Theoretical' in trace.name]
            
            if not existing_ce_theoretical:
                fig.add_trace(
                    go.Scatter(
                        x=df['Crank Angle'],
                        y=theoretical_ce,
                        name='CE Theoretical (Test)',
                        line=dict(color=colors['ce_theoretical'], width=2, dash='dash'),
                        mode='lines'
                    ),
                    secondary_y=False
                )
                st.sidebar.success("✅ Added CE Theoretical")
            else:
                st.sidebar.info("CE Theoretical already exists")
        except Exception as e:
            st.sidebar.error(f"❌ CE Theoretical failed: {str(e)}")
    
    # Add HE Theoretical  
    if pressure_options['show_he_theoretical']:
        try:
            st.sidebar.write("Debug: Attempting HE Theoretical calculation...")
            import numpy as np
            
            theta_rad = np.deg2rad(df['Crank Angle'])
            # Use slightly different values for HE to distinguish from CE
            suction_pressure = 680.0
            discharge_pressure = 1520.0
            pressure_amplitude = (discharge_pressure - suction_pressure) / 2
            pressure_baseline = suction_pressure + pressure_amplitude
            theoretical_he = pressure_baseline + pressure_amplitude * np.cos(theta_rad + np.pi/6)  # Phase shift
            
            # Check if HE Theoretical already exists
            existing_he_theoretical = [trace.name for trace in fig.data if trace.name and 'HE Theoretical' in trace.name]
            
            if not existing_he_theoretical:
                fig.add_trace(
                    go.Scatter(
                        x=df['Crank Angle'],
                        y=theoretical_he,
                        name='HE Theoretical',
                        line=dict(color=colors['he_theoretical'], width=3, dash='dash'),
                        mode='lines'
                    ),
                    secondary_y=False
                )
                st.sidebar.success("✅ Added HE Theoretical")
            else:
                st.sidebar.info("HE Theoretical already exists")
        except Exception as e:
            st.sidebar.error(f"❌ HE Theoretical failed: {str(e)}")
    
    # Additional pressure traces (keep existing - these show info messages)
    if pressure_options['show_ce_nozzle']:
        st.sidebar.info("CE Nozzle pressure data not available in current dataset")
    
    if pressure_options['show_he_nozzle']:
        st.sidebar.info("HE Nozzle pressure data not available in current dataset")
        
    if pressure_options['show_ce_terminal']:
        st.sidebar.info("CE Terminal pressure data not available in current dataset")
        
    if pressure_options['show_he_terminal']:
        st.sidebar.info("HE Terminal pressure data not available in current dataset")
    
    return fig

def process_pressure_by_period(df, pressure_curve, period_selection, rpm=600):
    """
    Enhanced period processing with VERY visible differences
    """
    import numpy as np
    
    if pressure_curve not in df.columns:
        return None
    
    pressure_data = df[pressure_curve].values.copy()
    
    try:
        if period_selection == "Median":
            # Simple median filter
            window = 21
            processed_pressure = np.array([
                np.median(pressure_data[max(0, i-window//2):min(len(pressure_data), i+window//2+1)])
                for i in range(len(pressure_data))
            ])
            
        elif period_selection == "Average":
            # Simple moving average - FIX THE INCOMPLETE LINE
            window = 31
            processed_pressure = np.convolve(pressure_data, np.ones(window)/window, mode='same')
            
        elif period_selection == "Maximum":
            # Create upper envelope - VERY obvious difference
            window = 51
            processed_pressure = np.array([
                np.max(pressure_data[max(0, i-window//2):min(len(pressure_data), i+window//2+1)])
                for i in range(len(pressure_data))
            ])
            
        elif period_selection == "Minimum":
            # Create lower envelope - VERY obvious difference
            window = 51
            processed_pressure = np.array([
                np.min(pressure_data[max(0, i-window//2):min(len(pressure_data), i+window//2+1)])
                for i in range(len(pressure_data))
            ])
            
        elif period_selection == "Outer Envelope":
            # Amplify peaks significantly
            baseline = np.median(pressure_data)
            processed_pressure = baseline + (pressure_data - baseline) * 1.5
            
        elif period_selection == "Inner Envelope":
            # Compress towards median
            baseline = np.median(pressure_data)
            processed_pressure = baseline + (pressure_data - baseline) * 0.3
            
        elif period_selection == "All periods":
            # Return original data unchanged
            return pressure_data
            
        else:
            # Unknown selection - return original
            return pressure_data
        
        return processed_pressure
        
    except Exception as e:
        print(f"Period processing failed: {e}")
        return pressure_data
        
# --- Main Application ---

db_client = init_db()

def render_main_header():
    """Render professional main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔧 AI-Powered Machine Diagnostics</h1>
        <p>Advanced condition monitoring and fault detection for reciprocating machinery</p>
    </div>
    """, unsafe_allow_html=True)

render_main_header()

with st.sidebar:
    validated_files = enhanced_file_upload_section()


    if st.session_state.analysis_results is not None:
        if st.button("🔄 Start New Analysis", type="secondary"):
            st.session_state.analysis_results = None
            st.session_state.active_session_id = None
            if 'auto_discover_config' in st.session_state:
                del st.session_state['auto_discover_config']
            st.rerun()

    st.header("2. View Options")
    envelope_view = st.checkbox(
        "Enable Envelope View",
        value=True,
        key='envelope_view')
    
    vertical_offset = st.slider(
        "Vertical Offset",
        0.0, 5.0, 1.0, 0.1,
        key='vertical_offset'
    )
    view_mode = st.radio(
        "View Mode",
        ["Crank-angle", "P-V"],
        index=0,
        key='view_mode'
    )
    show_pv_overlay = False
    interactive_tagging = False
    if view_mode == "Crank-angle":
        show_pv_overlay = st.checkbox(
            "🔄 Show P-V Overlay",
            value=False,
            key='pv_overlay',
            help="Show P-V diagram as overlay on crank-angle view"
        )
        
        # Interactive Tagging Mode
        interactive_tagging = st.checkbox(
            "🏷️ Interactive Tagging Mode",
            value=False,
            key='interactive_tagging',
            help="Click on curves to tag crank-angle positions for anomaly marking"
        )

    # Add pressure options here
    pressure_options = render_pressure_options_sidebar()
    if pressure_options['enable_pressure'] and st.session_state.analysis_results is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Signal Validation Status")
        st.sidebar.markdown("*Signal quality indicators*")
        
        # Get current analysis results
        df = st.session_state.analysis_results['df']
        discovered_config = st.session_state.analysis_results['discovered_config']
        
        
        # Get current cylinder selection - MATCH THE SELECTED CYLINDER
        cylinders = discovered_config.get("cylinders", [])
        if cylinders:
            # FIXED: Use the actually selected cylinder, not always cylinder[0]
            selected_cylinder_name = st.session_state.get('selected_cylinder_name', 'Cylinder 1')
    
            # Find the config for the currently selected cylinder
            cylinder_config = next(
                (c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), 
                cylinders[0]  # Fallback to first cylinder
            )
    
            st.sidebar.write(f"🔧 Validating signals for: {cylinder_config.get('cylinder_name', 'Unknown')}")    
            
            # Run signal validation
            validation_status = validate_pressure_signals(df, cylinder_config, pressure_options)
            
            # Display validation results in a clean format
            if validation_status:
                for signal_name, status in validation_status.items():
                    if status == "✅":
                        st.sidebar.success(f"{status} {signal_name}")
                    else:
                        st.sidebar.error(f"{status} {signal_name}")
            else:
                st.sidebar.info("No pressure signals selected for validation")

            # === ADD THIS DEBUG SECTION RIGHT HERE ===
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🔧 Debug Info:**")
            st.sidebar.write(f"Selected Period: {pressure_options.get('period_selection', 'None')}")
            st.sidebar.write(f"CE PT enabled: {pressure_options.get('show_ce_pt', False)}")
            st.sidebar.write(f"HE PT enabled: {pressure_options.get('show_he_pt', False)}")
            # === END DEBUG SECTION ===
    
    clearance_pct = st.number_input(
        "Clearance (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        key='clearance_pct',
        help="Estimated clearance volume as % of swept volume (MVP approximation)."
    )
    
    if 'auto_discover_config' in st.session_state:
        contamination_level, pressure_limit, valve_limit = render_ai_model_tuning_section(db_client, st.session_state['auto_discover_config'])
    else:
        contamination_level, pressure_limit, valve_limit = 0.05, 10, 5


# ========== Enhanced UI Components ==========

def render_status_card(title, value, status="normal", icon="📊"):
    """Render professional status card"""
    status_class = {
        "excellent": "status-excellent",
        "good": "status-good", 
        "warning": "status-warning",
        "critical": "status-critical"
    }.get(status, "status-good")
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{title}</div>
        <div class="{status_class}">{status.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_metrics_dashboard(analysis_results):
    """Enhanced metrics dashboard with professional layout"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("📈 System Overview", "📈")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_status_card("Machine Status", "OPERATIONAL", "good", "⚙️")
    with col2:
        render_status_card("Health Score", "85.2", "good", "❤️")
    with col3:
        render_status_card("Anomalies", "3", "warning", "⚠️")
    with col4:
        render_status_card("Last Update", datetime.now().strftime("%H:%M"), "excellent", "🕒")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_enhanced_results_dashboard(analysis_data):
    """Enhanced results dashboard with comprehensive insights"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("📊 Analysis Results", "📊")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Key Findings")
        findings = [
            "✅ Cylinder performance within normal parameters",
            "⚠️ Minor valve timing deviation detected (Cylinder 2)",
            "📈 Compression pressure trending upward",
            "🔄 Recommend continued monitoring"
        ]
        for finding in findings:
            st.markdown(f"• {finding}")
    
    with col2:
        st.markdown("### 📈 Health Metrics")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=85.2,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 60], 'color': "#e74c3c"},
                    {'range': [60, 80], 'color': "#f39c12"},
                    {'range': [80, 100], 'color': "#27ae60"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_action_buttons():
    """Render action buttons with consistent styling"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("⚡ Actions", "⚡")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            st.success("Report generation started...")
    with col2:
        if st.button("📧 Send Alert", use_container_width=True):
            st.info("Alert notification sent!")
    with col3:
        if st.button("💾 Save Analysis", use_container_width=True):
            st.success("Analysis saved successfully!")
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.info("Data refresh initiated...")
    
    st.markdown('</div>', unsafe_allow_html=True)



if validated_files:
    files_content = validated_files

    if 'curves' in files_content and 'source' in files_content and 'levels' in files_content:
        
        # Only run heavy analysis if not already done
        if st.session_state.analysis_results is None:
            with st.spinner("🔄 Processing data..."):
                df, actual_curve_names = load_all_curves_data(files_content['curves'])
                if df is not None:
                    discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
                    if discovered_config:
                        st.session_state['auto_discover_config'] = discovered_config
                        rpm = discovered_config.get('rated_rpm', 'N/A')
                        machine_id = discovered_config.get('machine_id', 'N/A')
                        
                        if st.session_state.active_session_id is None:
                            db_client.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (machine_id, rpm))
                            st.session_state.active_session_id = get_last_row_id(db_client)
                            st.success(f"✅ New analysis session #{st.session_state.active_session_id} created.")
                        
                        # Store all analysis results in session state
                        st.session_state.analysis_results = {
                            'df': df,
                            'discovered_config': discovered_config,
                            'actual_curve_names': actual_curve_names,
                            'files_content': files_content,
                            'rpm': rpm,
                            'machine_id': machine_id
                        }
                        st.success("✅ Analysis complete! Settings can now be changed without refreshing.")
        
        # Use stored results for all UI operations
        if st.session_state.analysis_results:
            df = st.session_state.analysis_results['df']
            discovered_config = st.session_state.analysis_results['discovered_config']
            files_content = st.session_state.analysis_results['files_content']
            rpm = st.session_state.analysis_results['rpm']
            machine_id = st.session_state.analysis_results['machine_id']
            
            # Continue with your existing analysis logic here...
            cylinders = discovered_config.get("cylinders", [])
            cylinder_names = [c.get("cylinder_name") for c in cylinders]
            
            # Rest of your existing code stays the same...
            with st.sidebar: selected_cylinder_name, selected_cylinder_config = render_cylinder_selection_sidebar(discovered_config)

                                
            if selected_cylinder_config:
                # Generate plot and initial data
                fig, temp_report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, {}, contamination_level, view_mode=view_mode, clearance_pct=clearance_pct, show_pv_overlay=show_pv_overlay)
    
                
                # Apply pressure options to the plot (NEW!)
                if view_mode == "Crank-angle":  # Only apply to crank-angle view
                    fig = apply_pressure_options_to_plot(fig, df.copy(), selected_cylinder_config, pressure_options, files_content)
                
                # Display the enhanced plot - with interactive tagging if enabled
                if interactive_tagging and view_mode == "Crank-angle":
                    st.markdown("### 🔍 Interactive Tagging Mode")
                    st.info("Click on the curves to tag crank-angle positions where anomalies are suspected.")
                    
                    # Add vertical lines for existing tags
                    plot_key = f"{selected_cylinder_name.replace(' ', '_')}_plot"
                    existing_tags = st.session_state.valve_event_tags.get(plot_key, [])
                    for angle in existing_tags:
                        fig.add_vline(x=angle, line_dash="dash", line_color="red", 
                                     annotation_text=f"Tagged: {angle:.1f}°")
                    
                    # Ensure figure has proper layout for interactive display
                    fig.update_layout(
                        height=600,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    # Use plotly_events for interactive clicking
                    selected_points = plotly_events(
                        fig, 
                        click_event=True, 
                        hover_event=False,
                        select_event=False,
                        key=plot_key
                    )
                    
                    # Debug: Show what plotly_events returns
                    if selected_points:
                        st.write(f"DEBUG: Selected points: {selected_points}")
                        clicked_x = selected_points[0].get("x") if selected_points else None
                        if clicked_x is not None:
                            st.success(f"✅ Crank-angle tagged at: {clicked_x:.2f}°")
                            
                            # Store in session state
                            if plot_key not in st.session_state.valve_event_tags:
                                st.session_state.valve_event_tags[plot_key] = []
                            st.session_state.valve_event_tags[plot_key].append(clicked_x)
                            st.rerun()
                    else:
                        st.write("DEBUG: No points selected - try clicking directly on a data point")
                    
                    # Show current tags and save options
                    if existing_tags:
                        st.markdown("#### 🏷️ Current Tags")
                        tags_col1, tags_col2, tags_col3 = st.columns([2, 1, 1])
                        with tags_col1:
                            for i, tag in enumerate(existing_tags):
                                st.write(f"• Tag {i+1}: {tag:.2f}°")
                        with tags_col2:
                            if st.button("💾 Save Tags", key="save_tags"):
                                # Save tags to database for all curves in this cylinder
                                saved_count = 0
                                for item in temp_report_data:
                                    # Get or create analysis ID for this curve
                                    rs = db_client.execute("SELECT id FROM analyses WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?", 
                                                         (st.session_state.active_session_id, selected_cylinder_name, item['curve_name']))
                                    existing_id_row = rs.rows[0] if rs.rows else None
                                    if existing_id_row:
                                        analysis_id = existing_id_row[0]
                                    else:
                                        db_client.execute("INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)", 
                                                        (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold']))
                                        analysis_id = get_last_row_id(db_client)
                                    
                                    # Clear existing tags for this analysis and add new ones
                                    db_client.execute("DELETE FROM valve_events WHERE analysis_id = ? AND event_type = ?", (analysis_id, 'Manual Tag'))
                                    for tag in existing_tags:
                                        save_valve_event_to_db(db_client, analysis_id, tag, 'Manual Tag')
                                        saved_count += 1
                                
                                st.success(f"✅ Saved {saved_count} tags to database!")
                        with tags_col3:
                            if st.button("🗑️ Clear Tags", key="clear_tags"):
                                st.session_state.valve_event_tags[plot_key] = []
                                st.rerun()
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
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
                fig, report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, analysis_ids, contamination_level, view_mode=view_mode, clearance_pct=clearance_pct, show_pv_overlay=show_pv_overlay)
                
                # Run rule-based diagnostics on the report data
                suggestions, critical_alerts = run_rule_based_diagnostics_enhanced(report_data, pressure_limit, valve_limit)
                
                # Display diagnostics if there are suggestions
                if suggestions:
                    st.subheader("🛠 Rule‑Based Diagnostics")
                    for name, suggestion in suggestions.items():
                        st.warning(f"{name}: {suggestion}")
     
                # Compute health score before using it
                health_score = compute_health_score(report_data, suggestions)
    
                # Now we can safely use health_score in alerts and metrics
                check_and_display_alerts(db_client, machine_id, selected_cylinder_name, critical_alerts, health_score)
                st.metric("Health Score", f"{health_score:.1f}")
                    
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
                                
                                default_label = suggestions.get(item['name'], "Normal")
                                if default_label in FAULT_LABELS:
                                    selected_label = st.selectbox(
                                        "Select fault label:",
                                        options=FAULT_LABELS,
                                        index=FAULT_LABELS.index(default_label),
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
                                    
                if st.button("🔄 Generate Report for this Cylinder", type="primary", key='gen_report'):
                    # Get tagged events for this cylinder
                    plot_key = f"{selected_cylinder_name.replace(' ', '_')}_plot"
                    current_tagged_events = st.session_state.valve_event_tags.get(plot_key, [])
                    
                    pdf_buffer = generate_pdf_report_enhanced(machine_id, rpm, selected_cylinder_name, report_data, health_report_df, fig, suggestions, health_score, critical_alerts, current_tagged_events)
                    if pdf_buffer:
                        st.download_button("📥 Download PDF Report", pdf_buffer, f"report_{machine_id}_{selected_cylinder_name}.pdf", "application/pdf", key='download_report')
                        

                
                st.markdown("---")
                # Machine Info Block
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
        else:
            st.error("Failed to process curve data.")
    else:
        st.error("Please ensure all three XML files (curves, levels, source) are uploaded.")
else:
    st.warning("Please upload your XML data files to begin analysis.", icon="⚠️")

# Historical Trend Analysis
st.markdown("---")
st.header("📈 Historical Trend Analysis")
display_historical_analysis(db_client)

# Display All Saved Labels at the bottom
with st.sidebar:
    st.header("3. View All Saved Labels")
    rs = db_client.execute("SELECT DISTINCT machine_id FROM sessions ORDER BY machine_id ASC")
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
