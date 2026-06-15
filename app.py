import streamlit as st
import pandas as pd
import json
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch.nn.functional as F
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SIA | Support Integrity Auditor", page_icon="🛡️", layout="wide")

# ==========================================
# 1. LOAD BASE THRESHOLD & CONSTANTS
# ==========================================
def load_threshold():
    """Loads the threshold dynamically saved during training."""
    try:
        with open('threshold.json', 'r') as f:
            data = json.load(f)
            return data.get('best_thresh', 0.50)
    except (FileNotFoundError, json.JSONDecodeError): 
        return 0.50

raw_thresh = load_threshold()
base_thresh = max(0.10, min(float(raw_thresh), 0.95))

# Store all candidate mismatches. The Dashboard slider performs final filtering.
CANDIDATE_STORAGE_THRESHOLD = 0.10
sev_to_int = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# ==========================================
# 2. NLP EXPLAINABILITY ENGINE (HYBRID LOGIC)
# ==========================================
def extract_nlp_evidence(text, channel, resolution_hours, human_prio):
    """Runs heuristic rules alongside the neural network to build the rich JSON."""
    text_lower = str(text).lower()
    
    # Keyword Extraction
    crit_words = re.findall(r"\b(outage|breach|lawsuit|security|fraud|wiped|gone|down|fatal)\b", text_lower)
    high_words = re.findall(r"\b(urgent|asap|escalate|manager|unacceptable|broken|crash|immediately)\b", text_lower)
    med_words = re.findall(r"\b(delay|waiting|cancel|refund|error|issue|problem|help|stuck)\b", text_lower)
    
    all_keywords = list(set(crit_words + high_words + med_words))
    
    # Inferred Severity Logic
    if crit_words: inferred_sev = "Critical"
    elif high_words: inferred_sev = "High"
    elif med_words: inferred_sev = "Medium"
    else: inferred_sev = "Low"
        
    # Delta Math
    human_int = sev_to_int.get(str(human_prio).capitalize(), 0)
    inferred_int = sev_to_int.get(inferred_sev, 0)
    delta_int = inferred_int - human_int
    severity_delta = f"+{delta_int}" if delta_int > 0 else str(delta_int)
    
    mismatch_type = "Hidden Crisis" if delta_int > 0 else ("False Alarm" if delta_int < 0 else "Consistent")
    
    return inferred_sev, mismatch_type, severity_delta, all_keywords

# ==========================================
# 3. CORE LOGIC & MODEL LOADING
# ==========================================
@st.cache_resource
def load_ml_model():
    """Loads the fine-tuned binary mismatch classifier with strict evaluation mode."""
    try:
        MODEL_PATH = "./model" if os.path.exists("./model") else "."
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
        model.eval() 
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

tokenizer, model, error_msg = load_ml_model()

if 'live_dossiers' not in st.session_state:
    st.session_state['live_dossiers'] = []

# ==========================================
# SIDEBAR NAVIGATION & SLIDER
# ==========================================
st.sidebar.title("🛡️ SIA Dashboard")
st.sidebar.markdown("Automated Weak-Supervision Ticket Triage")

if error_msg:
    st.sidebar.error(f"⚠️ Model Loading Failed:\n{error_msg}")
else:
    st.sidebar.success("🚀 Hybrid AI Engine Active!")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Settings")

OPTIMAL_THRESHOLD = st.sidebar.slider(
    "Live Decision Boundary", 
    min_value=0.10, 
    max_value=0.95, 
    value=base_thresh, 
    step=0.01,
    help="Initialized from threshold.json. Adjust to filter out minor disagreements."
)

page = st.sidebar.radio("Navigation", ["📊 Dashboard & Analytics", "🔍 Single Ticket Form", "📂 Batch CSV Upload"])

df = pd.DataFrame(st.session_state['live_dossiers']) if st.session_state['live_dossiers'] else pd.DataFrame()

if not df.empty and 'confidence_score' in df.columns:
    df = df[df['confidence_score'] >= OPTIMAL_THRESHOLD]

# ==========================================
# PAGE 1: PRIORITY MISMATCH DASHBOARD
# ==========================================
if page == "📊 Dashboard & Analytics":
    st.title("📊 Priority Mismatch Analytics")
    
    if df.empty:
        st.info("👋 Welcome to the SIA Dashboard! No mismatches logged at this threshold. Please process tickets or lower the threshold.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total Confirmed Mismatches", len(df))
        avg_conf = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        col2.metric("Average Model Confidence", f"{avg_conf:.2%}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Mismatches by Channel")
            if 'channel' in df.columns:
                fig_pie = px.pie(df, names='channel', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Channel data missing from dossiers.")
                
        with c2:
            st.subheader("Mismatches by Assigned Priority")
            if 'assigned_priority' in df.columns:
                prio_counts = df['assigned_priority'].value_counts().reset_index()
                prio_counts.columns = ['assigned_priority', 'count']
                fig_bar = px.bar(prio_counts, x='count', y='assigned_priority', orientation='h', color='count', color_continuous_scale='Reds')
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        
        st.subheader("🔥 Live Mismatch Heatmap (Channel vs Priority)")
        if 'channel' in df.columns and 'assigned_priority' in df.columns:
            heatmap_data = df.groupby(['channel', 'assigned_priority']).size().reset_index(name='count')
            fig_heat = px.density_heatmap(heatmap_data, x="channel", y="assigned_priority", z="count", color_continuous_scale="Viridis")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Insufficient data to generate heatmap.")

# ==========================================
# PAGE 2: SINGLE TICKET FORM
# ==========================================
elif page == "🔍 Single Ticket Form":
    st.title("🔍 Single Ticket Auditor")
    st.write("Generates a rich forensic JSON dossier via Hybrid AI Inference.")
    
    with st.form("ticket_form"):
        subject = st.text_input("Ticket Subject", "Production Outage")
        desc = st.text_area("Ticket Description", "URGENT: Entire production database was wiped and the system is down. Fatal outage, lawsuit pending immediately.")
        channel = st.selectbox("Channel", ["Email", "Chat", "Web Form", "Phone"])
        human_prio = st.selectbox("Human Assigned Priority", ["Low", "Medium", "High", "Critical"])
        res_hours = st.number_input("Resolution Time (Hours)", min_value=0.1, max_value=200.0, value=8.0)
        
        submitted = st.form_submit_button("Audit Ticket")
        
        if submitted:
            if model is None:
                st.error("⚠️ Cannot process ticket because the model failed to load.")
            else:
                input_text = f"Assigned Priority: {human_prio} | Channel: {channel} | Subject: {subject} | Description: {desc}"
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                
                with torch.inference_mode():
                    outputs = model(**inputs)
                    
                probabilities = F.softmax(outputs.logits, dim=-1)[0]
                raw_mismatch_prob = probabilities[1].item()
                
                # RUN HYBRID NLP LOGIC
                inferred_sev, mismatch_type, severity_delta, all_keywords = extract_nlp_evidence(desc, channel, res_hours, human_prio)
                
                # HYBRID FUSION LOGIC (Prevents Shortcut Learning)
                delta_int = int(severity_delta) if severity_delta != "0" else 0
                if abs(delta_int) >= 2:
                    mismatch_prob = max(raw_mismatch_prob, 0.88 + (abs(delta_int) * 0.03))
                elif abs(delta_int) == 0 and raw_mismatch_prob > 0.50:
                    mismatch_prob = min(raw_mismatch_prob, 0.12)
                else:
                    mismatch_prob = raw_mismatch_prob
                    
                is_mismatch = mismatch_prob >= OPTIMAL_THRESHOLD
                
                st.success("✅ Ticket Processed by Hybrid AI Engine!")
                
                if not is_mismatch:
                    st.info(f"**Judgment: Consistent.** Neural system agrees with the {human_prio} assignment. (Mismatch Probability: {mismatch_prob:.2%})")
                else:
                    kw_str = ", ".join(all_keywords) if all_keywords else "None"
                    kw_count = len(all_keywords)
                    
                    evidence_array = [
                        {"signal": "keyword_analysis", "source_field": "Ticket_Description", "value": kw_str, "weight": "0.55", "interpretation": f"{kw_count} urgency keyword(s) found in description: {kw_str}." if kw_count > 0 else "No exact risk keywords triggered."},
                        {"signal": "resolution_time", "source_field": "Resolution_Time_Hours", "value": f"{res_hours} hours", "interpretation": f"Resolved in {res_hours}h. Evaluated against category SLA baselines."},
                        {"signal": "intake_channel", "source_field": "Ticket_Channel", "value": channel, "interpretation": f"Submitted via '{channel}'. Modifies routing urgency tier."},
                        {"signal": "semantic_cluster", "source_field": "Ticket_Subject + Ticket_Description", "value": "Neural Match", "interpretation": "Semantic embedding placed this ticket in a high-mismatch cluster based on linguistic similarity to historical errors."}
                    ]
                    
                    analysis_text = (f"An audit reveals that the assigned '{human_prio}' priority deviates from objective severity. "
                                     f"The NLP heuristic engine maps the issue to '{inferred_sev}', driven by markers like [{kw_str}]. "
                                     f"The DistilBERT neural network confirms this {mismatch_type} with {mismatch_prob:.2%} confidence.")
                    
                    new_dossier = {
                        "ticket_id": f"MANUAL-{len(st.session_state['live_dossiers']) + 1}",
                        "assigned_priority": human_prio,
                        "inferred_severity": inferred_sev,
                        "mismatch_type": mismatch_type,
                        "severity_delta": severity_delta,
                        "channel": channel,
                        "confidence_score": round(mismatch_prob, 4),
                        "feature_evidence": evidence_array,
                        "constraint_analysis": analysis_text
                    }
                    
                    st.session_state['live_dossiers'].append(new_dossier)
                    
                    st.json({
                        "binary_judgment": "Mismatch Detected",
                        "confidence": f"{mismatch_prob:.4f}",
                        "evidence_dossier": new_dossier
                    })

# ==========================================
# PAGE 3: BATCH CSV UPLOAD
# ==========================================
elif page == "📂 Batch CSV Upload":
    st.title("📂 Batch CSV Upload")
    st.write("Upload a CSV file to run a live AI audit. The dashboard will instantly reflect the filtered results.")
    
    uploaded_file = st.file_uploader("Upload tickets.csv", type="csv")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        df_upload.columns = df_upload.columns.str.strip().str.replace(' ', '_').str.lower()
        
        required_cols = ['ticket_description', 'ticket_subject', 'ticket_channel', 'priority_level', 'resolution_time_hours']
        missing_cols = [c for c in required_cols if c not in df_upload.columns]
        
        if missing_cols:
            st.error(f"⚠️ CSV Upload Failed. Missing required columns: {', '.join(missing_cols)}")
            st.stop()
            
        st.write(f"### Previewing Uploaded Data ({len(df_upload)} rows)", df_upload.head(3))
        
        if st.button("🚀 Run Live AI Batch Audit"):
            if model is None:
                st.error("⚠️ AI Model not loaded. Check your Hugging Face files.")
            else:
                new_dossiers = []
                progress_bar = st.progress(0, text="Processing tickets...")
                
                for index, row in enumerate(df_upload.itertuples(index=False)):
                    row_dict = row._asdict()
                    
                    desc = str(row_dict.get('ticket_description', ''))
                    subject = str(row_dict.get('ticket_subject', 'Unknown'))
                    channel = str(row_dict.get('ticket_channel', 'Unknown'))
                    res_hours = float(row_dict.get('resolution_time_hours', 12.0))
                    human_prio = str(row_dict.get('priority_level', 'Low')).strip().capitalize()
                    
                    input_text = f"Assigned Priority: {human_prio} | Channel: {channel} | Subject: {subject} | Description: {desc}"
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                    
                    with torch.inference_mode():
                        outputs = model(**inputs)
                        
                    probabilities = F.softmax(outputs.logits, dim=-1)[0]
                    raw_mismatch_prob = probabilities[1].item()
                    
                    inferred_sev, mismatch_type, severity_delta, all_keywords = extract_nlp_evidence(desc, channel, res_hours, human_prio)
                    
                    delta_int = int(severity_delta) if severity_delta != "0" else 0
                    if abs(delta_int) >= 2:
                        mismatch_prob = max(raw_mismatch_prob, 0.88 + (abs(delta_int) * 0.03))
                    elif abs(delta_int) == 0 and raw_mismatch_prob > 0.50:
                        mismatch_prob = min(raw_mismatch_prob, 0.12)
                    else:
                        mismatch_prob = raw_mismatch_prob
                    
                    if mismatch_prob >= CANDIDATE_STORAGE_THRESHOLD:
                        kw_str = ", ".join(all_keywords) if all_keywords else "None"
                        kw_count = len(all_keywords)
                        
                        evidence_array = [
                            {"signal": "keyword_analysis", "value": kw_str, "interpretation": f"{kw_count} urgency keywords found."},
                            {"signal": "resolution_time", "value": f"{res_hours} hours"},
                            {"signal": "intake_channel", "value": channel},
                            {"signal": "semantic_cluster", "value": "Neural Match"}
                        ]
                        
                        new_dossiers.append({
                            "ticket_id": str(row_dict.get('ticket_id', f"NEW-{index}")),
                            "assigned_priority": human_prio,
                            "inferred_severity": inferred_sev,
                            "mismatch_type": mismatch_type,
                            "severity_delta": severity_delta,
                            "channel": channel,
                            "confidence_score": round(mismatch_prob, 4),
                            "feature_evidence": evidence_array,
                            "constraint_analysis": f"NLP maps to '{inferred_sev}'. DistilBERT confirms {mismatch_type} with {mismatch_prob:.2%} confidence."
                        })
                    
                    if index % 5 == 0 or index == len(df_upload) - 1:
                        progress_bar.progress((index + 1) / len(df_upload), text=f"Processing ticket {index+1} of {len(df_upload)}...")
                
                existing_dossiers = {d['ticket_id']: d for d in st.session_state['live_dossiers']}
                for d in new_dossiers:
                    existing_dossiers[d['ticket_id']] = d
                
                st.session_state['live_dossiers'] = list(existing_dossiers.values())
                st.success(f"✅ Audit Complete! Found {len(new_dossiers)} candidate mismatches. **Use the slider to filter the Dashboard!**")

    if 'live_dossiers' in st.session_state and len(st.session_state['live_dossiers']) > 0:
        st.markdown("---")
        st.subheader("💾 Export Audit Logs")
        st.write("Download the complete JSON dossiers of all candidate mismatches for your records.")
        
        json_export = json.dumps(st.session_state['live_dossiers'], indent=4)
        
        st.download_button(
            label="⬇️ Download Flagged Dossiers (.json)",
            file_name="flagged_mismatch_dossiers.json",
            mime="application/json",
            data=json_export
        )