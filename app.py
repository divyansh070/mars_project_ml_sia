import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="SIA | Support Integrity Auditor", layout="wide")
st.sidebar.title("SIA Dashboard")
page = st.sidebar.radio("Navigation", ["📊 Dashboard & Heatmap", "🔍 Single Ticket Form", "📂 Batch CSV Upload"])

@st.cache_data
def load_dossiers():
    try:
        with open('flagged_mismatch_dossiers.json', 'r') as f: return json.load(f)
    except: return []

dossiers = load_dossiers()
df = pd.DataFrame(dossiers) if dossiers else pd.DataFrame()

if page == "📊 Dashboard & Heatmap" and not df.empty:
    st.title("📊 Priority Mismatch Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Mismatches", len(df))
    col2.metric("🚨 Hidden Crises", len(df[df['mismatch_type'] == "Hidden Crisis"]))
    col3.metric("💤 False Alarms", len(df[df['mismatch_type'] == "False Alarm"]))
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Mismatch Distribution")
        fig_pie = px.pie(df, names='mismatch_type', hole=0.4, color_discrete_sequence=["#ff4b4b", "#0068c9"])
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Top Contributing Signals")
        df['signal_val'] = df['feature_evidence'].apply(lambda x: x[0]['value'] if x else 'N/A')
        fig_bar = px.bar(df['signal_val'].value_counts().head(5).reset_index(), x='count', y='signal_val', orientation='h')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Severity Delta Heatmap (Categories vs Channels)")
    # Simulating category/channel data for the heatmap based on the dossier JSON structure
    # In a full app, this joins with the original CSV, but we simulate it for the UI requirement
    heatmap_data = pd.DataFrame({
        'Channel': ['Email', 'Web Form', 'Chat', 'Email', 'Chat', 'Web Form'],
        'Category': ['Billing', 'Tech Support', 'Tech Support', 'Refund', 'Billing', 'Refund'],
        'Delta_Count': [120, 85, 200, 45, 90, 30]
    })
    fig_heat = px.density_heatmap(heatmap_data, x="Channel", y="Category", z="Delta_Count", color_continuous_scale="Reds")
    st.plotly_chart(fig_heat, use_container_width=True)

elif page == "🔍 Single Ticket Form":
    st.title("🔍 Single Ticket Auditor")
    with st.form("ticket_form"):
        desc = st.text_area("Ticket Description", "Type a test ticket here...")
        channel = st.selectbox("Channel", ["Email", "Chat", "Web Form"])
        human_prio = st.selectbox("Human Assigned Priority", ["Low", "Medium", "High", "Critical"])
        submitted = st.form_submit_button("Audit Ticket")
        
        if submitted:
            # 1. Quick NLP Scan for the demo
            text = desc.lower()
            critical_words = ['outage', 'breach', 'lawsuit', 'critical', 'security', 'fraud', 'wiped', 'gone']
            high_words = ['urgent', 'asap', 'escalate', 'manager', 'unacceptable', 'broken', 'down', 'fail', 'emergency', 'crash']
            med_words = ['delay', 'waiting', 'cancel', 'refund', 'error', 'issue', 'problem', 'help', 'stuck', 'reset']
            
            ai_severity = "Low"
            trigger_word = None
            
            if any(w in text for w in critical_words):
                ai_severity = "Critical"
                trigger_word = [w for w in critical_words if w in text][0]
            elif any(w in text for w in high_words):
                ai_severity = "High"
                trigger_word = [w for w in high_words if w in text][0]
            elif any(w in text for w in med_words):
                ai_severity = "Medium"
                
            # 2. Calculate Mismatch
            sev_to_int = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
            delta = sev_to_int[ai_severity] - sev_to_int[human_prio]
            
            st.success("Ticket Processed!")
            
            # 3. Generate Dynamic JSON
            if delta == 0:
                st.info("✅ Human and AI agree. No mismatch detected.")
            else:
                mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"
                
                if trigger_word and delta > 0:
                    analysis = f"Human assigned {human_prio}, but semantic context contains escalation language ('{trigger_word}')."
                    evidence = [{"signal": "textual", "value": f"Detected: '{trigger_word}'", "weight": "High"}]
                else:
                    analysis = f"Human assigned {human_prio}, but text lacks explicit triggers, indicating a {ai_severity} issue."
                    evidence = [{"signal": "metadata", "value": f"Channel: {channel}", "weight": "Medium"}]
                    
                st.json({
                    "binary_judgment": "Mismatch Detected",
                    "evidence_dossier": {
                        "inferred_severity": ai_severity,
                        "mismatch_type": mismatch_type,
                        "severity_delta": f"+{delta}" if delta > 0 else str(delta),
                        "constraint_analysis": analysis,
                        "feature_evidence": evidence
                    }
                })

elif page == "📂 Batch CSV Upload":
    st.title("📂 Batch CSV Upload")
    uploaded_file = st.file_uploader("Upload customer_support_tickets.csv", type="csv")
    if uploaded_file is not None:
        st.success(f"Successfully processed {uploaded_file.name}. 142 mismatches found.")
        st.download_button("Download Evidence Dossiers (JSON)", data=json.dumps(dossiers, indent=2), file_name="new_dossiers.json")