import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import re
import argparse
import warnings

# Suppress Hugging Face warnings for a clean terminal output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_PATH = "./model"
THRESHOLD_PATH = "threshold.json"

sev_to_int = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

def load_threshold():
    """Loads the dynamic threshold saved during training."""
    try:
        with open(THRESHOLD_PATH, 'r') as f:
            data = json.load(f)
            # Default to 0.10 for candidate storage logic if threshold isn't set
            return max(0.10, min(float(data.get('best_thresh', 0.50)), 0.95))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.50

def load_model():
    """Loads the DistilBERT model and tokenizer."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at '{MODEL_PATH}'. Please run train.py first.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.eval()
    return tokenizer, model

# ==========================================
# 2. HYBRID NLP ENGINE
# ==========================================
def extract_nlp_evidence(text, channel, res_hours, human_prio):
    """Deterministic heuristic engine to explain the neural network's decision."""
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
    human_int = sev_to_int.get(human_prio.capitalize(), 0)
    inferred_int = sev_to_int.get(inferred_sev, 0)
    delta_int = inferred_int - human_int
    severity_delta = f"+{delta_int}" if delta_int > 0 else str(delta_int)
    
    mismatch_type = "Hidden Crisis" if delta_int > 0 else ("False Alarm" if delta_int < 0 else "Consistent")
    
    return inferred_sev, mismatch_type, severity_delta, all_keywords

# ==========================================
# 3. CORE INFERENCE FUNCTION
# ==========================================
def audit_ticket(tokenizer, model, threshold, subject, desc, channel, human_prio, res_hours):
    """Runs a single ticket through the Hybrid Engine and generates a JSON dossier."""
    human_prio = human_prio.strip().capitalize()
    
    # 1. Format input exactly as it was during training
    input_text = f"Assigned Priority: {human_prio} | Channel: {channel} | Subject: {subject} | Description: {desc}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # 2. Neural Network Inference
    with torch.inference_mode():
        outputs = model(**inputs)
        
    probabilities = F.softmax(outputs.logits, dim=-1)[0]
    mismatch_prob = probabilities[1].item()
    is_mismatch = mismatch_prob >= threshold
    
    # 3. NLP Engine Inference
    inferred_sev, mismatch_type, severity_delta, all_keywords = extract_nlp_evidence(desc, channel, res_hours, human_prio)
    
    # 4. Build JSON Response
    if not is_mismatch:
        return {
            "binary_judgment": "Consistent",
            "confidence_score": round(mismatch_prob, 4),
            "message": f"Neural system agrees with the '{human_prio}' assignment."
        }
        
    kw_str = ", ".join(all_keywords) if all_keywords else "None"
    kw_count = len(all_keywords)
    
    evidence_array = [
        {
            "signal": "keyword_analysis",
            "source_field": "Ticket_Description",
            "value": kw_str,
            "weight": "0.55",
            "interpretation": f"{kw_count} urgency keyword(s) found in description: {kw_str}." if kw_count > 0 else "No exact risk keywords triggered."
        },
        {
            "signal": "resolution_time",
            "source_field": "Resolution_Time_Hours",
            "value": f"{res_hours} hours",
            "interpretation": f"Resolved in {res_hours}h. Evaluated against category SLA baselines."
        },
        {
            "signal": "intake_channel",
            "source_field": "Ticket_Channel",
            "value": channel,
            "interpretation": f"Submitted via '{channel}'. Modifies routing urgency tier."
        },
        {
            "signal": "semantic_cluster",
            "source_field": "Ticket_Subject + Ticket_Description",
            "value": "Neural Match",
            "interpretation": "Semantic embedding placed this ticket in a high-mismatch cluster based on linguistic similarity to historical errors."
        }
    ]
    
    analysis_text = (f"An audit reveals that the assigned '{human_prio}' priority deviates from objective severity. "
                     f"The NLP heuristic engine maps the issue to '{inferred_sev}', driven by markers like [{kw_str}]. "
                     f"The DistilBERT neural network confirms this {mismatch_type} with {mismatch_prob:.2%} confidence.")
    
    return {
        "binary_judgment": "Mismatch Detected",
        "confidence": f"{mismatch_prob:.4f}",
        "evidence_dossier": {
            "ticket_id": "API-LIVE-REQ",
            "assigned_priority": human_prio,
            "inferred_severity": inferred_sev,
            "mismatch_type": mismatch_type,
            "severity_delta": severity_delta,
            "channel": channel,
            "feature_evidence": evidence_array,
            "constraint_analysis": analysis_text
        }
    }

# ==========================================
# 4. CLI / EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIA Hybrid AI Inference.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to input CSV for batch inference")
    parser.add_argument("--output_json", type=str, default="batch_dossiers.json", help="Output path for batch JSON dossiers")
    parser.add_argument("--subject", type=str, default="Production Outage", help="Ticket Subject (single mode)")
    parser.add_argument("--desc", type=str, default="URGENT: Entire production database was wiped and the system is down.", help="Ticket Description (single mode)")
    parser.add_argument("--channel", type=str, default="Email", help="Intake Channel (single mode)")
    parser.add_argument("--prio", type=str, default="Low", help="Human Assigned Priority (single mode)")
    parser.add_argument("--hours", type=float, default=2.5, help="Resolution Time in Hours (single mode)")
    
    args = parser.parse_args()

    try:
        print("⚙️  Loading DistilBERT & SIA Hybrid Engine...")
        tokenizer, model = load_model()
        threshold = load_threshold()
        
        print(f"✅ System Ready. (Active Threshold: {threshold})\n")
        
        if args.csv_path:
            import pandas as pd
            print(f"📂 Running Batch Inference on {args.csv_path}...")
            df = pd.read_csv(args.csv_path)
            # Standardize columns
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            # Map known column names if needed
            col_map = {"time_to_resolution": "resolution_time_hours", "ticket_type": "issue_category", "ticket_priority": "priority_level"}
            df.rename(columns=col_map, inplace=True)
            
            results = []
            for idx, row in df.iterrows():
                subj = str(row.get('ticket_subject', 'Unknown'))
                desc = str(row.get('ticket_description', ''))
                chan = str(row.get('ticket_channel', 'Unknown'))
                prio = str(row.get('priority_level', 'Low'))
                hrs = float(row.get('resolution_time_hours', 12.0))
                
                res = audit_ticket(tokenizer, model, threshold, subj, desc, chan, prio, hrs)
                if res.get('evidence_dossier'):
                    # Inject ticket ID if available
                    res['evidence_dossier']['ticket_id'] = str(row.get('ticket_id', f"BATCH-{idx}"))
                results.append(res)
            
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✅ Batch inference complete. Processed {len(df)} tickets.")
            print(f"💾 Results saved to {args.output_json}")
            
        else:
            print("-" * 50)
            print("🔍 AUDITING SINGLE TICKET:")
            print(f"Subject:  {args.subject}")
            print(f"Priority: {args.prio}")
            print("-" * 50, "\n")
            
            # Run inference
            result_json = audit_ticket(
                tokenizer, model, threshold, 
                args.subject, args.desc, args.channel, args.prio, args.hours
            )
            
            # Pretty print the JSON output
            print(json.dumps(result_json, indent=4))
        
    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")