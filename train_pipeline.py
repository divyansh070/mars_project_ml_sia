import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, cohen_kappa_score
import re
import json
import torch.nn.functional as F
import warnings
from evaluate import load

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🛡️ Support Integrity Auditor (SIA) - Training Pipeline")
    print("="*60)

    # ==========================================
    # [1/8] LOADING & PREPROCESSING
    # ==========================================
    print("\n[1/8] Loading Data & Strict Stratified Split...")
    try:
        df_raw = pd.read_csv('customer_support_tickets.csv')
    except FileNotFoundError:
        print("❌ Error: 'customer_support_tickets.csv' not found in the current directory.")
        return

    COLUMN_MAP = {
        "ticket_priority": "priority_level",
        "time_to_resolution": "resolution_time_hours",
        "ticket_type": "issue_category",
        "ticket_subject": "ticket_subject"
    }
    df_raw.rename(columns=COLUMN_MAP, inplace=True)
    df_raw.columns = df_raw.columns.str.strip().str.replace(' ', '_').str.lower()

    df_raw['resolution_time_hours'] = pd.to_numeric(df_raw['resolution_time_hours'], errors='coerce')
    PRIORITY_MAP = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
    df_raw['priority_level'] = df_raw['priority_level'].astype(str).str.strip().str.lower().map(PRIORITY_MAP)

    df = df_raw.dropna(subset=['ticket_description', 'ticket_subject', 'priority_level', 'ticket_channel', 'resolution_time_hours', 'issue_category']).reset_index(drop=True)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['priority_level'])
    
    print(f"Training Samples: {len(df_train)} | Validation Samples: {len(df_val)}")

    # ==========================================
    # [2/8] WEAK SUPERVISION (PSEUDO-LABELING)
    # ==========================================
    print("\n[2/8] Dynamic Signal Fusion & Pseudo-Labeling...")
    train_medians = df_train.groupby('issue_category')['resolution_time_hours'].median().to_dict()

    median_times = df_train.groupby('priority_level')['resolution_time_hours'].median()
    crit_time = median_times.get('Critical', 1.0)
    low_time = median_times.get('Low', 10.0)
    sla_reversed = crit_time < low_time
    
    def signal_1_nlp_rules(text):
        text = str(text).lower()
        if re.search(r"\b(outage|breach|lawsuit|security|fraud|wiped|gone|down|fatal)\b", text): return "Critical"
        if re.search(r"\b(urgent|asap|escalate|manager|unacceptable|broken|crash|immediately)\b", text): return "High"
        if re.search(r"\b(delay|waiting|cancel|refund|error|issue|problem|help|stuck)\b", text): return "Medium"
        return "Low"

    def signal_2_resolution(res_time, category):
        median = train_medians.get(category, res_time)
        ratio = res_time / median if median > 0 else 1
        if sla_reversed: 
            if ratio < 0.35: return "Critical"
            if ratio < 0.75: return "High"
            if ratio > 1.50: return "Low"
            return "Medium"
        else: 
            if ratio > 2.0: return "Critical"
            if ratio > 1.5: return "High"
            if ratio < 0.5: return "Low"
            return "Medium"

    sev_to_int = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    int_to_sev = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

    df_train = df_train.copy()
    df_val = df_val.copy()

    df_train['sig1_nlp'] = df_train['ticket_description'].apply(signal_1_nlp_rules)
    df_train['sig2_res'] = df_train.apply(lambda r: signal_2_resolution(r['resolution_time_hours'], r['issue_category']), axis=1)

    df_train['pseudo_severity_int'] = np.round((df_train['sig1_nlp'].map(sev_to_int) * 0.80) + (df_train['sig2_res'].map(sev_to_int) * 0.20)).astype(int)
    df_train['pseudo_severity'] = df_train['pseudo_severity_int'].map(int_to_sev)

    def calculate_mismatch(human_priority, inferred_severity):
        return int(str(human_priority).strip().capitalize() != str(inferred_severity).strip().capitalize())

    df_train['binary_mismatch_label'] = df_train.apply(lambda r: calculate_mismatch(r['priority_level'], r['pseudo_severity']), axis=1)

    # Metrics
    agreement_pct = (df_train['sig1_nlp'] == df_train['sig2_res']).mean()
    kappa = cohen_kappa_score(df_train['sig1_nlp'], df_train['sig2_res'])
    print(f"Signal Agreement: {agreement_pct:.2%} | Cohen's Kappa: {kappa:.4f}")
    print(f"Fused System Mismatch Rate: {df_train['binary_mismatch_label'].mean():.2%}")

    # ==========================================
    # [3/8] DATA PREPARATION
    # ==========================================
    print("\n[3/8] Engineering Inputs...")
    def build_input(row):
        return (f"Assigned Priority: {row['priority_level']} | "
                f"Channel: {row['ticket_channel']} | "
                f"Subject: {row['ticket_subject']} | "
                f"Description: {row['ticket_description']}")

    train_texts = df_train.apply(build_input, axis=1).tolist()
    train_labels = df_train['binary_mismatch_label'].tolist()

    df_val['sig1_nlp'] = df_val['ticket_description'].apply(signal_1_nlp_rules)
    df_val['sig2_res'] = df_val.apply(lambda r: signal_2_resolution(r['resolution_time_hours'], r['issue_category']), axis=1)
    df_val['pseudo_severity_int'] = np.round((df_val['sig1_nlp'].map(sev_to_int) * 0.80) + (df_val['sig2_res'].map(sev_to_int) * 0.20)).astype(int)
    df_val['pseudo_severity'] = df_val['pseudo_severity_int'].map(int_to_sev)
    df_val['binary_mismatch_label'] = df_val.apply(lambda r: calculate_mismatch(r['priority_level'], r['pseudo_severity']), axis=1)

    val_texts = df_val.apply(build_input, axis=1).tolist()
    val_labels = df_val['binary_mismatch_label'].tolist()

    # ==========================================
    # [4/8] DISTILBERT SETUP
    # ==========================================
    print("\n[4/8] DistilBERT Binary Classifier Setup...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    class TicketDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
        def __len__(self):
            return len(self.labels)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    
    f1_metric = load("f1")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_metric.compute(predictions=predictions, references=labels, average="macro")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        report_to="none" # Silences W&B / external trackers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TicketDataset(train_encodings, train_labels),
        eval_dataset=TicketDataset(val_encodings, val_labels),
        compute_metrics=compute_metrics
    )

    # ==========================================
    # [5/8] TRAINING
    # ==========================================
    print("\n[5/8] Training Strict Binary Classifier...")
    trainer.train()

    # ==========================================
    # [6/8] METRICS & AUTO-TUNING
    # ==========================================
    print("\n[6/8] Final Validation Metrics & Auto-Tuner...")
    preds_output = trainer.predict(TicketDataset(val_encodings, val_labels))
    probabilities = F.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()

    best_thresh = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.7, 0.01):
        temp_pred = (probabilities[:, 1] >= t).astype(int)
        f1 = f1_score(val_labels, temp_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\n✅ Optimal Decision Boundary found at: {best_thresh:.2f}")
    y_pred = (probabilities[:, 1] >= best_thresh).astype(int)
    print(classification_report(val_labels, y_pred, target_names=["Consistent (0)", "Mismatch (1)"], zero_division=0))

    # ==========================================
    # [7/8] EXPORTING MODEL
    # ==========================================
    print("\n[7/8] Exporting Model & Configurations...")
    trainer.save_model('./model')
    tokenizer.save_pretrained('./model')

    with open('threshold.json', 'w') as f:
        json.dump({"best_thresh": float(best_thresh)}, f)
        
    print("✅ Saved ./model directory and threshold.json for Streamlit deployment.")

    # ==========================================
    # [8/8] ADVERSARIAL EVALUATION
    # ==========================================
    print("\n[8/8] Held-out Adversarial Set Evaluation...")
    adv_test_set = [
        {"ticket_channel": "Email", "ticket_subject": "Update", "ticket_description": "This is NOT urgent, take your time.", "priority_level": "Critical", "expected": 1},
        {"ticket_channel": "Chat", "ticket_subject": "Crash", "ticket_description": "I absolutely love when your servers crash during demos. Great job.", "priority_level": "Low", "expected": 1},
        {"ticket_channel": "Web Form", "ticket_subject": "Data", "ticket_description": "Database wiped.", "priority_level": "Low", "expected": 1},
        {"ticket_channel": "Email", "ticket_subject": "Help", "ticket_description": "No rush, but all customer records disappeared.", "priority_level": "Low", "expected": 1},
        {"ticket_channel": "Phone", "ticket_subject": "Refund", "ticket_description": "Refund pending for 2 weeks.", "priority_level": "Critical", "expected": 1},
        {"ticket_channel": "Email", "ticket_subject": "Login", "ticket_description": "Forgot my password, please help me reset.", "priority_level": "Critical", "expected": 1},
        {"ticket_channel": "Chat", "ticket_subject": "Bug", "ticket_description": "There is a minor typo on the homepage footer.", "priority_level": "High", "expected": 1},
        {"ticket_channel": "Web Form", "ticket_subject": "Urgent", "ticket_description": "URGENT: Entire production system is down, lawsuit pending.", "priority_level": "Low", "expected": 1},
        {"ticket_channel": "Email", "ticket_subject": "Invoice", "ticket_description": "Need a copy of last month's invoice.", "priority_level": "High", "expected": 1},
        {"ticket_channel": "Chat", "ticket_subject": "Feedback", "ticket_description": "Your UI is very confusing.", "priority_level": "Critical", "expected": 1}
    ]

    adv_df = pd.DataFrame(adv_test_set)
    adv_texts = adv_df.apply(lambda r: f"Assigned Priority: {r['priority_level']} | Channel: {r['ticket_channel']} | Subject: {r['ticket_subject']} | Description: {r['ticket_description']}", axis=1).tolist()
    adv_labels = adv_df['expected'].tolist()

    adv_encodings = tokenizer(adv_texts, truncation=True, padding=True, max_length=256)
    adv_dataset = TicketDataset(adv_encodings, adv_labels)

    adv_preds = trainer.predict(adv_dataset)
    adv_probs = F.softmax(torch.tensor(adv_preds.predictions), dim=-1).numpy()
    adv_y_pred = (adv_probs[:, 1] >= best_thresh).astype(int)

    correct_adv = sum(adv_y_pred == adv_labels)
    print(f"🎯 Adversarial Score: {correct_adv}/10")
    print("✅ Fully Compliant Pipeline Complete.")

if __name__ == "__main__":
    main()