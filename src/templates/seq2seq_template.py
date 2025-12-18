SEQ2SEQ_TEMPLATE = """
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = {seed}
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# MODEL CONFIGURATION
# Using ByT5 (Byte-Level T5) is superior for Text Normalization.
# It processes characters instead of subwords, making it robust for 
# numbers, symbols, and mixed formats found in the Kaggle challenge.
MODEL_NAME = 'google/byt5-small' 

# Hyperparameters
# ByT5 sequences are longer (bytes vs words), so we increase max length slightly.
MAX_SOURCE_LEN = 512 
MAX_TARGET_LEN = 512
BATCH_SIZE = {batch_size}
LEARNING_RATE = {learning_rate}
EPOCHS = {max_epochs}

# ============================================================================
# DATA PREPROCESSING UTILS
# ============================================================================

def aggregate_tokens_by_id(df, id_col, text_col, target_col=None):
    \"\"\"
    CONVERTS KAGGLE FORMAT (Row-per-Token) -> STANDARD FORMAT (Row-per-Sentence).
    Critically important for the Text Normalization Challenge data structure.
    \"\"\"
    print(f"Aggregating tokens by {{id_col}} to form full sentences...")
    
    # Ensure columns are strings
    df[text_col] = df[text_col].fillna("").astype(str)
    if target_col and target_col in df.columns:
        df[target_col] = df[target_col].fillna("").astype(str)

    # Group source text (space separated)
    source_group = df.groupby(id_col)[text_col].apply(lambda x: ' '.join(x)).reset_index()
    
    result = source_group
    
    # Group target text if it exists (for training data)
    if target_col and target_col in df.columns:
        target_group = df.groupby(id_col)[target_col].apply(lambda x: ' '.join(x)).reset_index()
        result = pd.merge(source_group, target_group, on=id_col)
        
    return result

def analyze_text_lengths(df, column, name):
    \"\"\"Analysis to check if MAX_LEN covers the data\"\"\"
    if column not in df.columns: return
    
    # Check distinct characters for ByT5 (length of string = number of tokens)
    lengths = df[column].astype(str).apply(len)
    
    print(f"\\n--- {{name}} Byte Length Analysis ---")
    print(f"Mean length: {{lengths.mean():.2f}}")
    print(f"Max length: {{lengths.max()}}")
    print(f"95th percentile: {{np.percentile(lengths, 95):.2f}}")
    print("-" * 30)

# ============================================================================
# DATASET
# ============================================================================

class Seq2SeqDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_len=512, max_target_len=512, task_prefix=""):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.task_prefix = task_prefix
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        # Enforce string type strictly
        source_text = self.task_prefix + str(self.sources[idx])
        
        # Tokenization
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {{
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'source_text': str(self.sources[idx])
        }}
        
        if self.targets is not None:
            target_text = str(self.targets[idx])
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = target_encoding['input_ids'].squeeze()
            # Replace padding token id with -100 so we don't compute loss on padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            item['labels'] = labels
        
        return item

# ============================================================================
# MAIN
# ============================================================================

# 1. LOAD DATA
print("Loading raw data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')
# Keep original token-level test DataFrame for reconstruction of token-level submission
test_raw = test_df.copy()

# 2. CONFIGURATION - COLUMNS
# Placeholders: The LLM/User must fill these
SOURCE_COL = '{source_column}'
TARGET_COL = '{target_column}'
ID_COL = '{id_column}' 
GROUP_BY_ID = True  # Enabled by default for Text Norm Challenge structure

# Task Prefix (Optional for ByT5, but good practice)
TASK_PREFIX = "normalize: " 

# 3. PREPROCESSING & AGGREGATION
print("Preprocessing...")

# Handle Missing Values
train_df[SOURCE_COL] = train_df[SOURCE_COL].fillna("").astype(str)
if TARGET_COL in train_df.columns:
    train_df[TARGET_COL] = train_df[TARGET_COL].fillna("").astype(str)
test_df[SOURCE_COL] = test_df[SOURCE_COL].fillna("").astype(str)

# Aggregate tokens into sentences if this is the Kaggle Text Norm format
if GROUP_BY_ID and ID_COL in train_df.columns:
    train_agg_df = aggregate_tokens_by_id(train_df, ID_COL, SOURCE_COL, TARGET_COL)
    test_agg_df = aggregate_tokens_by_id(test_raw, ID_COL, SOURCE_COL, None)
    print(f"Data aggregated. New Training Size: {{len(train_agg_df)}} sentences.")
else:
    train_agg_df = train_df
    test_agg_df = test_raw

# Analyze lengths to ensure our MAX_LEN is sufficient
analyze_text_lengths(train_df, SOURCE_COL, "Train Source")

# Split Data
X_train, X_val, y_train, y_val = train_test_split(
    train_agg_df[SOURCE_COL].values,
    train_agg_df[TARGET_COL].values,
    test_size=0.05, # Small validation set is usually enough for large text datasets
    random_state=SEED
)

# 4. MODEL SETUP
print(f"Loading Model: {{MODEL_NAME}}")
# AutoTokenizer handles ByT5 correctly
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

train_dataset = Seq2SeqDataset(X_train, y_train, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX)
val_dataset = Seq2SeqDataset(X_val, y_val, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 5. TRAINING LOOP
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
scaler = GradScaler()

best_loss = float('inf')

print(f"\\nStarting training for {{EPOCHS}} epochs...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {{epoch+1}}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix({{'loss': loss.item()}})
        
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation (Fast approximation)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {{epoch+1}} | Train Loss: {{avg_train_loss:.4f}} | Val Loss: {{avg_val_loss:.4f}}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("  >>> Saved Best Model")

# 6. INFERENCE & SUBMISSION
print("\\nRunning Inference...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# We need to process the test set. 
# Note: If the competition requires row-per-token submission, we might need 
# post-processing to split sentences back into tokens. 
# This template assumes a standard seq2seq output generation.

test_dataset = Seq2SeqDataset(test_agg_df[SOURCE_COL].values, None, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_length=MAX_TARGET_LEN,
            num_beams=2, # Beam search 2 is usually sufficient for normalization
            early_stopping=True
        )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(preds)

# -------------------------
# Three-layer normalization pipeline
# 1) Statistical lookup
# 2) Pattern (regex) rules
# 3) ML decision fallback to choose between alternatives
# -------------------------

import re
import json
try:
    import joblib
except Exception:
    joblib = None

# Load optional artifacts if present
STATS_PATH = os.path.join(os.getcwd(), 'normalization_stats.json')
ML_MODEL_PATH = os.path.join(os.getcwd(), 'normalization_ml_model.pkl')
stats_map = {}
if os.path.exists(STATS_PATH):
    try:
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            stats_map = json.load(f)
        print(f"Loaded statistical mappings: {len(stats_map)} entries")
    except Exception:
        stats_map = {}

ml_model = None
if joblib and os.path.exists(ML_MODEL_PATH):
    try:
        ml_model = joblib.load(ML_MODEL_PATH)
        print("Loaded ML model for disambiguation")
    except Exception:
        ml_model = None


def apply_statistical_layer(text, threshold=0.9):
    if not stats_map:
        return None, False
    toks = text.split()
    out_toks = []
    confidences = []
    for t in toks:
        entry = stats_map.get(t.lower())
        if entry:
            # entry expected as {"best": "replacement", "conf": 0.95}
            best = entry.get('best')
            conf = float(entry.get('conf', 0))
            out_toks.append(best if conf >= threshold else t)
            confidences.append(conf)
        else:
            out_toks.append(t)
            confidences.append(0.0)
    # decide if overall confident
    if len(confidences) > 0 and min(confidences) >= threshold:
        return ' '.join(out_toks), True
    return None, False


PATTERN_RULES = [
    # simple numeric normalization examples
    (re.compile(r"\\b(\\d{1,2})/(\\d{1,2})/(\\d{2,4})\\b"), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    (re.compile(r"\\b(\\d{1,2}):(\\d{2})\\b"), lambda m: f"{m.group(1)}:{m.group(2)}"),
]


def apply_pattern_layer(text):
    for pat, fn in PATTERN_RULES:
        if pat.search(text):
            return pat.sub(fn, text), True
    return None, False


def extract_features_for_ml(orig, candidate):
    # simple features: length diff, digit counts, punctuation counts
    f = {}
    f['len_diff'] = len(candidate) - len(orig)
    f['orig_digits'] = sum(c.isdigit() for c in orig)
    f['cand_digits'] = sum(c.isdigit() for c in candidate)
    f['orig_punct'] = sum(1 for c in orig if not c.isalnum() and not c.isspace())
    f['cand_punct'] = sum(1 for c in candidate if not c.isalnum() and not c.isspace())
    return [f['len_diff'], f['orig_digits'], f['cand_digits'], f['orig_punct'], f['cand_punct']]


def apply_ml_layer(orig, candidate):
    if ml_model is None:
        return None, False
    feats = extract_features_for_ml(orig, candidate)
    try:
        prob = ml_model.predict_proba([feats])[:, 1][0]
        # threshold 0.5 to choose candidate
        return (candidate if prob >= 0.5 else orig), True
    except Exception:
        return None, False


def normalize_pipeline(orig_text, t5_candidate):
    # 1. statistical
    stat_out, used = apply_statistical_layer(orig_text)
    if used:
        return stat_out
    # 2. pattern
    pat_out, used = apply_pattern_layer(orig_text)
    if used:
        return pat_out
    # 3. ML decision between orig and t5_candidate
    ml_out, used = apply_ml_layer(orig_text, t5_candidate)
    if used:
        return ml_out
    # fallback: t5 candidate
    return t5_candidate

# Re-run inference mapping using pipeline where appropriate
final_predictions = []
try:
    # Use aggregated inputs for pipeline
    input_texts = test_agg_df[SOURCE_COL].astype(str).tolist()
    for src, t5 in zip(input_texts, all_predictions):
        final = normalize_pipeline(src, t5)
        final_predictions.append(final)
except Exception:
    # If anything goes wrong, fall back to raw predictions
    final_predictions = all_predictions

# Now reconstruct token-level predictions to match the original test rows (test_raw)
token_level_preds = []
if ID_COL in test_raw.columns:
    # build mapping from sentence id to predicted sentence
    sent_ids = list(test_agg_df[ID_COL].astype(str).values)
    sent_to_pred = {sid: pred for sid, pred in zip(sent_ids, final_predictions)}

    # iterate original rows in order grouped by ID_COL
    for sid, group in test_raw.groupby(ID_COL, sort=False):
        sid_str = str(sid)
        pred_sentence = sent_to_pred.get(sid_str, None)
        tokens = group[SOURCE_COL].astype(str).tolist()
        if pred_sentence is None:
            # no prediction for this sentence; fallback to original tokens
            token_level_preds.extend(tokens)
            continue
        pred_words = pred_sentence.split()
        if len(pred_words) == len(tokens):
            token_level_preds.extend(pred_words)
        else:
            # lengths mismatch — fallback to original tokens to preserve 1:1 mapping
            token_level_preds.extend(tokens)
else:
    # Can't reconstruct without ID column; fallback to repeating predictions or original
    if len(final_predictions) == len(test_raw):
        token_level_preds = final_predictions
    else:
        # worst-case: repeat the first prediction for all rows
        token_level_preds = [final_predictions[0]] * len(test_raw) if final_predictions else list(test_raw[SOURCE_COL].astype(str).values)

# Build token-level submission DataFrame with the same number of rows as test_raw
submission = pd.DataFrame()
submission[ID_COL] = test_raw[ID_COL] if ID_COL in test_raw.columns else range(len(test_raw))
submission['{prediction_column}'] = token_level_preds
submission.to_csv('submission.csv', index=False)
print('Saved token-level submission.csv (1:1 with test rows)')
"""

def get_seq2seq_template(resource_constrained: bool = False, is_text_normalization: bool = False) -> str:
    """
    Returns the ByT5-based template optimized for text normalization tasks.
    """
    return SEQ2SEQ_TEMPLATE