"""Sequence-to-sequence strategy template using pretrained Transformers.

This template provides a generic seq2seq approach using T5 that can be adapted
by the LLM for specific competition requirements. No competition-specific
hardcoding - the LLM will customize based on competition description.
"""

# =============================================================================
# GENERIC SEQ2SEQ TEMPLATE - T5-based approach
# Works for any sequence-to-sequence task (translation, normalization, etc.)
# The LLM will customize column names, task prefix, etc. based on competition
# =============================================================================

SEQ2SEQ_TEMPLATE = """
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = {seed}
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# ============================================================================
# DATASET
# ============================================================================

class Seq2SeqDataset(Dataset):
    \"\"\"Generic dataset for sequence-to-sequence tasks.\"\"\"
    
    def __init__(self, sources, targets, tokenizer, max_source_len=128, max_target_len=128, task_prefix=""):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.task_prefix = task_prefix
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source_text = self.task_prefix + str(self.sources[idx])
        
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            truncation=True,
            return_tensors=None
        )
        
        item = {{
            'input_ids': torch.tensor(source_encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(source_encoding['attention_mask'], dtype=torch.long),
            'source_text': str(self.sources[idx])
        }}
        
        if self.targets is not None:
            target_text = str(self.targets[idx])
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                truncation=True,
                return_tensors=None
            )
            item['labels'] = torch.tensor(target_encoding['input_ids'], dtype=torch.long)
        
        return item


def collate_fn(batch, pad_token_id):
    \"\"\"Collate function for padding sequences in a batch.\"\"\"
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    source_texts = [item['source_text'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    output = {{
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'source_texts': source_texts
    }}
    
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
        output['labels'] = labels_padded
    
    return output


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("Loading data...")
train_path = '{train_path}'
test_path = '{test_path}'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {{train_df.shape}}")
print(f"Test shape: {{test_df.shape}}")
print(f"Train columns: {{list(train_df.columns)}}")
print(f"Test columns: {{list(test_df.columns)}}")

# ============================================================================
# COLUMN CONFIGURATION - LLM MUST CUSTOMIZE THESE BASED ON COMPETITION DATA
# ============================================================================
# The LLM should replace these placeholders with actual column names
# after examining train_data_preview and test_data_preview in the context

SOURCE_COLUMN = '{source_column}'  # Column containing input text (must exist in both train AND test)
TARGET_COLUMN = '{target_column}'  # Column containing output text (only in train, this is what we predict)
ID_COLUMN = '{id_column}'          # Column for submission IDs

# Task prefix for T5 (helps the model understand the task)
TASK_PREFIX = "transform: "  # LLM can customize: "normalize: ", "translate: ", etc.

print(f"Source column: {{SOURCE_COLUMN}}")
print(f"Target column: {{TARGET_COLUMN}}")
print(f"ID column: {{ID_COLUMN}}")

# ============================================================================
# HANDLE MISSING/EMPTY VALUES - Fill None and NaN with empty strings
# ============================================================================
# This is critical for text columns to avoid errors during tokenization

# Fill missing values in source column with empty string
train_df[SOURCE_COLUMN] = train_df[SOURCE_COLUMN].fillna('').astype(str)
test_df[SOURCE_COLUMN] = test_df[SOURCE_COLUMN].fillna('').astype(str)

# Fill missing values in target column with empty string (train only)
train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].fillna('').astype(str)

print(f"Missing values handled - all None/NaN converted to empty strings")

# Sample if dataset is too large for time budget
MAX_TRAIN_SAMPLES = 500000
if len(train_df) > MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=SEED)
    print(f"Sampled training data to {{len(train_df)}} rows")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    train_df[SOURCE_COLUMN].values,
    train_df[TARGET_COLUMN].values,
    test_size=0.1,
    random_state=SEED
)

print(f"Training samples: {{len(X_train)}}")
print(f"Validation samples: {{len(X_val)}}")

# Model and Tokenizer
model_name = 't5-small'
print(f"Loading model: {{model_name}}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")

# Create datasets
train_dataset = Seq2SeqDataset(X_train, y_train, tokenizer, 
                                max_source_len={max_source_length}, 
                                max_target_len={max_target_length},
                                task_prefix=TASK_PREFIX)
val_dataset = Seq2SeqDataset(X_val, y_val, tokenizer,
                              max_source_len={max_source_length},
                              max_target_len={max_target_length},
                              task_prefix=TASK_PREFIX)
test_dataset = Seq2SeqDataset(test_df[SOURCE_COLUMN].values, None, tokenizer,
                               max_source_len={max_source_length},
                               max_target_len={max_target_length},
                               task_prefix=TASK_PREFIX)

# DataLoaders
collate_with_padding = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, 
                          collate_fn=collate_with_padding, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False,
                        collate_fn=collate_with_padding, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size={batch_size}*2, shuffle=False,
                         collate_fn=collate_with_padding, num_workers=0)

# Training setup
optimizer = AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})
total_steps = len(train_loader) * {max_epochs}
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=total_steps // 10,
                                            num_training_steps=total_steps)
scaler = GradScaler()

best_val_loss = float('inf')
patience_counter = 0

print("Starting training...")
for epoch in range({max_epochs}):
    # Training
    model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{max_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {{epoch+1}}: Train Loss={{avg_train_loss:.4f}}, Val Loss={{avg_val_loss:.4f}}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  Saved best model (val_loss={{avg_val_loss:.4f}})')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print("Early stopping!")
            break

# Load best model
print("Loading best model for inference...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference
print("Running inference...")
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length={max_target_length},
            num_beams={beam_search_size},
            length_penalty={length_penalty},
            early_stopping=True
        )
        
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(preds)

# Create submission
print("Creating submission...")
submission = pd.DataFrame({{
    'id': test_df[ID_COLUMN].values,
    '{prediction_column}': predictions
}})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved: {{len(submission)}} rows")
print("Done!")
"""


def get_seq2seq_template(resource_constrained: bool = False, is_text_normalization: bool = False) -> str:
    """
    Get seq2seq template.
    
    Args:
        resource_constrained: Ignored - always use the optimized template
        is_text_normalization: Ignored - LLM customizes based on competition
        
    Returns:
        Template string
    """
    return SEQ2SEQ_TEMPLATE
