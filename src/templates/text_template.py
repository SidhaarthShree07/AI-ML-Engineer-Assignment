"""Text classification strategy template with DistilBERT and TF-IDF fallback"""

TEXT_DISTILBERT_TEMPLATE = """
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }}

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Prepare data
text_column = '{text_column}'
target_column = '{target_column}'

X_train = train_df[text_column].values
y_train_raw = train_df[target_column].values
X_test = test_df[text_column].values

# Encode labels if they are strings
label_encoder = None
if y_train_raw.dtype == object:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    class_labels = label_encoder.classes_
    print(f"Classes: {{class_labels}}")
else:
    y_train = y_train_raw
    class_labels = None

num_classes = {num_classes}
if num_classes <= 1:
    num_classes = len(np.unique(y_train))

print(f"Number of classes: {{num_classes}}")

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state={seed}, stratify=y_train
)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_classes
)
model = model.to(device)

# Create datasets
train_dataset = TextDataset(X_train_split, y_train_split, tokenizer, max_length={max_length})
val_dataset = TextDataset(X_val_split, y_val_split, tokenizer, max_length={max_length})
test_dataset = TextDataset(X_test, None, tokenizer, max_length={max_length})

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=2, pin_memory=True)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})
total_steps = len(train_loader) * {max_epochs}
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps={warmup_steps},
    num_training_steps=total_steps
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range({max_epochs}):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), {gradient_clip_norm})
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}, Val Loss = {{avg_val_loss:.4f}}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print(f'Early stopping at epoch {{epoch+1}}')
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference
predictions = []
test_ids = test_df['{id_column}'].values

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.softmax(outputs.logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())

# Save predictions - handle both probability and class output formats
predictions = np.array(predictions)

# Check if submission needs probability columns (e.g., for multi-class log loss evaluation)
# Common patterns: author prediction with EAP, HPL, MWS columns
if class_labels is not None and len(class_labels) > 2:
    # Multi-class with probability outputs per class
    submission = pd.DataFrame({{'id': test_ids}})
    for i, label in enumerate(class_labels):
        submission[label] = predictions[:, i]
else:
    # Single prediction column
    submission = pd.DataFrame({{
        '{id_column}': test_ids,
        '{prediction_column}': predictions.argmax(axis=1)
    }})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows and columns: {{list(submission.columns)}}")
print("Training and inference complete!")
"""

TEXT_TFIDF_FALLBACK_TEMPLATE = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Prepare data
text_column = '{text_column}'
target_column = '{target_column}'

X_train = train_df[text_column].fillna('').astype(str).values
y_train_raw = train_df[target_column].values
X_test = test_df[text_column].fillna('').astype(str).values
test_ids = test_df['{id_column}'].values

# Encode labels if they are strings
label_encoder = None
if y_train_raw.dtype == object:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    class_labels = label_encoder.classes_
    print(f"Classes: {{class_labels}}")
else:
    y_train = y_train_raw
    class_labels = None

num_classes = len(np.unique(y_train))
print(f"Number of classes: {{num_classes}}")

# TF-IDF Vectorization with character and word n-grams
print("Creating TF-IDF features...")

# Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    analyzer='word'
)

# Character-level TF-IDF (captures writing style)
char_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(2, 6),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    analyzer='char'
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

# Combine features
from scipy.sparse import hstack
X_train_combined = hstack([X_train_word, X_train_char])
X_test_combined = hstack([X_test_word, X_test_char])

print(f"Combined feature shape: {{X_train_combined.shape}}")

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.1, random_state={seed}, stratify=y_train
)

# Create ensemble models
print("Training ensemble model...")

lr_model = LogisticRegression(
    max_iter=1000,
    C=4.0,
    random_state={seed},
    n_jobs=-1,
    solver='saga',
    multi_class='multinomial'
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state={seed},
    n_jobs=-1,
    objective='multiclass',
    num_class=num_classes
)

# LinearSVC with probability calibration
svc_base = LinearSVC(C=1.0, max_iter=10000, random_state={seed})
svc_model = CalibratedClassifierCV(svc_base, cv=3, method='sigmoid')

# Voting ensemble with soft voting for probability outputs
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('lgb', lgb_model),
        ('svc', svc_model)
    ],
    voting='soft',
    n_jobs=-1
)

# Train ensemble
ensemble.fit(X_train_combined, y_train)

# Make probability predictions
predictions = ensemble.predict_proba(X_test_combined)

# Save predictions - handle both probability and class output formats
if class_labels is not None and len(class_labels) > 2:
    # Multi-class with probability outputs per class
    submission = pd.DataFrame({{'id': test_ids}})
    for i, label in enumerate(class_labels):
        submission[label] = predictions[:, i]
else:
    # Single prediction column
    submission = pd.DataFrame({{
        '{id_column}': test_ids,
        '{prediction_column}': predictions.argmax(axis=1) if predictions.ndim > 1 else predictions
    }})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows and columns: {{list(submission.columns)}}")
print("Training and inference complete!")
"""

TEXT_RESOURCE_CONSTRAINED_TEMPLATE = """
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }}

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

# Prepare data
X_train = train_df['{text_column}'].values
y_train = train_df['{target_column}'].values
X_test = test_df['{text_column}'].values

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state={seed}
)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels={num_classes}
)
model = model.to(device)

# Create datasets with reduced max_length
train_dataset = TextDataset(X_train_split, y_train_split, tokenizer, max_length=128)
val_dataset = TextDataset(X_val_split, y_val_split, tokenizer, max_length=128)
test_dataset = TextDataset(X_test, None, tokenizer, max_length=128)

# Create dataloaders with smaller batch size
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})

# Training loop with gradient accumulation
best_val_loss = float('inf')
patience_counter = 0
accumulation_steps = {gradient_accumulation_steps}

for epoch in range({max_epochs}):
    # Training
    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}, Val Loss = {{avg_val_loss:.4f}}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print(f'Early stopping at epoch {{epoch+1}}')
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.softmax(outputs.logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())

# Save predictions
predictions = np.array(predictions)
submission = pd.DataFrame({{
    '{id_column}': test_df['{id_column}'],
    '{prediction_column}': predictions.argmax(axis=1)
}})
submission.to_csv('submission.csv', index=False)

print("Training and inference complete")
"""


def get_text_template(use_fallback: bool = False, resource_constrained: bool = False) -> str:
    """
    Get appropriate text template based on requirements.
    
    Args:
        use_fallback: Whether to use TF-IDF fallback instead of DistilBERT
        resource_constrained: Whether to use resource-constrained variant
        
    Returns:
        Template string
    """
    if use_fallback:
        return TEXT_TFIDF_FALLBACK_TEMPLATE
    elif resource_constrained:
        return TEXT_RESOURCE_CONSTRAINED_TEMPLATE
    else:
        return TEXT_DISTILBERT_TEMPLATE
