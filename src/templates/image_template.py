"""Image classification strategy template with EfficientNet and augmentation

Enhanced with:
- Class weighting for imbalanced datasets (melanoma pattern)
- Heavy augmentation pipeline (flip, rotate, CoarseDropout, color jitter)
- GPU-accelerated data loading with pin_memory
- TFRecord-style efficient data pipeline
- EfficientNet-B5 with mixed precision training
"""

IMAGE_STANDARD_TEMPLATE = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device configuration with GPU preference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False, id_col='{id_column}', img_col='{image_column}', target_col='{target_column}'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.id_col = id_col
        self.img_col = img_col
        self.target_col = target_col
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image filename - handle various column naming
        if self.img_col in self.df.columns:
            img_name = row[self.img_col]
        else:
            # Try to find image column
            for col in self.df.columns:
                if 'image' in col.lower() or 'file' in col.lower() or 'path' in col.lower():
                    img_name = row[col]
                    break
            else:
                img_name = row[self.id_col]  # Fallback to ID column
        
        # Handle various image path formats
        img_name = str(img_name)
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            img_name = img_name + '.jpg'
        
        img_path = os.path.join(self.image_dir, img_name)
        
        # Try alternative paths if main path doesn't exist
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = os.path.join(self.image_dir, img_name.rsplit('.', 1)[0] + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Warning: Could not load image {{img_path}}: {{e}}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
            return image, row[self.id_col]
        else:
            label = row[self.target_col]
            return image, label

# =============================================================================
# HEAVY AUGMENTATION PIPELINE (Melanoma-style for medical/rare class detection)
# =============================================================================
train_transform = A.Compose([
    A.Resize({image_size}, {image_size}),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.3),
        A.ElasticTransform(p=0.3),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, 
                   min_height=16, min_width=16, fill_value=0, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize({image_size}, {image_size}),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Detect number of classes
num_classes = {num_classes}
is_binary = num_classes == 2

print(f"Number of classes: {{num_classes}}, Binary: {{is_binary}}")

# Create datasets
train_dataset = ImageDataset(train_df, '{image_dir}', transform=train_transform)
test_dataset = ImageDataset(test_df, '{image_dir}', transform=test_transform, is_test=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=4, pin_memory=True)

# Model - EfficientNet-B5 with proper output for binary/multiclass
model = models.efficientnet_b5(pretrained=True)
num_features = model.classifier[1].in_features

if is_binary:
    # Binary classification: single output with sigmoid
    model.classifier[1] = nn.Linear(num_features, 1)
else:
    # Multi-class classification
    model.classifier[1] = nn.Linear(num_features, num_classes)

model = model.to(device)

# Loss function with class weighting for imbalanced data
if is_binary:
    # Binary cross entropy with logits
    # Calculate class weights if imbalanced
    target_counts = train_df['{target_column}'].value_counts()
    if len(target_counts) == 2:
        pos_weight = torch.tensor([target_counts[0] / target_counts[1]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
else:
    # Multi-class cross entropy
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={max_epochs})

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop
best_loss = float('inf')
patience_counter = 0

for epoch in range({max_epochs}):
    model.train()
    train_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        images = images.to(device)
        
        if is_binary:
            labels = labels.float().unsqueeze(1).to(device)
        else:
            labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    scheduler.step()
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}')
    
    # Early stopping
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
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
ids = []

with torch.no_grad():
    for images, batch_ids in tqdm(test_loader, desc='Inference'):
        images = images.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
            if is_binary:
                # Binary: sigmoid to get probabilities
                preds = torch.sigmoid(outputs).squeeze()
            else:
                # Multi-class: softmax to get probabilities
                preds = torch.softmax(outputs, dim=1)
        
        if is_binary:
            predictions.extend(preds.cpu().numpy().tolist())
        else:
            predictions.extend(preds.cpu().numpy())
        ids.extend(batch_ids)

# Save predictions
if is_binary:
    # For binary classification, save probabilities
    submission = pd.DataFrame({{
        '{id_column}': ids,
        '{prediction_column}': predictions
    }})
else:
    # For multi-class, save class predictions or probabilities based on column name
    predictions = np.array(predictions)
    if '{prediction_column}'.lower() in ['target', 'class', 'label']:
        submission = pd.DataFrame({{
            '{id_column}': ids,
            '{prediction_column}': predictions.argmax(axis=1)
        }})
    else:
        submission = pd.DataFrame({{
            '{id_column}': ids,
            '{prediction_column}': predictions[:, 1] if predictions.shape[1] == 2 else predictions.argmax(axis=1)
        }})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows")
print("Training and inference complete!")
"""

IMAGE_RESOURCE_CONSTRAINED_TEMPLATE = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False, id_col='{id_column}', img_col='{image_column}', target_col='{target_column}'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.id_col = id_col
        self.img_col = img_col
        self.target_col = target_col
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image filename
        if self.img_col in self.df.columns:
            img_name = row[self.img_col]
        else:
            for col in self.df.columns:
                if 'image' in col.lower() or 'file' in col.lower() or 'path' in col.lower():
                    img_name = row[col]
                    break
            else:
                img_name = row[self.id_col]
        
        img_name = str(img_name)
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            img_name = img_name + '.jpg'
        
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = os.path.join(self.image_dir, img_name.rsplit('.', 1)[0] + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Warning: Could not load image {{img_path}}: {{e}}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
            return image, row[self.id_col]
        else:
            label = row[self.target_col]
            return image, label

# Augmentation pipeline (reduced)
train_transform = A.Compose([
    A.Resize({image_size}, {image_size}),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize({image_size}, {image_size}),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Detect number of classes
num_classes = {num_classes}
is_binary = num_classes == 2

# Create datasets
train_dataset = ImageDataset(train_df, '{image_dir}', transform=train_transform)
test_dataset = ImageDataset(test_df, '{image_dir}', transform=test_transform, is_test=True)

# Create dataloaders with smaller batch size
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=2, pin_memory=True)

# Model - EfficientNet-B3 (smaller)
model = models.efficientnet_b3(pretrained=True)
num_features = model.classifier[1].in_features

if is_binary:
    model.classifier[1] = nn.Linear(num_features, 1)
else:
    model.classifier[1] = nn.Linear(num_features, num_classes)

model = model.to(device)

# Loss with class weighting
if is_binary:
    target_counts = train_df['{target_column}'].value_counts()
    if len(target_counts) == 2:
        pos_weight = torch.tensor([target_counts[0] / target_counts[1]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={max_epochs})

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop with gradient accumulation
best_loss = float('inf')
patience_counter = 0
accumulation_steps = {gradient_accumulation_steps}

for epoch in range({max_epochs}):
    model.train()
    train_loss = 0.0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}')):
        images = images.to(device)
        
        if is_binary:
            labels = labels.float().unsqueeze(1).to(device)
        else:
            labels = labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
    
    avg_train_loss = train_loss / len(train_loader)
    scheduler.step()
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}')
    
    # Early stopping
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
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
ids = []

with torch.no_grad():
    for images, batch_ids in tqdm(test_loader, desc='Inference'):
        images = images.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
            if is_binary:
                preds = torch.sigmoid(outputs).squeeze()
            else:
                preds = torch.softmax(outputs, dim=1)
        
        if is_binary:
            predictions.extend(preds.cpu().numpy().tolist())
        else:
            predictions.extend(preds.cpu().numpy())
        ids.extend(batch_ids)

# Save predictions
if is_binary:
    submission = pd.DataFrame({{
        '{id_column}': ids,
        '{prediction_column}': predictions
    }})
else:
    predictions = np.array(predictions)
    if '{prediction_column}'.lower() in ['target', 'class', 'label']:
        submission = pd.DataFrame({{
            '{id_column}': ids,
            '{prediction_column}': predictions.argmax(axis=1)
        }})
    else:
        submission = pd.DataFrame({{
            '{id_column}': ids,
            '{prediction_column}': predictions[:, 1] if predictions.shape[1] == 2 else predictions.argmax(axis=1)
        }})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows")
print("Training and inference complete!")
"""


def get_image_template(resource_constrained: bool = False) -> str:
    """
    Get image template.
    
    Args:
        resource_constrained: Ignored - always use full template
        
    Returns:
        Template string
    """
    # Always use the full template - LLM optimizes as needed
    return IMAGE_STANDARD_TEMPLATE
