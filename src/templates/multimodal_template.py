"""Multimodal strategy template with dual encoder architecture"""

MULTIMODAL_DUAL_ENCODER_TEMPLATE = """
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
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dual Encoder Model
class DualEncoderModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes, image_output_dim=512):
        super(DualEncoderModel, self).__init__()
        
        # Image encoder - EfficientNet-B5
        self.image_encoder = models.efficientnet_b5(pretrained=True)
        num_features = self.image_encoder.classifier[1].in_features
        self.image_encoder.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, image_output_dim)
        )
        
        # Tabular encoder - MLP
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion and classifier
        fusion_dim = image_output_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, tabular_features):
        # Encode image
        image_features = self.image_encoder(images)
        
        # Encode tabular
        tabular_features = self.tabular_encoder(tabular_features)
        
        # Concatenate fusion
        fused = torch.cat([image_features, tabular_features], dim=1)
        
        # Classify
        output = self.classifier(fused)
        return output

# Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, df, image_dir, tabular_cols, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.tabular_cols = tabular_cols
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['{image_column}'])
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get tabular features
        tabular = torch.tensor(row[self.tabular_cols].values.astype(np.float32))
        
        if self.is_test:
            return image, tabular, row['{id_column}']
        else:
            label = row['{target_column}']
            return image, tabular, label

# Augmentation pipeline
train_transform = A.Compose([
    A.Resize({image_size}, {image_size}),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
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

# Identify tabular columns (exclude image, id, target)
exclude_cols = ['{image_column}', '{id_column}', '{target_column}']
tabular_cols = [col for col in train_df.columns if col not in exclude_cols]

# Handle missing values in tabular features
train_df[tabular_cols] = train_df[tabular_cols].fillna(train_df[tabular_cols].median())
test_df[tabular_cols] = test_df[tabular_cols].fillna(train_df[tabular_cols].median())

# Normalize tabular features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[tabular_cols] = scaler.fit_transform(train_df[tabular_cols])
test_df[tabular_cols] = scaler.transform(test_df[tabular_cols])

# Create datasets
train_dataset = MultimodalDataset(train_df, '{image_dir}', tabular_cols, transform=train_transform)
test_dataset = MultimodalDataset(test_df, '{image_dir}', tabular_cols, transform=test_transform, is_test=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=4)

# Initialize model
num_tabular_features = len(tabular_cols)
model = DualEncoderModel(num_tabular_features, {num_classes})
model = model.to(device)

# Loss and optimizer (FocalLoss for imbalanced data)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss()
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
    
    for images, tabular, labels in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images, tabular)
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
    for images, tabular, batch_ids in tqdm(test_loader, desc='Inference'):
        images = images.to(device)
        tabular = tabular.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, tabular)
            preds = torch.softmax(outputs, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        ids.extend(batch_ids)

# Save predictions
predictions = np.array(predictions)
submission = pd.DataFrame({{
    '{id_column}': ids,
    '{prediction_column}': predictions.argmax(axis=1)
}})
submission.to_csv('submission.csv', index=False)

print("Training and inference complete")
"""

MULTIMODAL_RESOURCE_CONSTRAINED_TEMPLATE = """
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
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dual Encoder Model (smaller)
class DualEncoderModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes, image_output_dim=256):
        super(DualEncoderModel, self).__init__()
        
        # Image encoder - EfficientNet-B3 (smaller)
        self.image_encoder = models.efficientnet_b3(pretrained=True)
        num_features = self.image_encoder.classifier[1].in_features
        self.image_encoder.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, image_output_dim)
        )
        
        # Tabular encoder - MLP (smaller)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion and classifier
        fusion_dim = image_output_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, tabular_features):
        # Encode image
        image_features = self.image_encoder(images)
        
        # Encode tabular
        tabular_features = self.tabular_encoder(tabular_features)
        
        # Concatenate fusion
        fused = torch.cat([image_features, tabular_features], dim=1)
        
        # Classify
        output = self.classifier(fused)
        return output

# Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, df, image_dir, tabular_cols, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.tabular_cols = tabular_cols
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['{image_column}'])
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get tabular features
        tabular = torch.tensor(row[self.tabular_cols].values.astype(np.float32))
        
        if self.is_test:
            return image, tabular, row['{id_column}']
        else:
            label = row['{target_column}']
            return image, tabular, label

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

# Identify tabular columns (exclude image, id, target)
exclude_cols = ['{image_column}', '{id_column}', '{target_column}']
tabular_cols = [col for col in train_df.columns if col not in exclude_cols]

# Feature selection - top 20 tabular features
if len(tabular_cols) > 20:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, random_state={seed})
    train_df_temp = train_df[tabular_cols].fillna(train_df[tabular_cols].median())
    rf.fit(train_df_temp, train_df['{target_column}'])
    feature_importance = pd.DataFrame({{
        'feature': tabular_cols,
        'importance': rf.feature_importances_
    }}).sort_values('importance', ascending=False)
    tabular_cols = feature_importance.head(20)['feature'].tolist()

# Handle missing values in tabular features
train_df[tabular_cols] = train_df[tabular_cols].fillna(train_df[tabular_cols].median())
test_df[tabular_cols] = test_df[tabular_cols].fillna(train_df[tabular_cols].median())

# Normalize tabular features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[tabular_cols] = scaler.fit_transform(train_df[tabular_cols])
test_df[tabular_cols] = scaler.transform(test_df[tabular_cols])

# Create datasets
train_dataset = MultimodalDataset(train_df, '{image_dir}', tabular_cols, transform=train_transform)
test_dataset = MultimodalDataset(test_df, '{image_dir}', tabular_cols, transform=test_transform, is_test=True)

# Create dataloaders with smaller batch size
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=2)

# Initialize model
num_tabular_features = len(tabular_cols)
model = DualEncoderModel(num_tabular_features, {num_classes})
model = model.to(device)

# Loss and optimizer
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss()
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
    
    for i, (images, tabular, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}')):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, tabular)
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
    for images, tabular, batch_ids in tqdm(test_loader, desc='Inference'):
        images = images.to(device)
        tabular = tabular.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, tabular)
            preds = torch.softmax(outputs, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        ids.extend(batch_ids)

# Save predictions
predictions = np.array(predictions)
submission = pd.DataFrame({{
    '{id_column}': ids,
    '{prediction_column}': predictions.argmax(axis=1)
}})
submission.to_csv('submission.csv', index=False)

print("Training and inference complete")
"""


def get_multimodal_template(resource_constrained: bool = False) -> str:
    """
    Get multimodal template.
    
    Args:
        resource_constrained: Ignored - always use full template
        
    Returns:
        Template string
    """
    # Always use the full template - LLM optimizes as needed
    return MULTIMODAL_DUAL_ENCODER_TEMPLATE
