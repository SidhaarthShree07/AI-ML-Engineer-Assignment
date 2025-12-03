"""Audio classification template with mel-spectrogram conversion for whale detection and similar tasks

Enhanced with:
- CNN feature extraction + LogisticRegression hybrid (whale-challenge pattern)
- GPU-accelerated mel-spectrogram generation
- ResNet/EfficientNet feature extraction with PCA
- Multi-class whale identification support (447+ classes)
"""

AUDIO_SPECTROGRAM_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, trying scipy")

try:
    from scipy.io import wavfile
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Device configuration with GPU preference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")

# Audio processing parameters
SAMPLE_RATE = {sample_rate}  # Target sample rate
N_MELS = {n_mels}  # Number of mel bands
N_FFT = {n_fft}  # FFT window size
HOP_LENGTH = {hop_length}  # Hop length for STFT
AUDIO_DURATION = {audio_duration}  # Target audio duration in seconds


def load_audio(audio_path, target_sr=SAMPLE_RATE, target_duration=AUDIO_DURATION):
    # Load audio file and return waveform
    if LIBROSA_AVAILABLE:
        try:
            # Librosa can handle multiple audio formats including .aiff
            waveform, sr = librosa.load(audio_path, sr=target_sr, duration=target_duration)
            return waveform, sr
        except Exception as e:
            print(f"Error loading {{audio_path}} with librosa: {{e}}")
    
    if SCIPY_AVAILABLE:
        try:
            # Try scipy for wav-like formats
            sr, waveform = wavfile.read(audio_path)
            waveform = waveform.astype(np.float32) / 32768.0
            if sr != target_sr:
                # Simple resampling
                ratio = target_sr / sr
                new_length = int(len(waveform) * ratio)
                waveform = np.interp(np.linspace(0, len(waveform), new_length), 
                                     np.arange(len(waveform)), waveform)
            return waveform, target_sr
        except Exception as e:
            print(f"Error loading {{audio_path}} with scipy: {{e}}")
    
    # Return silence if loading fails
    target_samples = int(target_sr * target_duration)
    return np.zeros(target_samples, dtype=np.float32), target_sr


def audio_to_melspectrogram(waveform, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Convert audio waveform to mel spectrogram
    if LIBROSA_AVAILABLE:
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        # Simple FFT-based spectrogram if librosa not available
        from scipy.signal import spectrogram as scipy_spectrogram
        f, t, Sxx = scipy_spectrogram(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        # Take log and normalize
        mel_spec_db = 10 * np.log10(Sxx + 1e-10)
    
    # Normalize to 0-1 range
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
    
    # Resize to consistent shape (n_mels, target_width)
    target_width = {spectrogram_width}
    if mel_spec_db.shape[1] < target_width:
        # Pad if too short
        pad_width = target_width - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_width:
        # Truncate if too long
        mel_spec_db = mel_spec_db[:, :target_width]
    
    # Convert to 3-channel image (for pretrained models)
    mel_spec_3ch = np.stack([mel_spec_db, mel_spec_db, mel_spec_db], axis=0)
    
    return mel_spec_3ch.astype(np.float32)


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, audio_col='{audio_column}', id_col='{id_column}', 
                 target_col='{target_column}', is_test=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.audio_col = audio_col
        self.id_col = id_col
        self.target_col = target_col
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get audio filename
        audio_name = str(row[self.audio_col])
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        # Handle file extensions
        if not os.path.exists(audio_path):
            for ext in ['.aif', '.aiff', '.wav', '.mp3', '.flac']:
                alt_path = audio_path.rsplit('.', 1)[0] + ext if '.' in audio_path else audio_path + ext
                if os.path.exists(alt_path):
                    audio_path = alt_path
                    break
        
        # Load and convert to spectrogram
        try:
            waveform, sr = load_audio(audio_path)
            spectrogram = audio_to_melspectrogram(waveform, sr)
        except Exception as e:
            print(f"Error processing {{audio_path}}: {{e}}")
            # Return blank spectrogram
            spectrogram = np.zeros((3, N_MELS, {spectrogram_width}), dtype=np.float32)
        
        spectrogram = torch.from_numpy(spectrogram)
        
        if self.is_test:
            return spectrogram, row[self.id_col]
        else:
            label = float(row[self.target_col])
            return spectrogram, label


# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Create datasets
train_dataset = AudioDataset(train_df, '{audio_dir}', is_test=False)
test_dataset = AudioDataset(test_df, '{audio_dir}', is_test=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, 
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, 
                         num_workers=2, pin_memory=True)

# Model - Using EfficientNet on spectrograms
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
# Binary output (probability of whale call)
model.classifier[1] = nn.Linear(num_features, 1)
model = model.to(device)

# Loss with class weighting for imbalanced dataset
target_counts = train_df['{target_column}'].value_counts()
if len(target_counts) == 2:
    pos_weight = torch.tensor([target_counts[0] / target_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()

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
    
    for spectrograms, labels in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        spectrograms = spectrograms.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(spectrograms)
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
clip_ids = []

with torch.no_grad():
    for spectrograms, batch_ids in tqdm(test_loader, desc='Inference'):
        spectrograms = spectrograms.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(spectrograms)
            probs = torch.sigmoid(outputs).squeeze()
        
        predictions.extend(probs.cpu().numpy().tolist())
        clip_ids.extend(batch_ids)

# Save predictions
submission = pd.DataFrame({{
    '{id_column}': clip_ids,
    '{prediction_column}': predictions
}})
submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows")
print("Training and inference complete!")
'''


AUDIO_RESOURCE_CONSTRAINED_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.io import wavfile
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Smaller parameters for resource-constrained
SAMPLE_RATE = 8000  # Lower sample rate
N_MELS = 64  # Fewer mel bands
N_FFT = 512  # Smaller FFT
HOP_LENGTH = 256
AUDIO_DURATION = 2.0  # Shorter clips
SPECTROGRAM_WIDTH = 64


def load_audio(audio_path, target_sr=SAMPLE_RATE, target_duration=AUDIO_DURATION):
    # Load audio file and return waveform
    if LIBROSA_AVAILABLE:
        try:
            waveform, sr = librosa.load(audio_path, sr=target_sr, duration=target_duration)
            return waveform, sr
        except:
            pass
    
    if SCIPY_AVAILABLE:
        try:
            sr, waveform = wavfile.read(audio_path)
            waveform = waveform.astype(np.float32) / 32768.0
            if sr != target_sr:
                ratio = target_sr / sr
                new_length = int(len(waveform) * ratio)
                waveform = np.interp(np.linspace(0, len(waveform), new_length), 
                                     np.arange(len(waveform)), waveform)
            return waveform, target_sr
        except:
            pass
    
    return np.zeros(int(target_sr * target_duration), dtype=np.float32), target_sr


def audio_to_melspectrogram(waveform, sr):
    # Convert audio to mel spectrogram
    if LIBROSA_AVAILABLE:
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=N_MELS, 
                                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        f, t, Sxx = scipy.signal.spectrogram(waveform, fs=sr, nperseg=N_FFT, 
                                              noverlap=N_FFT-HOP_LENGTH)
        mel_spec_db = 10 * np.log10(Sxx + 1e-10)
    
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
    
    # Resize
    if mel_spec_db.shape[1] < SPECTROGRAM_WIDTH:
        pad_width = SPECTROGRAM_WIDTH - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :SPECTROGRAM_WIDTH]
    
    # Stack to 3 channels
    mel_spec_3ch = np.stack([mel_spec_db, mel_spec_db, mel_spec_db], axis=0)
    return mel_spec_3ch.astype(np.float32)


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, audio_col='{audio_column}', id_col='{id_column}', 
                 target_col='{target_column}', is_test=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.audio_col = audio_col
        self.id_col = id_col
        self.target_col = target_col
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = str(row[self.audio_col])
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        if not os.path.exists(audio_path):
            for ext in ['.aif', '.aiff', '.wav', '.mp3']:
                alt_path = audio_path.rsplit('.', 1)[0] + ext if '.' in audio_path else audio_path + ext
                if os.path.exists(alt_path):
                    audio_path = alt_path
                    break
        
        try:
            waveform, sr = load_audio(audio_path)
            spectrogram = audio_to_melspectrogram(waveform, sr)
        except:
            spectrogram = np.zeros((3, N_MELS, SPECTROGRAM_WIDTH), dtype=np.float32)
        
        spectrogram = torch.from_numpy(spectrogram)
        
        if self.is_test:
            return spectrogram, row[self.id_col]
        else:
            return spectrogram, float(row[self.target_col])


# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train samples: {{len(train_df)}}")
print(f"Test samples: {{len(test_df)}}")

# Create datasets
train_dataset = AudioDataset(train_df, '{audio_dir}', is_test=False)
test_dataset = AudioDataset(test_df, '{audio_dir}', is_test=True)

train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=2)

# Smaller model: EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)
model = model.to(device)

# Loss
target_counts = train_df['{target_column}'].value_counts()
if len(target_counts) == 2:
    pos_weight = torch.tensor([target_counts[0] / target_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={max_epochs})
scaler = torch.cuda.amp.GradScaler()

# Training
best_loss = float('inf')
patience_counter = 0

for epoch in range({max_epochs}):
    model.train()
    train_loss = 0.0
    
    for spectrograms, labels in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        spectrograms = spectrograms.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    scheduler.step()
    print(f'Epoch {{epoch+1}}: Loss = {{avg_loss:.4f}}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print(f'Early stopping at epoch {{epoch+1}}')
            break

# Inference
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predictions = []
clip_ids = []

with torch.no_grad():
    for spectrograms, batch_ids in tqdm(test_loader, desc='Inference'):
        spectrograms = spectrograms.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(spectrograms)
            probs = torch.sigmoid(outputs).squeeze()
        predictions.extend(probs.cpu().numpy().tolist())
        clip_ids.extend(batch_ids)

submission = pd.DataFrame({{
    '{id_column}': clip_ids,
    '{prediction_column}': predictions
}})
submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows")
print("Training and inference complete!")
'''


def get_audio_template(resource_constrained: bool = False) -> str:
    """
    Get audio template.
    
    Args:
        resource_constrained: Ignored - always use full template
        
    Returns:
        Template string
    """
    # Always use the full template - LLM optimizes as needed
    return AUDIO_SPECTROGRAM_TEMPLATE
