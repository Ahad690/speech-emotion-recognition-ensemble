"""
Speech Emotion Recognition - Inference Script
=============================================
This script loads the trained ensemble model and predicts emotion from audio files.

Usage:
    python inference.py                          # Uses random sample from sample_audio/
    python inference.py --audio path/to/file.wav # Uses specific audio file
    python inference.py --audio path/to/file.wav --save  # Saves result image
"""

import os
import sys
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display

# ==========================================
# Configuration
# ==========================================
class Config:
    """Configuration for inference"""
    sample_rate = 16000
    duration = 3.0
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    n_classes = 7
    dropout = 0.3

config = Config()

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# Emotion emojis for display
EMOTION_EMOJIS = {
    'neutral': '😐',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'disgust': '🤢',
    'surprise': '😲'
}

# Emotion colors for visualization
EMOTION_COLORS = {
    'neutral': '#95A5A6',
    'happy': '#F1C40F',
    'sad': '#3498DB',
    'angry': '#E74C3C',
    'fear': '#9B59B6',
    'disgust': '#27AE60',
    'surprise': '#E67E22'
}

# ==========================================
# Model Definitions (Same as training)
# ==========================================
class CNNModel(nn.Module):
    """CNN for emotion recognition"""
    
    def __init__(self, config):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.n_classes)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class LSTMModel(nn.Module):
    """LSTM for emotion recognition"""
    
    def __init__(self, config):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=config.n_mels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.n_classes)
        )
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        attended = torch.sum(lstm_out * attn_weights, dim=1)
        
        return self.classifier(attended)


class TransformerModel(nn.Module):
    """Transformer for emotion recognition"""
    
    def __init__(self, config):
        super().__init__()
        
        self.input_projection = nn.Linear(config.n_mels, 256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.n_classes)
        )
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)
        
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)


class EnsembleModel(nn.Module):
    """Ensemble of CNN, LSTM, and Transformer"""
    
    def __init__(self, config):
        super().__init__()
        
        self.cnn = CNNModel(config)
        self.lstm = LSTMModel(config)
        self.transformer = TransformerModel(config)
        
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x)
        transformer_out = self.transformer(x)
        
        w = F.softmax(self.weights, dim=0)
        output = w[0] * cnn_out + w[1] * lstm_out + w[2] * transformer_out
        
        return output


# ==========================================
# Audio Processing
# ==========================================
def load_and_preprocess_audio(audio_path, config):
    """
    Load and preprocess audio file for inference
    """
    print(f"Loading audio: {audio_path}")
    
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    
    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    
    # Pad or truncate to fixed duration
    target_length = int(config.sample_rate * config.duration)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    else:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    
    # Extract Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length
    )
    
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return waveform, mel_spec_db


# ==========================================
# Model Loading
# ==========================================
def load_model(model_path, device):
    """
    Load the trained model
    """
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = EnsembleModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle DataParallel saved models
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    val_acc = checkpoint.get('val_acc', None)
    if val_acc is not None:
        print(f"   Validation Accuracy: {val_acc:.2f}%")
    else:
        print(f"   Validation Accuracy: N/A")
    
    return model


# ==========================================
# Inference
# ==========================================
def predict_emotion(model, mel_spec, device):
    """
    Predict emotion from mel-spectrogram
    """
    # Convert to tensor
    features = torch.FloatTensor(mel_spec).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Get prediction
    predicted_idx = np.argmax(probabilities)
    predicted_emotion = EMOTION_LABELS[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    return predicted_emotion, confidence, probabilities


# ==========================================
# Visualization
# ==========================================
def visualize_prediction(audio_path, waveform, mel_spec, predicted_emotion, 
                         confidence, probabilities, save_path=None):
    """
    Create a comprehensive visualization of the prediction
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Title with emoji
    emoji = EMOTION_EMOJIS.get(predicted_emotion, '🎭')
    fig.suptitle(f'Speech Emotion Recognition Result\n{emoji} {predicted_emotion.upper()} ({confidence*100:.1f}% confidence)', 
                 fontsize=18, fontweight='bold')
    
    # 1. Waveform
    ax1 = fig.add_subplot(2, 2, 1)
    time = np.linspace(0, len(waveform) / config.sample_rate, len(waveform))
    ax1.plot(time, waveform, color='#3498DB', linewidth=0.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, config.duration])
    
    # 2. Mel-Spectrogram
    ax2 = fig.add_subplot(2, 2, 2)
    img = librosa.display.specshow(
        mel_spec, 
        sr=config.sample_rate, 
        hop_length=config.hop_length,
        x_axis='time', 
        y_axis='mel',
        ax=ax2,
        cmap='magma'
    )
    ax2.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # 3. Emotion Probabilities (Horizontal Bar)
    ax3 = fig.add_subplot(2, 2, 3)
    colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
    bars = ax3.barh(EMOTION_LABELS, probabilities * 100, color=colors, edgecolor='black', linewidth=1)
    
    # Highlight predicted emotion
    predicted_idx = EMOTION_LABELS.index(predicted_emotion)
    bars[predicted_idx].set_edgecolor('gold')
    bars[predicted_idx].set_linewidth(3)
    
    ax3.set_xlabel('Probability (%)', fontsize=12)
    ax3.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 100])
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax3.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2, 
                 f'{prob*100:.1f}%', va='center', fontsize=10)
    
    # 4. Prediction Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""
    ╔═══════════════════════════════════════╗
    ║         PREDICTION SUMMARY            ║
    ╠═══════════════════════════════════════╣
    ║  Audio File:                          ║
    ║  {os.path.basename(audio_path)[:35]:<35} ║
    ║                                       ║
    ║  Predicted Emotion: {predicted_emotion.upper():<15} ║
    ║  Confidence: {confidence*100:.2f}%                  ║
    ║                                       ║
    ║  Top 3 Predictions:                   ║
    """
    
    # Get top 3 predictions
    top3_idx = np.argsort(probabilities)[::-1][:3]
    for i, idx in enumerate(top3_idx):
        emoji = EMOTION_EMOJIS[EMOTION_LABELS[idx]]
        summary_text += f"    ║  {i+1}. {emoji} {EMOTION_LABELS[idx]:<12} {probabilities[idx]*100:>5.1f}%        ║\n"
    
    summary_text += """    ╚═══════════════════════════════════════╝"""
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Result saved to: {save_path}")
    
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Result saved to: prediction_result.png")
    
    plt.show()


# ==========================================
# Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Inference')
    parser.add_argument('--audio', type=str, default=None, 
                        help='Path to audio file (default: random from sample_audio/)')
    parser.add_argument('--model', type=str, default='src/models/final_model.pth',
                        help='Path to model file')
    parser.add_argument('--save', action='store_true',
                        help='Save prediction result image')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎤 SPEECH EMOTION RECOGNITION - INFERENCE")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find audio file
    if args.audio:
        audio_path = args.audio
        # Warn if not WAV
        if not audio_path.lower().endswith('.wav'):
            print("⚠️  WARNING: Model was trained on .wav files only!")
            print("   Other formats may work but are not guaranteed.")
            print("   For best results, convert to .wav format.\n")
    else:
        # Use random sample from sample_audio/ (WAV files only)
        sample_dir = 'sample_audio'
        if os.path.exists(sample_dir):
            audio_files = [f for f in os.listdir(sample_dir) 
                          if f.lower().endswith('.wav')]
            if audio_files:
                audio_path = os.path.join(sample_dir, random.choice(audio_files))
                print(f"📁 Selected random WAV sample: {audio_path}")
            else:
                print("❌ No WAV files found in sample_audio/")
                print("   Please add .wav files or use --audio flag")
                sys.exit(1)
        else:
            print("❌ sample_audio/ directory not found")
            print("   Please create it and add audio files, or use --audio flag")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Load model
    if not os.path.exists(args.model):
        # Try alternative paths
        alt_paths = [
            'src/models/final_model.pth',
            'src/models/best_model.pth',
            'models/final_model.pth',
            'final_model.pth',
            'best_model.pth'
        ]
        model_found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                args.model = alt_path
                model_found = True
                break
        
        if not model_found:
            print(f"❌ Model file not found!")
            print(f"   Searched: {args.model} and alternatives")
            sys.exit(1)
    
    model = load_model(args.model, device)
    
    # Process audio
    waveform, mel_spec = load_and_preprocess_audio(audio_path, config)
    
    # Predict
    print("\n🔮 Predicting emotion...")
    predicted_emotion, confidence, probabilities = predict_emotion(model, mel_spec, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    emoji = EMOTION_EMOJIS.get(predicted_emotion, '🎭')
    print(f"   Predicted Emotion: {emoji} {predicted_emotion.upper()}")
    print(f"   Confidence: {confidence * 100:.2f}%")
    print("\n   All Probabilities:")
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
        bar = '█' * int(prob * 30)
        emoji = EMOTION_EMOJIS[label]
        marker = " ← PREDICTED" if label == predicted_emotion else ""
        print(f"   {emoji} {label:10s}: {bar:30s} {prob*100:5.1f}%{marker}")
    print("=" * 60)
    
    # Visualize
    save_path = 'prediction_result.png' if args.save else None
    visualize_prediction(audio_path, waveform, mel_spec, predicted_emotion, 
                        confidence, probabilities, save_path)
    
    print("\n✅ Inference complete!")
    return predicted_emotion, confidence


if __name__ == "__main__":
    main()