"""
Speech Emotion Recognition - Inference Script (FIXED)
=====================================================
This script loads the trained ensemble model and predicts emotion from audio files.

Usage:
    python inference.py                          # Uses random sample from sample_audio/
    python inference.py --audio path/to/file.wav # Uses specific audio file
    python inference.py --play                   # Play audio before prediction
    python inference.py --audio file.wav --save  # Saves result image
    python inference.py --debug                  # Show debug information
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
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display

# Try to import IPython for Colab audio playback
try:
    from IPython.display import Audio, display as ipython_display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# ==========================================
# Configuration (MUST MATCH TRAINING!)
# ==========================================
class Config:
    """Configuration for inference - matches training config"""
    sample_rate = 16000
    duration = 3.0
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    n_classes = 7
    dropout = 0.3

config = Config()

# Emotion labels (MUST match training order!)
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
# Model Definitions (EXACT COPY FROM TRAINING)
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
    
    def get_individual_predictions(self, x):
        """Get predictions from each model separately (for debugging)"""
        with torch.no_grad():
            cnn_out = F.softmax(self.cnn(x), dim=1)
            lstm_out = F.softmax(self.lstm(x), dim=1)
            transformer_out = F.softmax(self.transformer(x), dim=1)
        return {
            'cnn': cnn_out.cpu().numpy()[0],
            'lstm': lstm_out.cpu().numpy()[0],
            'transformer': transformer_out.cpu().numpy()[0]
        }


# ==========================================
# Audio Processing (MATCHES TRAINING EXACTLY!)
# ==========================================
def load_and_preprocess_audio(audio_path, config, debug=False):
    """
    Load and preprocess audio file for inference
    MUST MATCH TRAINING PREPROCESSING EXACTLY!
    """
    print(f"Loading audio: {audio_path}")
    
    # Load audio using torchaudio (same as training)
    # Fallback to soundfile backend if torchcodec not available
    try:
        waveform, sr = torchaudio.load(audio_path)
    except (ImportError, RuntimeError) as e:
        if debug:
            print(f"   torchaudio.load() failed: {e}")
            print("   Falling back to soundfile backend...")
        try:
            # Try with soundfile backend
            waveform, sr = torchaudio.load(audio_path, backend='soundfile')
        except Exception:
            # Final fallback: use librosa and convert to torch tensor
            if debug:
                print("   Falling back to librosa...")
            import librosa
            waveform_np, sr = librosa.load(audio_path, sr=None, mono=False)
            # Convert to torch tensor format [channels, samples]
            if waveform_np.ndim == 1:
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                waveform = torch.from_numpy(waveform_np)
    
    if debug:
        print(f"   Original: shape={waveform.shape}, sr={sr}")
    
    # Resample if necessary (same as training)
    if sr != config.sample_rate:
        resampler = T.Resample(sr, config.sample_rate)
        waveform = resampler(waveform)
        if debug:
            print(f"   Resampled to {config.sample_rate} Hz")
    
    # Convert to mono if stereo (same as training)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if debug:
            print(f"   Converted to mono")
    
    # Pad or truncate to fixed duration (same as training)
    target_length = int(config.sample_rate * config.duration)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    if debug:
        print(f"   After pad/truncate: shape={waveform.shape}")
    
    # Extract Mel-spectrogram (EXACTLY as in training!)
    mel_transform = T.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convert to dB (EXACTLY as in training!)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    
    if debug:
        print(f"   Mel-spectrogram shape: {mel_spec_db.shape}")
        print(f"   Mel-spectrogram range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
    
    # Remove channel dimension for return (shape: [n_mels, time])
    mel_spec_db = mel_spec_db.squeeze(0)
    
    # Convert waveform to numpy for visualization
    waveform_np = waveform.squeeze().numpy()
    
    return waveform_np, mel_spec_db


# ==========================================
# Audio Playback
# ==========================================
def play_audio(audio_path, sample_rate=16000):
    """Play audio file - works in Colab/Jupyter"""
    try:
        if IPYTHON_AVAILABLE:
            # Try torchaudio with fallback
            try:
                waveform, sr = torchaudio.load(audio_path)
            except (ImportError, RuntimeError):
                try:
                    waveform, sr = torchaudio.load(audio_path, backend='soundfile')
                except Exception:
                    # Fallback to librosa
                    import librosa
                    waveform_np, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
            if sr != sample_rate:
                resampler = T.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            print("🔊 Playing audio...")
            audio_widget = Audio(waveform.numpy().squeeze(), rate=sample_rate, autoplay=True)
            ipython_display(audio_widget)
            return True
        else:
            print("⚠️  Audio playback only available in Jupyter/Colab")
            return False
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")
        return False


# ==========================================
# Model Loading
# ==========================================
def load_model(model_path, device, debug=False):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = EnsembleModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if debug:
        print(f"   Checkpoint keys: {checkpoint.keys()}")
    
    # Handle DataParallel saved models
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    # Show validation accuracy if available
    if 'val_acc' in checkpoint:
        print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"   Validation Accuracy: Not saved in checkpoint")
    
    # Show epoch if available
    if 'epoch' in checkpoint:
        print(f"   Trained Epochs: {checkpoint['epoch'] + 1}")
    
    # Show ensemble weights
    with torch.no_grad():
        weights = F.softmax(model.weights, dim=0)
        print(f"   Ensemble Weights: CNN={weights[0]:.3f}, LSTM={weights[1]:.3f}, Trans={weights[2]:.3f}")
    
    return model, checkpoint


# ==========================================
# Inference
# ==========================================
def predict_emotion(model, mel_spec, device, debug=False):
    """Predict emotion from mel-spectrogram"""
    
    # mel_spec is already a tensor from preprocessing
    if isinstance(mel_spec, np.ndarray):
        features = torch.FloatTensor(mel_spec).unsqueeze(0).to(device)
    else:
        features = mel_spec.unsqueeze(0).to(device)
    
    if debug:
        print(f"   Input shape to model: {features.shape}")
    
    # Inference
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        if debug:
            # Get individual model predictions
            individual = model.get_individual_predictions(features)
            print("\n   Individual Model Predictions:")
            for model_name, probs in individual.items():
                pred_idx = np.argmax(probs)
                print(f"      {model_name.upper():12s}: {EMOTION_LABELS[pred_idx]:10s} ({probs[pred_idx]*100:.1f}%)")
    
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
    """Create visualization of the prediction"""
    
    # Convert mel_spec to numpy if tensor
    if isinstance(mel_spec, torch.Tensor):
        mel_spec_np = mel_spec.numpy()
    else:
        mel_spec_np = mel_spec
    
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
    img = ax2.imshow(mel_spec_np, aspect='auto', origin='lower', cmap='magma',
                     extent=[0, config.duration, 0, config.n_mels])
    ax2.set_title('Mel-Spectrogram (dB)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Mel Bin', fontsize=12)
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # 3. Emotion Probabilities
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
    
    top3_idx = np.argsort(probabilities)[::-1][:3]
    for i, idx in enumerate(top3_idx):
        emoji = EMOTION_EMOJIS[EMOTION_LABELS[idx]]
        summary_text += f"    ║  {i+1}. {emoji} {EMOTION_LABELS[idx]:<12} {probabilities[idx]*100:>5.1f}%        ║\n"
    
    summary_text += """    ╚═══════════════════════════════════════╝"""
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Always save to prediction_result.png
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Result saved to: prediction_result.png")
    
    if save_path and save_path != 'prediction_result.png':
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Result also saved to: {save_path}")
    
    plt.show()


# ==========================================
# Debug Function
# ==========================================
def debug_checkpoint(model_path):
    """Debug function to inspect checkpoint contents"""
    print("\n" + "=" * 60)
    print("🔍 CHECKPOINT DEBUG INFO")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"\nModel layers ({len(checkpoint[key])} total):")
            for i, (k, v) in enumerate(checkpoint[key].items()):
                if i < 5:
                    print(f"   {k}: {v.shape}")
            if len(checkpoint[key]) > 5:
                print(f"   ... and {len(checkpoint[key]) - 5} more layers")
        elif key == 'config':
            print(f"\nSaved config: {type(checkpoint[key])}")
        else:
            print(f"\n{key}: {checkpoint[key]}")
    
    print("=" * 60)


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
    parser.add_argument('--play', action='store_true',
                        help='Play audio file (works in Colab/Jupyter)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎤 SPEECH EMOTION RECOGNITION - INFERENCE")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find model file
    if not os.path.exists(args.model):
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
    
    # Debug checkpoint if requested
    if args.debug:
        debug_checkpoint(args.model)
    
    # Find audio file
    if args.audio:
        audio_path = args.audio
        if not audio_path.lower().endswith('.wav'):
            print("⚠️  WARNING: Model was trained on .wav files only!")
    else:
        sample_dir = 'sample_audio'
        if os.path.exists(sample_dir):
            audio_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.wav')]
            if audio_files:
                audio_path = os.path.join(sample_dir, random.choice(audio_files))
                print(f"📁 Selected random WAV sample: {audio_path}")
            else:
                print("❌ No WAV files found in sample_audio/")
                sys.exit(1)
        else:
            print("❌ sample_audio/ directory not found")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Load model
    model, checkpoint = load_model(args.model, device, debug=args.debug)
    
    # Process audio (USING TRAINING-MATCHING PREPROCESSING!)
    waveform, mel_spec = load_and_preprocess_audio(audio_path, config, debug=args.debug)
    
    # Play audio if requested
    if args.play:
        print()
        play_audio(audio_path, config.sample_rate)
    
    # Predict
    print("\n🔮 Predicting emotion...")
    predicted_emotion, confidence, probabilities = predict_emotion(model, mel_spec, device, debug=args.debug)
    
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