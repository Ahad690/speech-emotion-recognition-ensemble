# Speech Emotion Recognition using Ensemble Deep Learning

## Project Overview
**Course**: Artificial Neural Networks & Deep Learning (CS380)  
**Semester**: 5th Semester, BSAI  
**Author**: Ahad Imran  
**Accuracy**: 66.74% on test set  
**Architecture**: CNN + LSTM + Transformer Ensemble  

## Key Results
- **Dataset**: 11,682 samples (RAVDESS, TESS, CREMA-D)
- **Classes**: 7 emotions (neutral, happy, sad, angry, fear, disgust, surprise)
- **Best Performance**: Surprise (88.8%), Angry (76.8%)
- **Training Time**: 19s/epoch with dual GPU optimization

## Quick Start
```bash
# Clone repository
git clone https://github.com/Ahad690/speech-emotion-recognition-ensemble.git

# Install dependencies
pip install -r requirements.txt

# Run on Kaggle (recommended for GPU)
# Upload notebook to Kaggle and select GPU T4 x2
```

## 🚀 Quick Inference (Google Colab)

Run emotion recognition on any audio file:

```python
# ============================================
# Speech Emotion Recognition - Complete Setup & Inference
# ============================================
import os

# 1. Clone Repository
REPO_NAME = 'speech-emotion-recognition-ensemble'
GITHUB_USER = 'Ahad690'

if not os.path.exists(REPO_NAME):
    print("📥 Cloning repository...")
    !git clone https://github.com/{GITHUB_USER}/{REPO_NAME}.git
else:
    print("📁 Repository already exists, updating...")
    %cd {REPO_NAME}
    !git pull
    %cd ..
    
%cd {REPO_NAME}

# 2. Install Dependencies (including soundfile for torchaudio fallback)
print("\n📦 Installing dependencies...")
!pip install -q torch torchaudio librosa matplotlib soundfile numpy scipy

# 3. Run Inference (randomly selects a .wav file from sample_audio/)
print("\n🔮 Running emotion recognition on random sample...")
!python inference.py

# 4. Play Audio Manually (plays the SAME file that was analyzed)
print("\n🔊 Playing audio manually...")
from IPython.display import Audio, display

# Read the audio file that was just analyzed
if os.path.exists('.last_audio.txt'):
    with open('.last_audio.txt', 'r') as f:
        audio_file = f.read().strip()
    print(f"Playing analyzed file: {audio_file}")
    display(Audio(audio_file))
else:
    print("⚠️  Could not find .last_audio.txt - file path not saved")

# 5. Display Result
print("\n📊 Displaying result...")
from IPython.display import Image
if os.path.exists('prediction_result.png'):
    display(Image('prediction_result.png'))
else:
    print("❌ Result image not found")

print("\n✅ Complete! Run again to test a different random sample.")
```

### Using Custom Audio File

```python
# Upload your audio file
from google.colab import files
uploaded = files.upload()

# Get filename
audio_file = list(uploaded.keys())[0]

# Run inference
!python inference.py --audio {audio_file}

# Display result
from IPython.display import Image, display
display(Image('prediction_result.png'))
```

### Command Line Options

```bash
# Use random sample (randomly selects a .wav file from sample_audio/)
python inference.py

# Use specific audio file
python inference.py --audio path/to/your/audio.wav

# Play audio automatically (works in Colab/Jupyter)
python inference.py --play

# Show debug information
python inference.py --debug

# Save result image
python inference.py --audio audio.wav --save
```

### Alternative: Automatic Audio Playback

```python
# Run inference with automatic audio playback
!python inference.py --play

# Display result
from IPython.display import Image
if os.path.exists('prediction_result.png'):
    display(Image('prediction_result.png'))
```

### Kaggle Notebook
[View on Kaggle](https://www.kaggle.com/code/mahad69/speech-emotion-recognition-ensemble)

## Repository Structure
```
speech-emotion-recognition-ensemble/
├── .gitignore
├── README.md                          # Updated with inference instructions
├── requirements.txt                   # Updated with inference deps
├── inference.py                       # Main inference script
├── config.yaml
├── config_optimized.yaml
├── final_config.yaml
├── notebooks/
│   ├── 01_initial_implementation.ipynb
│   ├── 02_gpu_optimization.ipynb
│   └── 03_test_time_augmentation.ipynb
├── report/
│   ├── ANN_DL_Final_Report.pdf
│   └── ANN_DL_Report.md
├── results/
│   ├── confidence_distribution.png
│   ├── confusion_matrix.png
│   ├── ensemble_model_contribution_weights.png
│   ├── per-class_accuracy.png
│   ├── results.json
│   ├── top_20_most_important_features.png
│   └── training_history.png
├── sample_audio/                      # Sample audio files
│   ├── README.md
│   └── *.wav                          # Audio samples for testing
└── src/
    ├── augmentation.py
    └── models/
        ├── best_model.pth
        └── final_model.pth
```

## Performance Metrics
| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| CNN | 62.01% | 1.1M | 15s/epoch |
| LSTM | 67.66% | 1.2M | 18s/epoch |
| Transformer | 45.07% | 1.0M | 20s/epoch |
| **Ensemble** | **66.74%** | **3.3M** | **19s/epoch** |

## Technical Highlights
- Multi-GPU training with DataParallel
- Mixed precision (FP16) computation
- CosineAnnealingWarmRestarts scheduler
- Test Time Augmentation
- Attention mechanisms in LSTM and Transformer

## Citation
If you use this code, please cite:
```bibtex
@misc{AhadImran2025speech,
  title={Speech Emotion Recognition using Ensemble Deep Learning},
  author={[Ahad Imran]},
  year={2025},
  school={[NUTECH]}
}
```

## 📄 License
This project is for educational purposes. MIT License.
