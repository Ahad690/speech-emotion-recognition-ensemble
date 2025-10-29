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
git clone https://github.com/ahad420/speech-emotion-recognition-ensemble.git

# Install dependencies
pip install -r requirements.txt

# Run on Kaggle (recommended for GPU)
# Upload notebook to Kaggle and select GPU T4 x2
```

## Repository Structure
```
├── notebooks/
│   ├── 01_initial_implementation.ipynb
│   ├── 02_gpu_optimization.ipynb
│   └── 03_test_time_augmentation.ipynb
├── reports/
│   ├── ANN_DL_Final_Report.pdf
|   └── ANN_DL_Report.md
├── src/
│   ├── models/
|
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
└── requirements.txt
├── config.yaml
├── README.md
│   
└── requirements.txt
├── .gitignore
├── config_optimized.yaml
└── final_config.yaml

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
```