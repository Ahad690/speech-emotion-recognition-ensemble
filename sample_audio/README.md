# Sample Audio Files

Place your audio files here for testing the emotion recognition model.

## Supported Formats
- `.wav` (recommended)
- `.mp3`
- `.flac`
- `.ogg`

## Requirements
- Duration: 1-5 seconds (will be padded/truncated to 3 seconds)
- Content: Clear speech with emotional expression

## Sample Sources
You can download sample audio from:
1. [RAVDESS Dataset](https://zenodo.org/record/1188976) (CC BY-NC-SA 4.0)
2. [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)
3. Record your own voice samples!

## Filename Convention (Optional)
For better organization, use this naming:
```
<emotion>_<description>.wav
Example: happy_greeting.wav, angry_complaint.wav
```

## Usage
The inference script will randomly select a WAV file from this directory if no specific file is provided:
```bash
python inference.py  # Randomly selects a .wav file from sample_audio/
python inference.py --audio sample_audio/your_file.wav  # Use specific file
```

