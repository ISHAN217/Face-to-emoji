# Computer Vision Projects

A collection of applied computer vision projects spanning vision-language modeling, real-time emotion detection, and sign language recognition.

---


## Project 2 — Emojify: Real-Time Facial Emotion Detection

Detects facial emotions from a webcam feed in real time and replaces the face region with the corresponding emoji.

### How it works

```
Webcam frame → Haar cascade face detection → Grayscale crop (48×48)
            → CNN emotion classifier (7 classes) → Emoji overlay → Display
```

**Model:** 4-layer CNN (Conv2D → MaxPool → Dropout) trained on FER-2013  
**Classes:** Angry · Sad · Surprise · Happy · Neutral · Disgusted · Fear  
**Input:** 48×48 grayscale face crop, normalized to [0,1]

### Quickstart

```bash
pip install tensorflow opencv-python numpy

# Requires: prj.h5 (pretrained weights) + emojis/ folder with 7 PNG files
python Emoji4.py
# Press Q to quit
```

### Files

```
├── Emoji4.py              # Main real-time inference loop
├── prj.h5                 # Pretrained CNN weights
└── emojis/
    ├── angry.png
    ├── sad.png
    ├── surprise.png
    ├── happy.png
    ├── neutral.png
    ├── disgust.png
    └── fear.png
```

---


*Ishan Bhardwaj — github.com/ISHAN217*
