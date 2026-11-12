# RiskRadar AI – Weapon Detection Model

A YOLOv8 object detection model that detects weapons (guns and knives) in images and CCTV-style frames. It is one of the five specialised models inside **RiskRadar AI**, a team project that analyses street scenes and gives each scene an overall risk score.

I designed and built this model end to end: dataset collection, cleaning and preprocessing, training, evaluation and iterative improvement.

## Results

Validation results after 30 training epochs:

| Metric | Score |
| --- | --- |
| mAP@50 | **90.2%** |
| mAP@50-95 | 60.4% |
| Precision | 90.9% |
| Recall | 84.9% |

![Training curves](results/training_curves.png)

## How it fits into RiskRadar AI

RiskRadar AI runs five detection models on a street scene: weapons, violence, fire, crowd counting and face masks. Their outputs are combined into features and passed to a machine learning classifier that labels the scene as **safe** or **potentially dangerous**. This repository contains the weapon detection part, which was my responsibility in the team.

## Model and training setup

| Setting | Value |
| --- | --- |
| Framework | Ultralytics YOLOv8 |
| Task | Object detection, 1 class (`weapon`) |
| Starting point | Pretrained YOLOv8 weights (transfer learning) |
| Image size | 640 × 640 |
| Batch size | 16 |
| Epochs | 30 (early stopping patience: 10) |
| Training environment | Google Colab (GPU) |

The full training configuration is in [`config/args.yaml`](config/args.yaml) and the per-epoch metrics are in [`results/results.csv`](results/results.csv).

## Dataset

- Images of handguns, rifles, shotguns and knives, including CCTV-style and real-world scenes, labelled with bounding boxes.
- I cleaned and preprocessed the data before training. The label distribution is shown in [`results/label_distribution.jpg`](results/label_distribution.jpg).
- The dataset itself is not included in this repository because of its size and image licensing.

## Repository structure

```
riskradar-weapon-detection/
├── README.md
├── config/
│   └── args.yaml              # training configuration
├── results/
│   ├── results.csv            # metrics per epoch
│   ├── training_curves.png    # metrics and loss curves
│   └── label_distribution.jpg
└── weights/
    └── best.pt                # trained model weights
```

## Usage

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict("your_image.jpg", conf=0.5)
results[0].show()
```

## Tech stack

Python · Ultralytics YOLOv8 · PyTorch · OpenCV · Google Colab

## Author

**Besan Aldhesat** – Data Science & AI student, Mutah University
[LinkedIn](https://linkedin.com/in/besan-aldhesat-271666329) · [GitHub](https://github.com/besan05)
