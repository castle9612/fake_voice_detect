# Fake Voice Detection

음성 위변조 탐지를 위한 오디오 특징 추출, MFCC/CNN 실험, TabNet 기반 분류 실험을 정리한 프로젝트입니다.

## Tech Stack

- Python
- Librosa / spafe
- TensorFlow / Keras
- PyTorch / TabNet
- scikit-learn
- pandas / NumPy
- Jupyter Notebook

## Features

- Librosa 기반 MFCC 특징 추출
- 스펙트럼 및 음성 통계 특징 생성
- CNN/RCNN, TabNet 기반 fake/real 분류 실험
- 노트북 실험과 실행용 Python 스크립트 분리

## Results / Highlights

- 원본 음성 파일에서 MFCC, mel-spectrogram, spectrum/statistical feature를 추출하는 전처리 파이프라인을 구성했습니다.
- CNN/RCNN 기반 end-to-end 음성 분류 실험과 TabNet 기반 tabular feature 분류 실험을 분리해 비교할 수 있게 정리했습니다.
- 대용량 MFCC NumPy 배열과 학습된 모델은 공개 저장소에서 제외하고, 재생성 가능한 스크립트만 유지했습니다.
- zero feature 샘플 필터링 스크립트를 포함해 전처리 품질 점검 흐름을 남겼습니다.

Primary outputs when running locally:

- `outputs/mfcc_features.npy`
- `outputs/labels.npy`
- `outputs/cnn_model_test.h5`
- `outputs/submission_rcnn.csv`

## Project Structure

```text
.
├── docs/
│   └── experiment_notes.md
├── notebooks/
│   ├── audio_baseline.ipynb
│   ├── feature_extraction_test.ipynb
│   ├── feature_extraction_train.ipynb
│   ├── mfcc_generation.ipynb
│   ├── pca_lda_test.ipynb
│   ├── pca_lda_train.ipynb
│   ├── preprocessing_workflow.ipynb
│   ├── rcnn_experiment.ipynb
│   ├── semi_supervised_learning.ipynb
│   └── unlabeled_mel_spectrogram.ipynb
├── src/
│   ├── evaluate_rcnn.py
│   ├── extract_mfcc.py
│   ├── filter_zero_features.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Private Data

원본 음성 파일, 전처리 CSV, NumPy 특징 배열, 학습된 모델 파일은 포함하지 않습니다.

```text
data/
├── train.csv
├── test.csv
├── audio/
└── features/
    ├── train_sr_16000.csv
    └── test_sr_44100.csv
```

`train.csv`와 `test.csv`에는 최소한 `id`, `path`, `label` 컬럼이 필요합니다. `path`는 절대 경로이거나 프로젝트 루트 기준 상대 경로를 사용할 수 있습니다.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate MFCC arrays:

```bash
python src/extract_mfcc.py
```

Train a TabNet classifier from precomputed features:

```bash
python src/train.py
```

Run the RCNN experiment:

```bash
python src/evaluate_rcnn.py
```

Generated artifacts are written to `outputs/`.

## Notes

- `data/`, `outputs/`, model checkpoints, NumPy arrays, and generated submissions are ignored by Git.
- Notebooks are kept as experiment records; scripts under `src/` are the cleaned entry points.

## Lessons / Improvements

- 음성 모델링에서는 MFCC/mel-spectrogram 같은 feature를 안정적으로 생성하고 저장하는 과정이 모델 학습만큼 중요했습니다.
- CNN/RCNN과 TabNet을 함께 실험하면서 raw audio representation과 tabular acoustic feature 접근을 비교할 수 있었습니다.
- 다음 단계에서는 fixed validation split, confusion matrix 저장, threshold tuning 결과를 자동 산출하도록 개선할 수 있습니다.
