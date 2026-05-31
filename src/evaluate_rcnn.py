import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, TimeDistributed, Bidirectional
from tqdm import tqdm
from pathlib import Path

# ?곗씠??寃쎈줈
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
train_csv = DATA_DIR / "train.csv"
test_csv = DATA_DIR / "test.csv"

# CSV ?뚯씪 濡쒕뱶
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# MFCC ?뱀꽦 異붿텧 ?⑥닔
def extract_features(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=32000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs


def resolve_audio_path(file_path):
    path = Path(str(file_path).strip())
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

# train ?곗씠???꾩쿂由?
train_df['features'] = train_df['path'].apply(lambda x: extract_features(resolve_audio_path(x)))
X = np.array([x for x in tqdm(train_df['features'])])
X = np.expand_dims(X, -1)  # CNN ?낅젰 ?뺥깭??留욎텛湲??꾪빐 李⑥썝 異붽?
y = pd.get_dummies(train_df['label']).values

# test ?곗씠???꾩쿂由?
test_df['features'] = test_df['path'].apply(lambda x: extract_features(resolve_audio_path(x)))
X_test = np.array([x for x in tqdm(test_df['features'])])
X_test = np.expand_dims(X_test, -1)  # CNN ?낅젰 ?뺥깭??留욎텛湲??꾪빐 李⑥썝 異붽?


# ?숈뒿 ?곗씠?곗? 寃利??곗씠??遺꾪븷
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 紐⑤뜽 ?뺤쓽
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 紐⑤뜽 而댄뙆??
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 紐⑤뜽 ?숈뒿
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# ?덉륫
predictions = model.predict(X_test)
test_df['fake'] = predictions[:, 0]
test_df['real'] = predictions[:, 1]

# ?쒖텧 ?뚯씪 ?묒꽦
submission_df = test_df[['id', 'fake', 'real']]
OUTPUT_DIR.mkdir(exist_ok=True)
submission_df.to_csv(OUTPUT_DIR / 'submission_rcnn.csv', index=False)

