#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import librosa
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# In[2]:


# ?꾩쿂由?諛??뱀쭠 異붿텧 ?뚮씪誘명꽣
sampling_rate = 16000  # ?섑뵆留??띾룄
fft_size = 1024  # FFT ?ъ씠利?
window_length = int(0.025 * sampling_rate)  # ?덈룄??湲몄씠 (25ms)
hop_length = int(0.010 * sampling_rate)  # ??湲몄씠 (10ms)
n_mfcc = 100

# In[3]:


def load_file(filename, sampling_rate=16000):
    data, sr = librosa.load(filename, sr=sampling_rate)
    return data, sr

# In[4]:


def remove_silence(y, sr, top_db=20):
    # 臾댁쓬 援ш컙???쒖옉怨??앹쓣 李얘린
    intervals = librosa.effects.split(y, top_db=top_db)

    # ?좏슚???ㅻ뵒???좏샇瑜??⑹튂湲?
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

    return y_trimmed

# In[5]:


def mel_spectrogram_generator(data, sr, fft_size, hop_length):
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=fft_size, hop_length=hop_length, window='hamming')
    return mel_spectrogram

# In[6]:


def normalize_spectrogram(mel_spectrogram):
    scaler = MinMaxScaler()
    mel_spectrogram_norm = scaler.fit_transform(mel_spectrogram.T).T
    return mel_spectrogram_norm

# In[7]:


def compute_log_mel_spectrogram(mel_spectrogram_norm):
    return np.log(mel_spectrogram_norm + 1e-6)

# In[8]:


def apply_dct(log_mel_spectrogram, n_mfcc):
    return scipy.fftpack.dct(log_mel_spectrogram, type=2, axis=1, norm='ortho')[:, :n_mfcc]

# In[9]:


def plot_mfcc(mfcc, sampling_rate, hop_length, vmin=1, vmax=-1):
    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc.T, aspect='auto', origin='lower', 
               extent=[0, mfcc.shape[0] * hop_length / sampling_rate, 0, mfcc.shape[1]], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

# In[10]:


def compute_mfcc(filename, sampling_rate=16000, fft_size=1024, window_length=400, hop_length=160, n_mfcc=13):
    data, sr = load_file(filename, sampling_rate)

    # 臾댁쓬 ?쒓굅
    data = remove_silence(data, sr)
    
    mel_spectrogram = mel_spectrogram_generator(data, sr, fft_size, hop_length)
    mel_spectrogram_norm = normalize_spectrogram(mel_spectrogram)
    log_mel_spectrogram = compute_log_mel_spectrogram(mel_spectrogram_norm)
    mfcc = apply_dct(log_mel_spectrogram, n_mfcc)
    
    return mfcc

# In[11]:


df = pd.read_csv(DATA_DIR / 'train.csv')
file_paths = df['path'].values
labels = df['label'].apply(lambda x: 1 if x == 'real' else 0).values
OUTPUT_DIR.mkdir(exist_ok=True)

# In[12]:


mfcc_features = []
for path in tqdm(file_paths):
    mfcc = compute_mfcc(path, sampling_rate, fft_size, window_length, hop_length, n_mfcc)
    mfcc_features.append(mfcc)

# In[45]:


max_length = max(mfcc.shape[0] for mfcc in mfcc_features)
max_length

# In[42]:


max_length = max(mfcc.shape[1] for mfcc in mfcc_features)
max_length

# In[48]:


max_features = n_mfcc 

# In[50]:


padded_mfcc_features = []

for mfcc in tqdm(mfcc_features):
    pad_width = max_length - mfcc.shape[0]
    pad_feature_width = max_features - mfcc.shape[1] if mfcc.shape[1] < max_features else 0
    mfcc = np.pad(mfcc, ((0, pad_width), (0, pad_feature_width)), mode='constant')
    padded_mfcc_features.append(mfcc)

# In[61]:


df['label'] = df['label'].apply(lambda x: 1 if x == 'real' else 0)
labels = df['label'].values

# In[51]:


padded_mfcc_features = np.array(padded_mfcc_features)
labels = np.array(labels)

# In[180]:


from keras.utils import to_categorical

# In[182]:


labels = to_categorical(labels, num_classes=2)

# In[183]:


labels

# In[184]:


np.save(OUTPUT_DIR / 'mfcc_features.npy', padded_mfcc_features)
np.save(OUTPUT_DIR / 'labels.npy', labels)

# In[53]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# In[54]:


mfcc_features = np.load(OUTPUT_DIR / 'mfcc_features.npy')
labels = np.load(OUTPUT_DIR / 'labels.npy')

# In[55]:


padded_mfcc_features = padded_mfcc_features[..., np.newaxis]

# In[56]:


# labels = to_categorical(labels, )

# In[185]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(padded_mfcc_features, labels, test_size=0.2, random_state=42)

# In[106]:


from keras.layers import Conv2D , MaxPool2D , Input , GlobalAveragePooling2D ,AveragePooling2D, Dense , Dropout ,Activation , BatchNormalization

# In[115]:


from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers
from keras.layers import concatenate
from tensorflow.keras.utils import plot_model

# In[188]:


def InceptionV4():
    
    input_layer = Input(shape=(128, 100, 1))
    
    x = stemBlock(prev_layer=input_layer)
    
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    x = InceptionBlock_A(prev_layer=x)
    
    x = reduction_A_Block(prev_layer=x)
    
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    x = InceptionBlock_B(prev_layer=x)
    
    x = reduction_B_Block(prev_layer= x)
    
    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)
    x = InceptionBlock_C(prev_layer=x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(units = 1536, activation='relu') (x)
    x = Dropout(rate = 0.8) (x)
    x = Dense(units = 2, activation='sigmoid')(x)
    
    model = Model(inputs = input_layer , outputs = x , name ='Inception-V4')
    
    return model


def conv2d_with_Batch(prev_layer , nbr_kernels , filter_size , strides = (1,1) , padding = 'same'):
    x = Conv2D(filters = nbr_kernels, kernel_size = filter_size, strides=strides , padding=padding) (prev_layer)
    x = BatchNormalization()(x)
    x = Activation(activation = 'relu') (x)
    return x


def stemBlock(prev_layer):
    x = conv2d_with_Batch(prev_layer, nbr_kernels = 32, filter_size = (3,3), strides = (2,2))
    x = conv2d_with_Batch(x, nbr_kernels = 32, filter_size = (3,3))
    x = conv2d_with_Batch(x, nbr_kernels = 64, filter_size = (3,3))
    
    x_1 = conv2d_with_Batch(x, nbr_kernels = 96, filter_size = (3,3), strides = (2,2) )
    x_2 = MaxPooling2D(pool_size=(3,3) , strides=(2,2), padding='same') (x)
    
    x = concatenate([x_1 , x_2], axis = 3)
    
    x_1 = conv2d_with_Batch(x, nbr_kernels = 64, filter_size = (1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels = 64, filter_size = (1,7) , padding ='same')
    x_1 = conv2d_with_Batch(x_1, nbr_kernels = 64, filter_size = (7,1), padding ='same')
    x_1 = conv2d_with_Batch(x_1, nbr_kernels = 96, filter_size = (3,3))
    
    x_2 = conv2d_with_Batch(x, nbr_kernels = 96, filter_size = (1,1))
    x_2 = conv2d_with_Batch(x_2, nbr_kernels = 96, filter_size = (3,3))
    
    x = concatenate([x_1 , x_2], axis = 3)
    
    x_1 = conv2d_with_Batch(x, nbr_kernels = 192, filter_size = (3,3) , strides=2)
    x_2 = MaxPooling2D(pool_size=(3,3) , strides=(2,2), padding='same') (x)
    
    x = concatenate([x_1 , x_2], axis = 3)
    
    return x


def reduction_A_Block(prev_layer):
    x_1 = conv2d_with_Batch(prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=224, filter_size=(3,3), padding='same')
    x_1 = conv2d_with_Batch(x_1, nbr_kernels=256, filter_size=(3,3), strides=(2,2))

    x_2 = conv2d_with_Batch(prev_layer, nbr_kernels=384, filter_size=(3,3), strides=(2,2))

    x_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(prev_layer)

    x = concatenate([x_1, x_2, x_3], axis=3)

    return x



def reduction_B_Block(prev_layer):
    x_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    
    x_2 = conv2d_with_Batch(prev_layer=prev_layer, nbr_kernels=192, filter_size=(1,1))
    x_2 = conv2d_with_Batch(prev_layer=x_2, nbr_kernels=192, filter_size=(3,3), strides=(1,1), padding='same')
    
    x_3 = conv2d_with_Batch(prev_layer=prev_layer, nbr_kernels=256, filter_size=(1,1))
    x_3 = conv2d_with_Batch(prev_layer=x_3, nbr_kernels=256, filter_size=(1,7), padding='same')
    x_3 = conv2d_with_Batch(prev_layer=x_3, nbr_kernels=320, filter_size=(7,1), padding='same')
    x_3 = conv2d_with_Batch(prev_layer=x_3, nbr_kernels=320, filter_size=(3,3), strides=(1,1), padding='same')
    
    x = concatenate([x_1, x_2, x_3], axis=3)
    return x



def InceptionBlock_A(prev_layer):
    
    x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = (1,1))
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = (3,3) , strides=(1,1), padding='same' )
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = (3,3) , strides=(1,1) , padding='same')
    
    x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = (1,1))
    x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 96, filter_size = (3,3) , padding='same')
    
    x_3 = AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same')(prev_layer)
    x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 96, filter_size = (1,1) , padding='same')
    
    x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 96, filter_size = (1,1))
    
    output = concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)

    return output


def InceptionBlock_B(prev_layer):
    
    x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 192, filter_size = (7,1) , padding='same')
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (1,7) , padding='same')
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (7,1) , padding='same')
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,7), padding='same')
    
    x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
    x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 192, filter_size = (1,7) , padding='same')
    x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 224, filter_size = (7,1), padding='same')
    x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 224, filter_size = (1,7), padding='same')
    x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (7,1), padding='same')
    
    x_3 = AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same')(prev_layer)
    x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 128, filter_size = (1,1))
    
    x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))

    output = concatenate([x_1 , x_2 ,x_3, x_4], axis = 3) 
    return output


def InceptionBlock_C(prev_layer):
    
    x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 448, filter_size = (3,1) , padding='same')
    x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 512, filter_size = (1,3) , padding='same')
    x_1_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,3), padding='same')
    x_1_2 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (3,1), padding='same')
    x_1 = concatenate([x_1_1 , x_1_2], axis = 3)
    
    x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))
    x_2_1 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (1,3), padding='same')
    x_2_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (3,1), padding='same')
    x_2 = concatenate([x_2_1 , x_2_2], axis = 3)
    
    x_3 = MaxPooling2D(pool_size=(3,3),strides = 1 , padding='same')(prev_layer)
    x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 256, filter_size = 3  , padding='same')
    
    x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 256, filter_size = (1,1))
    
    output = concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)
    
    return output

# In[189]:


model = InceptionV4()

# In[201]:


model.build((None, padded_mfcc_features.shape[1], padded_mfcc_features.shape[2], 1))

# In[202]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # ?댁쭊 遺꾨쪟??寃쎌슦 binary_crossentropy ?ъ슜
              metrics=['AUC', 'accuracy'])

# In[203]:


# model.summary()

# In[204]:


# print(f"Shape of x_train: {X_train.shape}")
# print(f"Shape of x_test: {X_valid.shape}")
# print(f"Sample values in x_train: {np.unique(X_train)}")
# print(f"Shape of model input: {model.input_shape}")

# In[205]:


print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_valid.shape}")
print(f"Sample values in y_train: {np.unique(y_train)}")
print(f"Shape of model output: {model.output_shape}")

# In[ ]:


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# In[ ]:


loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# In[97]:


y_pred = model.predict(X_valid)
y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_valid, axis=1)

# In[99]:


y_pred

# In[101]:


from sklearn.metrics import confusion_matrix, classification_report

# In[102]:


conf_matrix = confusion_matrix(y_valid, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)

# In[103]:


class_report = classification_report(y_valid, y_pred_classes, target_names=['fake', 'real'])
print('Classification Report:')
print(class_report)

# In[ ]:


model.save(str(OUTPUT_DIR / 'cnn_model_test.h5'))

# In[ ]:




# In[ ]:




