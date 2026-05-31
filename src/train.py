import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from tqdm import tqdm
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

data = pd.read_csv(DATA_DIR / 'features' / 'train_sr_16000.csv')

data = data.drop(['id'],axis=1)

def process_complex(x):
    try:
        if isinstance(x, str) and '(' in x and ')' in x:
            # 臾몄옄?댁뿉??愿꾪샇瑜??쒓굅?섍퀬 蹂듭냼?섎줈 蹂?섑븳 ???ㅼ닔 遺遺?異붿텧
            x = complex(x.strip('()'))
            return x.real
        elif isinstance(x, complex):
            return x.real
        else:
            return float(x)
    except ValueError:
        return np.nan

for col in tqdm(data.columns):
    if data[col].dtype == 'object':
        data[col]=data[col].apply(process_complex).astype(float)

X=data.drop(['label'],axis=1).values
y=data['label'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2024)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_valid = scaler.fit_transform(X_valid)
print('finish_scaler')
print('train_ready')
clf = TabNetClassifier(
    n_steps=3,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=64,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr':29e-3},
    scheduler_fn=None,
    clip_value=2.0,
    momentum=0.3,
    scheduler_params={'step_size':50, 'gamma': 0.9},
    mask_type='sparsemax',
    device_name='cuda' if torch.cuda.is_available() else 'cpu',
    n_d=16,
    n_a=8,
    seed=100,
    lambda_sparse=0.3,
    epsilon=1e-15,
)

clf.fit(
    x_train, y_train,
    eval_set=[(x_train,y_train), (x_valid,y_valid)],
    eval_metric=['auc','logloss'],
    max_epochs=1000,
    patience=200,
    batch_size=32,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
)

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

y_pred = clf.predict(x_valid)
y_pred_proba = clf.predict_proba(x_valid)

accuracy = accuracy_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_pred_proba[:,1])
logloss = log_loss(y_valid,y_pred_proba)

print(f'accuracy: {accuracy}')
print(f'roc_auc : {roc_auc}')
print(f'logloss : {logloss}')
