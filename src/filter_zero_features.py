import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ?덉젣 ?곗씠?곕? ?ъ슜?섏뿬 ?곗씠?고봽?덉엫 ?앹꽦
data = pd.read_csv(DATA_DIR / 'features' / 'test_sr_44100.csv')

# ?뱀젙 ?댁뿉 0.0???ы븿???됱쓣 ?꾪꽣留곹븯?????
numeric_data = data.drop(columns=['id']).apply(pd.to_numeric, errors='coerce')

# 'id' ?댁쓣 ?쒖쇅???섎㉧吏 ?대뱾???⑹씠 0.0???됱쓣 ?꾪꽣留곹븯?????
condition = numeric_data.sum(axis=1) == 0.0
filtered_data = data[condition]

# ?꾪꽣留곷맂 寃곌낵瑜?異쒕젰
filtered_data = pd.DataFrame(filtered_data)
OUTPUT_DIR.mkdir(exist_ok=True)
filtered_data.to_csv(OUTPUT_DIR / 'value_is_zero.csv', index=False)
