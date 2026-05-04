import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import json

df = pd.read_csv('data/ton_iot.csv').dropna()

if "label" not in df.columns:
    for alt in ["Label", "type", "attack_type", "class"]:
        if alt in df.columns:
            df.rename(columns={alt: "label"}, inplace=True)
            break

y = df['label']
if y.dtype == object:
    y = y.astype('category').cat.codes

X = df.select_dtypes(include='number').drop(columns=['label', 'type', 'attack_type'], errors='ignore')
mi = mutual_info_classif(X, y, random_state=42)
mi_dict = dict(zip(X.columns, mi))
top_30 = sorted(mi_dict.items(), key=lambda item: item[1], reverse=True)[:30]
print(json.dumps([col for col, score in top_30]))
