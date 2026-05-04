import pandas as pd
df = pd.read_csv('data/ton_iot.csv', nrows=5)
print(df.columns.tolist())
