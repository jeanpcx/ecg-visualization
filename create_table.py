import pandas as pd
import os

# Load the CSV file
file_path = '../results/signals.csv'
signals_df = pd.read_csv(file_path)

file_path = '../results/embeddings_50.csv'
e_df = pd.read_csv(file_path)

df = pd.DataFrame()
df['_id'] = signals_df['_id']

# Convertir las columnas de señales a un array
df['signal'] = signals_df.iloc[:, 1:].values.tolist()
df['embedding'] = e_df.iloc[:, 1:].values.tolist()

df.to_csv('../results/signalstable.csv', index=False)