from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.spatial import distance
import json
import pandas as pd

app = Flask(__name__)

"Gola"

# Convertir el DataFrame a un formato que JSON pueda serializar
def serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serializable(value) for key, value in obj.items()}
    else:
        return obj
    
# # Cargar el modelo entrenado
# pca = joblib.load('pca_model.pkl')

# Load the data from the JSON file
with open('data/data_200.json', 'r') as f:
    print('Reading JSON...')
    data = json.load(f)
    print('Check!')

# Convert to DataFrame and drop the 'ecg' column for optimization
df_data = pd.DataFrame(data)
# min_count = metadata['tag'].value_counts().min()
# min_count = 100
# df_data = df_data.groupby('cluster').apply(lambda x: x.sample(min_count)).reset_index(drop = True)
# print(f'Total Records Balanced: {len(df_data)}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    copy = df_data.copy()
    copy = copy.drop(columns=['ecg'])

    # Seleccionar solo 100 registros de cada cluster
    df_limited = copy.groupby('cluster').head(200).reset_index(drop=True)

    # Convertir el DataFrame filtrado a una lista de diccionarios
    df_json = df_limited.to_dict(orient='records')

    return jsonify(df_json)

@app.route('/get_signal', methods=['POST'])
def get_nearby():
    content = request.json
    id = content['id']
    ids = content['ids']
    n = content['n'] + 1

    point = df_data[df_data['id'] == id]
    filtered_df = df_data[df_data['id'].isin(ids)]
    points = filtered_df[['x', 'y']].to_numpy()

    distances = distance.cdist([point[['x', 'y']].iloc[0]], points, 'euclidean')[0]
    closest_indices = np.argsort(distances)[:n]
    closest_data_indices = filtered_df.iloc[closest_indices].index

    signals = []
    signal_ids = []
    
    for idx in closest_data_indices:
        signal_ids.append(df_data.loc[idx, 'id'])
        signals.append(serializable(df_data.loc[idx, 'ecg']))

    return jsonify({'ids': signal_ids, 'signals': signals})


if __name__ == '__main__':
    app.run(debug=True)