import os
from flask import Flask, request, jsonify, render_template, redirect, url_for

from bson import json_util
import numpy as np
from scipy.spatial import distance
import json
import pandas as pd
import torch

from model import *

from flask_socketio import SocketIO, emit

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate
from sqlalchemy import func, text, desc
from sqlalchemy import create_engine, Sequence, Column, Float, String, Integer, JSON
import os


app = Flask(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgres://uesm16qpp36267:p71df9743039f4f983238a5f5be1678ed16374aa713a8cf8beafe9ba8c4bdc46e@cb5ajfjosdpmil.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d6629s16654mbt')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

embeddings_dict = {}

def load_embeddings():
    """Carga los embeddings y los almacena en una variable global."""
    global embeddings_dict
    with app.app_context():
        all_embeddings = db.session.query(Signal._id, Signal.embedding).all()
        embeddings_dict = {e._id: np.array(e.embedding) for e in all_embeddings}

# Define the model for META table
class Meta(db.Model):
    __tablename__ = 'meta'
    _id = Column(Integer, Sequence('meta_id_seq'), primary_key=True, autoincrement=True)
    age = Column(Integer, nullable=True)
    sex = Column(String(10), nullable=True)
    pred = Column(Integer, nullable=True)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    cluster = Column(Integer, nullable=True)
    label = Column(String(10), nullable=True)
    near_1 = Column(Integer, nullable=True)
    near_2 = Column(Integer, nullable=True)
    selected_1 = Column(Integer, nullable=True)
    selected_2 = Column(Integer, nullable=True)

class Signal(db.Model):
    __tablename__ = 'signals'
    _id = Column(Integer, Sequence('signal_id_seq'), primary_key=True, autoincrement=True)
    signal = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    # Get order per cluster
    subquery = db.session.query(
        Meta,
        func.row_number().over(
            partition_by=Meta.cluster,
            order_by=Meta._id
        ).label('row_num')
    ).subquery()#.filter(Meta.cluster.isnot(None)).subquery()

    # Get n records
    query = db.session.query(subquery).filter(subquery.c.row_num <= 50)
    result = query.all()

    json = [dict(row._mapping) for row in result]
    return jsonify(json)
    # result = db.session.query(Meta).all()
    # json_result = [dict(row.__dict__) for row in result]
    # for item in json_result:
    #     item.pop('_sa_instance_state', None)  # Eliminar el estado interno de SQLAlchemy si está presente
    # return jsonify(json_result)

@app.route('/get_signal', methods=['POST'])
def get_nearby():
    content = request.json
    id = content['id']
    ids = content['ids']
    n = 2
    global embeddings_dict

    # Obtener el embedding objetivo
    target_embedding = embeddings_dict[id]
    
    # Filtrar los embeddings necesarios
    embeddings = {key: embeddings_dict[key] for key in ids if key != id}
    
    # Convertir los valores del diccionario a un array de NumPy para operaciones vectorizadas
    embedding_ids = np.array(list(embeddings.keys()))
    embedding_values = np.array(list(embeddings.values()))
    
    # Calcular las distancias usando operaciones vectorizadas
    dists = np.linalg.norm(embedding_values - target_embedding, axis=1)
    
    # Obtener los índices de los n embeddings más cercanos
    closest_indices = np.argsort(dists)[:n]
    closest_ids = embedding_ids[closest_indices].tolist()

    # Incluir el objetivo mismo como el primer elemento
    closest_ids.insert(0, id)

    # Obtener las señales correspondientes
    filter = db.session.query(Signal._id, Signal.signal).filter(Signal._id.in_(closest_ids)).all()
   
    # Convertir las señales a una lista
    signals_dict = {signal._id: signal.signal for signal in filter}
    signals_list = [signals_dict[_id] for _id in closest_ids]

    return jsonify({"ids": closest_ids, "signals": signals_list})


@app.route('/get_look', methods=['POST'])
def get_look():
    content = request.json
    id = content['id']

    query = db.session.query(Signal.signal).filter(Signal._id == id).first()
    signal = query.signal

    return jsonify(signal)

@app.route('/upload', methods=['POST'])
def upload_file():
    sex = request.form['sex']
    age = request.form['age']
    file = request.files['csv']
    if not file:
        return "No file"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    signal = df.iloc[:, 0].tolist()
    x = loader_data(df)
    embedding, prediction = get_embedding(x)
    umap = get_umap(embedding)

    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    prediction = int(prediction)
    umap_x = float(umap.iloc[0][0])
    umap_y = float(umap.iloc[0][1])

    new_signal = Signal(signal=signal, embedding=embedding)
    new_data = Meta(age=age, sex=sex, pred=prediction, x=umap_x, y=umap_y)
    
    db.session.add(new_signal)
    db.session.add(new_data)
    db.session.commit()

    load_embeddings()

    print('Create: ',new_signal._id)
    print('Create: ',new_data._id)



    # For demonstration, just return the DataFrame as HTML
    # return umap.to_html()
    return redirect(url_for('index'))

















# @app.route('/get_signal', methods=['POST'])
# def get_nearby():
#     content = request.json
#     id = content['id']
#     ids = content['ids']
#     n = 2
    
#     # Fetch embeddings for the given IDs
#     filter = db.session.query(Signal._id, Signal.embedding).filter(Signal._id.in_(ids)).all()
#     embeddings = {e._id: np.array(e.embedding) for e in filter}

#     # Get the target embedding and remove it from the dictionary
#     target_embedding = embeddings.pop(id)
    
#     # Convert dictionary values to a numpy array for vectorized operations
#     embedding_ids = np.array(list(embeddings.keys()))
#     embedding_values = np.array(list(embeddings.values()))
    
#     # Calculate distances using vectorized operations
#     dists = np.linalg.norm(embedding_values - target_embedding, axis=1)
    
#     # Get the indices of the n closest embeddings
#     closest_indices = np.argsort(dists)[:n]
#     closest_ids = embedding_ids[closest_indices].tolist()

#     # Include the target itself as the first element
#     closest_ids.insert(0, id)

#     # Get Signals
#     filter = db.session.query(Signal._id, Signal.signal).filter(Signal._id.in_(closest_ids)).all()
   
#     # Convert signals to a list
#     signals_dict = {signal._id: signal.signal for signal in filter}
#     signals_list = [signals_dict[_id] for _id in closest_ids]

#     return jsonify({"ids": closest_ids, "signals": signals_list})

# @app.route('/get_signal', methods=['POST'])
# def get_nearby():
#     content = request.json
#     id = content['id']
#     ids = content['ids']
#     n = 2
    
#     # Fetch embeddings for the given IDs
#     filter = db.session.query(Signal).filter(Signal._id.in_(ids)).all()
#     # Convert embeddings to numpy arrays
#     embeddings = {e._id: e.embedding for e in filter}    
#     # Get the target embedding and remove it from the dictionary
#     target_embedding = embeddings.pop(id)
    
#     # Calculate distances
#     distances = {}
#     for _id, embedding in embeddings.items():
#         dist = distance.euclidean(target_embedding, embedding)
#         distances[_id] = dist
    
#     # Find the n closest embeddings
#     closest_ids = sorted(distances, key=distances.get)[:n]
#     closest_distances = {cid: distances[cid] for cid in closest_ids}

#     # Include the target itself as the first element
#     closest_ids.insert(0, id)
#     print(closest_ids)

#     # Get Signals
#     filter = db.session.query(Signal.signal).filter(Signal._id.in_(closest_ids)).all()
   
#     # Convert signals to a list
#     signals_list = [signal.signal for signal in filter]

#     return jsonify({"ids": closest_ids, "signals": signals_list})




    
# # Cargar el modelo entrenado
# pca = joblib.load('pca_model.pkl')





# @app.route('/get_data', methods=['GET'])
# def get_data():
#     copy = df_data.copy()
#     copy = copy.drop(columns=['ecg'])

#     # Seleccionar solo 100 registros de cada cluster
#     df_limited = copy.groupby('cluster').head(100).reset_index(drop=True)

#     # Convertir el DataFrame filtrado a una lista de diccionarios
#     df_json = df_limited.to_dict(orient='records')

#     return jsonify(df_json)

# @app.route('/get_signal', methods=['POST'])
# def get_nearby():
#     content = request.json
#     id = content['id']
#     ids = content['ids']
#     n = content['n'] + 1

#     point = df_data[df_data['id'] == id]
#     filtered_df = df_data[df_data['id'].isin(ids)]
#     points = filtered_df[['x', 'y']].to_numpy()

#     distances = distance.cdist([point[['x', 'y']].iloc[0]], points, 'euclidean')[0]
#     closest_indices = np.argsort(distances)[:n]
#     closest_data_indices = filtered_df.iloc[closest_indices].index

#     signals = []
#     signal_ids = []
    
#     for idx in closest_data_indices:
#         signal_ids.append(df_data.loc[idx, 'id'])
#         signals.append(serializable(df_data.loc[idx, 'ecg']))

#     return jsonify({'ids': signal_ids, 'signals': signals})


if __name__ == '__main__':
    load_embeddings()

    # db.session.execute('SELECT setval(\'meta_id_seq\', (SELECT MAX(_id) FROM meta) + 1);')
    # db.session.execute('SELECT setval(\'signal_id_seq\', (SELECT MAX(_id) FROM signals) + 1);')
    # db.session.commit()
    app.run(debug=True)