import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
# from model import *
# from model import CNN
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, text, desc, create_engine, Sequence, Column, Float, String, Integer, JSON

app = Flask(__name__)
DATABASE_URL = os.getenv('DATABASE_URL')
# DATABASE_URL = 'postgresql://postgres:CXJNCPIJGrjaxxKnyLrIVbwOzBPazpQF@roundhouse.proxy.rlwy.net:13721/railway'

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
embeddings_dict = {}

import torch
import torch.nn as nn
import torch.nn.functional as F
# import pandas as pd
import numpy as np
import joblib
# import umap

# Cargar los objetos StandardScaler y UMAP
print('Loading tools..')
scaler = joblib.load('data/scaler.pkl')
reducer = joblib.load('data/reducer.pkl')
print('Loaded! tools..')


class Relu(nn.Module):
  def forward(self, x):
    return torch.relu(x)
  
class ConvNormPool(nn.Module):
  """Conv Skip-connection module"""
  def __init__(
    self,
    input_size,
    hidden_size,
    kernel_size,
    norm_type='batch',
    dropout_rate=0.1
  ):
    super().__init__()

    self.kernel_size = kernel_size
    self.conv_1 = nn.Conv1d(
      in_channels=input_size,
      out_channels=hidden_size,
      kernel_size=kernel_size,
      padding=(kernel_size -1) // 2
    )
    self.conv_2 = nn.Conv1d(
      in_channels=hidden_size,
      out_channels=hidden_size,
      kernel_size=kernel_size,
      padding=(kernel_size -1) // 2
    )
    self.conv_3 = nn.Conv1d(
      in_channels=hidden_size,
      out_channels=hidden_size,
      kernel_size=kernel_size,
      padding=(kernel_size -1) // 2
    )
    self.relu_1 = nn.ReLU()
    self.relu_2 = nn.ReLU()
    self.relu_3 = nn.ReLU()

    if norm_type == 'group':
      self.normalization_1 = nn.GroupNorm(
        num_groups=8,
        num_channels=hidden_size
      )
      self.normalization_2 = nn.GroupNorm(
        num_groups=8,
        num_channels=hidden_size
      )
      self.normalization_3 = nn.GroupNorm(
        num_groups=8,
        num_channels=hidden_size
      )
    else:
      self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
      self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
      self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)

    self.pool = nn.MaxPool1d(kernel_size=2)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward(self, input):
    conv1 = self.conv_1(input)
    x = self.normalization_1(conv1)
    x = self.relu_1(x)

    x = self.conv_2(x)
    x = self.normalization_2(x)
    x = self.relu_2(x)

    conv3 = self.conv_3(x)
    x = self.normalization_3(conv1+conv3)
    x = self.relu_3(x)

    x = self.pool(x)
    x = self.dropout(x)
    return x

class CNN(nn.Module):
  def __init__(
    self,
    input_size = 12,
    hid_size = 128,
    kernel_size = 3,
    num_classes = 5,
    dropout_rate=0.1,
  ):

    super().__init__()

    self.conv1 = ConvNormPool(
      input_size=input_size,
      hidden_size=hid_size,
      kernel_size=kernel_size,
    )
    self.conv2 = ConvNormPool(
      input_size=hid_size,
      hidden_size=hid_size//2,
      kernel_size=kernel_size,
    )
    self.conv3 = ConvNormPool(
      input_size=hid_size//2,
      hidden_size=hid_size//4,
      kernel_size=kernel_size,
    )
    self.avgpool = nn.AdaptiveAvgPool1d((1))
    self.dropout = nn.Dropout( p = dropout_rate)
    self.fc = nn.Linear(in_features=hid_size//4, out_features=num_classes)

  def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.avgpool(x)
    x = x.view(-1, x.size(1) * x.size(2))
    x = self.dropout(x)
    x = self.fc(x)
    #x = torch.sigmoid(self.fc(x))
    # x = F.softmax(self.fc(x), dim=1) #Cross Entropy lo hace automáticamente.
    return x

# model = CNN(num_classes = num_classes, hid_size = 128)

activation = {}
def get_activation(name):
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def load_model():
  model = CNN(num_classes = 5, hid_size = 128)
  # model.load_state_dict(torch.load('model_50.pth'))
  # model.load_state_dict()
  
  model = torch.load('data/model_50.pth', map_location='cpu')
  model.avgpool.register_forward_hook(get_activation('avgpool')) 
  model.eval()
  return model

def loader_data(df):
  df = df.astype(np.float64)
  df = df.to_numpy()
  df = scaler.transform(df)
  signal = df[:, 0].tolist()

  x = df.T.reshape(1, 12, 1000)
  x = torch.tensor(x).float()
  return x, signal

def get_embedding(x):
  model = load_model()
  with torch.no_grad():
    yhat = model(x)
    prediction = (torch.argmax(yhat, dim=1)).cpu().numpy()[0]
    embeddings = activation['avgpool'].cpu().numpy()
  result = np.squeeze(embeddings)

  return result, prediction

def get_umap(embedding):
  # Transformar usando el reducer cargado
  point = reducer.transform([embedding])
  # Verificar el punto UMAP
  print("Punto UMAP:", point)
  
  return point


# # Prueba de la función con diferentes embeddings
# embedding1 = [0.1] * 32  # Reemplaza con tus valores reales
# embedding2 = [3] * 32 # Reemplaza con tus valores reales

# point1 = get_umap(embedding1)
# point2 = get_umap(embedding2)

# print("Point 1:", point1)
# print("Point 2:", point2)

# df = pd.read_csv('../results/21837_lr.csv')
# df = df.astype(np.float64)
# df = df.to_numpy()
# x = df.T.reshape(1, 12, 1000)
# x = torch.tensor(x).float()

# e = get_embedding(x)

# print(e)

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

def get_nearby(id, filter_ids, n = 2):
    # Use the embeddings pre-load
    global embeddings_dict

    # Get the target embedding
    target = embeddings_dict[id]

    # Keep only filter points
    embeddings = {key: embeddings_dict[key] for key in filter_ids if key != id}

    # Find 2 nearby embeddings
    embedding_ids = np.array(list(embeddings.keys())) # Vectorized operations
    embedding_values = np.array(list(embeddings.values()))
    dists = np.linalg.norm(embedding_values - target, axis = 1)
    closest_indices = np.argpartition(dists, 2)[:n]
    closest_ids = embedding_ids[closest_indices].tolist()

    return closest_ids

def get_signal(ids):
    if not ids:
        return jsonify({'error': 'List of IDs is empty'})
    
    if isinstance(ids, (int, float)):
        # Get single Signal
        query = db.session.query(Signal.signal).filter(Signal._id == ids).first()
        if query is None:
            return jsonify({'error': 'Record not found'})
        else:
            result = query.signal
    else:
        # Get multiple Signals
        query = db.session.query(Signal._id, Signal.signal).filter(Signal._id.in_(ids)).all()
        if query is None:
            return jsonify({'error': 'Records not found'})
    
        signals_dict = {signal._id: signal.signal for signal in query}
        result = [signals_dict[_id] for _id in ids]

    return result

@app.route('/get_signals', methods=['POST'])
def get_signals():
    content = request.json
    id = content['id']
    filter_ids = content['filterIds']

    # Found get nearby ids
    nearby_ids = get_nearby(id, filter_ids)
    nearby_ids.insert(0, id)

    # Get target info
    meta_target = db.session.query(Meta).filter(Meta._id == id).first()
    if not meta_target:
        return jsonify({'message': 'Record not found'}), 404

    selected_ids = None
    nearby1 = meta_target.selected_1
    nearby2 = meta_target.selected_2

    if nearby1 and nearby2:
        # User select two points
        selected_ids = [nearby1, nearby2]
    elif nearby1:
        # User select only nearby1
        selected_ids = [nearby1, nearby_ids[2]]
    elif nearby2:
        # User select only nearby2
        selected_ids = [nearby_ids[1], nearby2]

    # If user select any point, so graph
    if selected_ids:
        selected_ids.insert(0, id)
        signals = get_signal(selected_ids)
    else:
        signals = get_signal(nearby_ids)
    
    return jsonify({"nearbyIDs": nearby_ids, "selectedIds": selected_ids, "signal": signals})

@app.route('/examine_signal', methods=['POST'])
def examine_signal():
    content = request.json
    id = content['id']
    result = get_signal(id)

    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload_file():
    sex = request.form['sex']
    age = request.form['age']
    file = request.files['csv']

    # Handle error when not file is uploaded
    if not file:
        return jsonify({'message': 'No file found to upload'}), 404

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    x, signal = loader_data(df) # Transform df to send to model, algo get the signal
    embedding, prediction = get_embedding(x) # Evaluate in model
    umap = get_umap(embedding)

    # Transform embedding into listo, to upload as JSON
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    # Create documents
    new_signal = Signal(signal = signal, embedding = embedding)
    new_data = Meta(age = age, sex = sex, pred = int(prediction), x = float(umap[0][0]), y = float(umap[0][1]))
    
    db.session.add(new_signal)
    db.session.add(new_data)
    db.session.commit() # Send and update database

    # Reload embeddings with updated
    load_embeddings()
    return jsonify({'message': 'Data uploaded successfully!'}), 200

@app.route('/update_nearby', methods=['POST'])
def update_nearby():
    content = request.json
    id = content['id']
    i = content.get('selected')
    id_nearby = content.get('selected_id')

    # Find the record
    meta_record = db.session.query(Meta).filter(Meta._id == id).first()
    if meta_record:
        if id_nearby is not None:
            if (i == "1"):
                # Save in nearby1
                meta_record.selected_1 = id_nearby
            else:
                # Save in nearby2
                meta_record.selected_2 = id_nearby

        db.session.commit()
        return jsonify({'message': 'Record updated successfully!'}), 200
    else:
        return jsonify({'message': 'Record not found'}), 404

# PRELoad Embeddings
load_embeddings()
if __name__ == '__main__':
    app.run(debug=True)
