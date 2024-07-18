import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
from model import CNN, ConvNormPool, Relu
from model import get_activation, load_model, loader_data, get_embedding, get_umap

form model import *

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, text, desc, create_engine, Sequence, Column, Float, String, Integer, JSON

# from dotenv import load_dotenv
# load_dotenv()

app = Flask(__name__)
DATABASE_URL = os.getenv('DATABASE_URL')

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
