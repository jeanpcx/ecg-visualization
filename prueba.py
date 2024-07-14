import joblib
import numpy as np
import umap

# Cargar scaler y reducer desde los archivos
scaler = joblib.load('scaler.pkl')
reducer = joblib.load('umap_reducer.pkl')

# Embedding específico que se transformará
embedding_1 = np.array([3.9728004e-01, 6.5472919e+02, 7.1145081e+02, 8.6832965e+02, 
               4.6252689e-01, 0.0000000e+00, 7.2489392e+02, 5.7611450e+02, 
               5.1367674e-02, 0.0000000e+00, 5.7852307e+02, 7.7807810e-03, 
               3.5000140e+02, 9.0980142e-01, 1.2114165e+00, 6.9101334e+00, 
               2.9063303e+02, 7.8520477e+02, 3.0262068e+01, 0.0000000e+00, 
               3.7417139e+02, 9.5701700e-01, 4.5661414e+02, 2.2108996e+00, 
               6.8789148e+02, 2.7777234e+02, 0.0000000e+00, 0.0000000e+00, 
               0.0000000e+00, 1.0109432e-01, 2.5273657e+01, 1.6352271e+00]).reshape(1, -1)

# Aplicar la transformación
embedding_scaled = scaler.transform(embedding_1)
embedding_reduced = reducer.transform(embedding_scaled)

print(embedding_reduced)
