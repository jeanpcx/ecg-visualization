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