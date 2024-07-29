import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

# Define the map classes
class_tag = {
    0: 'NORM',
    1: 'MI',
    2: 'STTC',
    3: 'CD',
    4: 'HYP'
}

def apply_standardizer(X, ss):
  """
  Apply a standard scaler to each element in a list of arrays.
  Returns:    np.array: Array containing standardized versions of each input array in X.
  """
  X_tmp = []
  for x in X:
    x_shape = x.shape
    X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
  X_tmp = np.array(X_tmp)

  return X_tmp

# Load objects StandardScaler (For normalize input) and UMAP
print('Loading tools..')
scaler = joblib.load('data/scaler.pkl')
reducer = joblib.load('data/reducer.pkl')
print('Loaded! tools..')

# Define model
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
    x = F.softmax(x, dim=1) #Cross Entropy lo hace automáticamente.
    return x

# Define Hook to get embeddings (latent space)
activation = {}
def get_activation(name):
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

# Function to load model, define hook and set evaluation mode
def load_model():
  model = CNN(num_classes = 5, hid_size = 128)
  model.load_state_dict(torch.load('data/model_dict.pth', map_location=torch.device('cpu')))
  model.avgpool.register_forward_hook(get_activation('avgpool')) 
  model.eval()
  return model

# Fcuntion to read csv, convert to npy float, standarize and get signal
def loader_data(df):
  df = df.astype(np.float64)
  # df = df.to_numpy()
  df = apply_standardizer(df.to_numpy(), scaler)
  # df = scaler.transform(df)
  signal = df[:, 0].tolist()
  x = df.T.reshape(1, 12, 1000)
  x = torch.tensor(x).float()

  return x, signal

# Function to get embedding, prediction and label: load model, pass model
def get_embedding(x):
  model = load_model()
  with torch.no_grad():
    yhat = model(x)
    prediction = (torch.argmax(yhat, dim=1)).cpu().numpy()[0]
    embeddings = activation['avgpool'].cpu().numpy()
    label = class_tag.get(prediction, 0)
  result = np.squeeze(embeddings)

  return result, prediction, label

# Get 2d projection with the reducer pre train
def get_umap(embedding):
  point = reducer.transform([embedding])
  return point