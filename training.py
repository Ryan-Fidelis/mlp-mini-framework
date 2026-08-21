import mini_framework as mf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('cumulative.csv', sep=',')
columns_used = ['koi_duration', 'koi_depth', 'koi_steff', 'koi_model_snr', 'koi_score', 'koi_prad', 'koi_period']
df.dropna(subset=columns_used, inplace=True)
df.reset_index(drop=True, inplace=True)

def z(x):
    media = np.mean(x)
    dp = np.std(x)
    return (x - media) / dp

z_duration = z(df['koi_duration'])
z_depth = z(df['koi_depth'])
z_steff = z(df['koi_steff'])
z_model_snr = z(df['koi_model_snr'])
z_prad = z(df['koi_prad'])
z_period = z(df['koi_period'])

x_features = np.column_stack((z_duration, z_depth, z_steff, z_model_snr, z_prad, z_period))

y_raw = df["koi_score"].values
y_one_hot = np.zeros((len(y_raw), 2))

for i in range(len(y_raw)):
    if y_raw[i] >= 0.5:
        y_one_hot[i] = [1, 0] 
    else:
        y_one_hot[i] = [0, 1] 

x_np = np.expand_dims(x_features, axis=0) 
y_np = np.expand_dims(y_one_hot, axis=0)  

x_train, x_test, y_train, y_test = train_test_split(x_features, y_one_hot, test_size=0.20, random_state=42)

x_train_np = np.expand_dims(x_train, axis=0) 
y_train_np = np.expand_dims(y_train, axis=0)  

x_test_np = np.expand_dims(x_test, axis=0)
y_test_np = np.expand_dims(y_test, axis=0)

train = mf.Simple_Training_Loop(n_hidden_layers=2, n_inputs=6, epochs=500, eta=0.01)
train.hidden_layer_neuron_set(0, 20)
train.hidden_layer_neuron_set(1, 12)

print("Training...")
train.loop(x_train_np, y_train_np)

print("Testing...")
train.test(x_test_np, y_test_np)
