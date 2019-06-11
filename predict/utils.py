import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np

def visualize_triangular(batch, idx, model_name, name):
    batch = batch[:, :16110]
    tri = np.zeros((180, 180))
    tri[np.triu_indices(180,1)] = batch[idx]
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./analysis/" + model_name + "/" + name)
    plt.show()
    plt.clf()

def visualize_matrix(batch, idx, model_name, name):
    tri = batch[idx].reshape((180,180))
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./analysis/" + model_name + "/" + name)
    plt.clf()

def get_consecutive_batch_VAE(start, batch_size, data):
    idx = np.arange(start, start + batch_size)
    batch = data[:,idx]
    return np.transpose(batch)

def get_random_batch(batch_size, data):
    idx = np.random.randint(data.shape[1], size=batch_size)
    batch = data[:,idx]
    return np.transpose(batch)

def get_random_batch(batch_size):
    idx = np.random.randint(data.shape[1], size=batch_size)
    batch = data[:16110,idx]
    labels = data[16110:,idx]
    return np.transpose(batch), np.transpose(labels)
