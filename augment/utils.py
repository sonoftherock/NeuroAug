import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np

def normalize_adj(adj):
    # Account for negative connectivity values
    adj = np.abs(adj)
    rowsum = np.array(adj.sum(2))
    adj_norm = np.zeros(adj.shape, np.float32)
    for i in range(rowsum.shape[0]):
        degree_mat_inv_sqrt = np.diag(np.sign(rowsum[i])*np.power(np.abs(rowsum[i]), -0.5).flatten())
        degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.
        adj_norm[i] = adj[i].dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_norm

def construct_feed_dict_VGAE(adj_norm, adj_orig, features, dropout, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict

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

def visualize_latent_space_VAE(z_mean, labels, model_name):
    plt.figure(figsize=(28,28))
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10,i+1+j*10)
            plt.plot(z_mean[labels==0,i], z_mean[labels==0,j], 'o', label='Control', alpha=0.5)
            plt.plot(z_mean[labels==1,i], z_mean[labels==1,j], 'o', label='Schizophrenic', alpha=0.5)
    plt.legend()
    plt.savefig('./analysis/' + model_name + "/" + 'latent_space.png')
    plt.show()
    plt.close()

def visualize_latent_space_VGAE(z_mean, labels, model_name):
    plt.figure(figsize=(14,14))
    # Check first 5 nodes
    for k in range(5):
        for i in range(5):
            for j in range(5):
                plt.subplot(5,5,i+1+j*5)
                plt.plot(z_mean[labels==0,k,i], z_mean[labels==0,k,j], 'o', label='Control', alpha=0.5)
                plt.plot(z_mean[labels==1,k,i], z_mean[labels==1,k,j], 'o', label='Schizophrenic', alpha=0.5)
        plt.legend()
        plt.savefig('./analysis/' + model_name + '/latent_space_%i.png' %(k))
        plt.close()

def get_random_batch_VGAE(batch_size, adj, adj_norm):
    num_nodes = adj.shape[1]
    adj_idx = np.random.randint(adj.shape[0], size=batch_size)
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch, adj_idx

def get_random_batch_VAE(batch_size, data):
    idx = np.random.randint(data.shape[1], size=batch_size)
    batch = data[:,idx]
    return np.transpose(batch)

def get_consecutive_batch_VAE(start, batch_size, data):
    idx = np.arange(start, start + batch_size)
    batch = data[:,idx]
    return np.transpose(batch)

def get_consecutive_batch_VGAE(start, batch_size, adj, adj_norm):
    adj_idx = np.arange(start, start + batch_size)
    num_nodes = adj.shape[1]
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch, adj_idx
