import torch
import pandas as pd
import numpy as np,networkx as nx, scipy.sparse as sp
import graphgallery as gg
from graphgallery.datasets import Planetoid, NPZDataset
from collections import Counter
from os import path
from graphgallery import functional as gf
from scipy import sparse
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils import data as Data
import random
import os

class ReadData():
  def __init__(self, rpath="./dataset/datafromgg"):#you should download dataset in the rpath
    self.rpath = rpath

  def read_raw_data(self, dataname, verbose=0, **kwargs):
    data = gg.datasets.NPZDataset(dataname, root=self.rpath, verbose=verbose, **kwargs)
    return data

  def get_fixed_splited_data(self, dataname):
    data = Planetoid(dataname, root=self.rpath, verbose=False)
    return data

  def sample_per_class(self, num_per_class, data):
    sample_result =[]
    classes = np.zeros(3)
    graph = data.graph.to_undirected()
    random_tr_ind = list(np.hstack((data.split_nodes().train_nodes,data.split_nodes().val_nodes)))  
    random.shuffle(random_tr_ind)
    for i in random_tr_ind:
      if classes[graph.y[i]]<num_per_class:
          sample_result.append(i)
          classes[graph.y[i]]+=1
    sample_result = np.array(sample_result)
    return sample_result
  
  def ggp_embedding(self,num_per_class,data):
    graph = data.graph.to_undirected()
    A = graph.A
    if num_per_class==20:
      variance = 1.0 
      offset= 8.801428
    elif num_per_class==10:
      variance = 1.0 
      offset= 10.187971
    elif num_per_class==5:
      variance = 1.0 
      offset= 8.72369
    else:
      print("No valid sample!")
    degree = 3# The trained parameters for 140 training nodes
    X =graph.x
    sparse_P = graph.A.toarray()
    sparse_P = sparse_P/np.sum(sparse_P, 1, keepdims=True)
    base_K_mat = (variance * X@X.T + offset)**degree
    t1 = sparse_P@base_K_mat
    t2 = sparse_P@t1.T
    tag = graph.y
    t2 = t2 / t2.max(axis=0)
    return t2, tag
  def conv_subgraph(self, data):
    #Get the lagest subgragh
    graph = data.graph.to_undirected()
    t = graph.A.toarray()
    sg = list(nx.connected_component_subgraphs(nx.from_numpy_matrix(t)))
    vid_largest_graph = sg[np.argmax([nx.adjacency_matrix(g).shape[0] for g in sg])].nodes()
    adj = t[vid_largest_graph,:]; adj = adj[:, vid_largest_graph]
    adj_mat, features, labels = adj, graph.x[vid_largest_graph,:], graph.y[vid_largest_graph]
    # adj_csr = sp.csr_matrix(adj_mat)
    all_x = np.reshape(np.arange(labels.shape[0]), (-1,1))

    #graph convolution
    adj_nor = gg.functional.normalize_adj(adj_mat)
    h = adj_nor@adj_nor@features
    return h, adj_mat, features, all_x, labels


  def conv_graph(self, data,PE_file_path, pos_enc_init):
    graph = data.graph.to_undirected()
    A = graph.A
    A = gg.functional.normalize_adj(A)
    X = graph.x
    mat = A@A@X
    tag = graph.y


    if pos_enc_init=='rand_walk':
      if os.path.exists(PE_file_path):
          PE = np.load(PE_file_path)
      else:
          # compute the degree matrix
          d = np.zeros([len(X)])#     print(sample_result)
#     print(graph.y[sample_result])
          ind = np.where(A.todense() != 0)[0]
          for i in range(len(X)):
              d[i] = len(np.where(ind == i)[0])
          Dinv = sp.diags(d ** -1.0, dtype=float)
          RW = A * Dinv
          M = RW

          # Iterate
          nb_pos_enc = 32
          PE = [torch.from_numpy(M.diagonal()).float()]
          M_power = M
          print('save RWPE begin')
          for _ in range(nb_pos_enc - 1):
              M_power = M_power * M
              PE.append(torch.from_numpy(M_power.diagonal()).float())
          PE = torch.stack(PE, dim=-1)
          np.save(PE_file_path, PE)
          print('save RWPE done')
#     print(sample_result)
#     print(graph.y[sample_result])
          ## LSPE
          #  d = np.zeros([len(X)])
          #  ind = np.where(A.todense() != 0)[0]
          #  # val = np.where(A.todense() != 0)[1]
          #  for i in range(len(X)):
          #    d[i] = len(np.where(ind == i)[0])
          #  N = sp.diags(d ** -0.5, dtype=float)
          #  L = sp.eye(len(X)) - N * A * N
          #
          #  # Eigenvectors with numpy
          #  nb_pos_enc =  32
          #  EigVal, EigVec = np.linalg.eig(L.toarray())
          #  idx = EigVal.argsort()  # increasing order
          #  EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
          #  PE = torch.from_numpy(EigVec[:, 1:nb_pos_enc + 1]).float()
          #  np.save(PE_file_path, PE)
          #  print('save LSPE done')

    return mat, tag, np.array(PE)




  
  def rate_numpy2dataloader(self, tr_mat, ts_mat, tr_tag, ts_tag, batch_size=64):
    num_tr, _ = tr_mat.shape
    num_val = int(num_tr * 0.1)
    val_mat, val_tag = tr_mat[:num_val], tr_tag[:num_val]
    tr_mat, tr_tag = tr_mat[num_val:], tr_tag[num_val:]

    tr_mat, ts_mat, val_mat = torch.from_numpy(tr_mat), torch.from_numpy(ts_mat), torch.from_numpy(val_mat)
    tr_tag, ts_tag, val_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag), torch.from_numpy(val_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)
    val_dataset = Data.TensorDataset(val_mat, val_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    return train_data_loader, val_data_loader, test_data_loader

  def tr_ts_numpy2dataloader(self,tr_mat, ts_mat, tr_tag, ts_tag, batch_size=64):
    
    tr_mat, ts_mat = torch.from_numpy(tr_mat), torch.from_numpy(ts_mat)
    tr_tag, ts_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    return train_data_loader, test_data_loader
  def tr_ts_val_numpy2dataloader(self,tr_mat, ts_mat, val_mat, tr_tag, ts_tag, val_tag, batch_size=64):
    
    tr_mat, ts_mat, val_mat = torch.from_numpy(tr_mat), torch.from_numpy(ts_mat), torch.from_numpy(val_mat)
    tr_tag, ts_tag, val_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag), torch.from_numpy(val_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)
    val_dataset = Data.TensorDataset(val_mat, val_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    return train_data_loader, val_data_loader, test_data_loader

  def tr_ts_val_numpy2dataloader_LSPE(self, tr_mat, ts_mat, val_mat,tr_PE, ts_PE, val_PE, tr_tag, ts_tag, val_tag, batch_size=64):

    tr_mat, ts_mat, val_mat = torch.from_numpy(np.concatenate([tr_mat,tr_PE],axis=1)), torch.from_numpy(np.concatenate([ts_mat,ts_PE],axis=1)), torch.from_numpy(np.concatenate([val_mat,val_PE],axis=1))
    tr_tag, ts_tag, val_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag), torch.from_numpy(val_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)
    val_dataset = Data.TensorDataset(val_mat, val_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=False)
    return train_data_loader, val_data_loader, test_data_loader

  def tr_ts_numpy2dataloader(self,tr_mat, ts_mat, tr_tag, ts_tag, batch_size=64):
    
    tr_mat, ts_mat = torch.from_numpy(tr_mat), torch.from_numpy(ts_mat), torch.from_numpy(val_mat)
    tr_tag, ts_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag), torch.from_numpy(val_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    return train_data_loader, test_data_loader

  def sample_numpy2dataloader(self, num_per_class, data_fixed, mat,tag, batch_size=64):
    
    graph = data_fixed.graph.to_undirected()
    # print("tag:", graph.y[:140])
    # num_tr = int(num_tr_val * 70/640)
    tr_ind = self.sample_per_class(num_per_class,data_fixed)
    print("num_training nodes?", len(tr_ind))
    
    val_ind = data_fixed.split_nodes().val_nodes
    ts_ind = data_fixed.split_nodes().test_nodes
    
    tr_mat = mat[tr_ind]
    tr_tag = tag[tr_ind]
    val_mat = mat[val_ind]
    val_tag = tag[val_ind]
    ts_mat = mat[ts_ind]
    ts_tag = tag[ts_ind]
    tr_mat, ts_mat, val_mat = torch.from_numpy(tr_mat), torch.from_numpy(ts_mat), torch.from_numpy(val_mat)
    tr_tag, ts_tag, val_tag = torch.from_numpy(tr_tag), torch.from_numpy(ts_tag), torch.from_numpy(val_tag)

    train_dataset = Data.TensorDataset(tr_mat, tr_tag)
    test_dataset = Data.TensorDataset(ts_mat, ts_tag)
    val_dataset = Data.TensorDataset(val_mat, val_tag)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
      shuffle=True,
      drop_last=False)
    return train_data_loader, val_data_loader, test_data_loader

  def normalize_col(self,mat):
    data_norm = np.zeros(mat.shape)
    for x in range(mat.shape[-1]):
      data_norm[:,x]=mat[:,x]/np.max(np.abs(mat[:,x]))
    #   print("th.max(th.abs(dataset[:,x])):",th.max(th.abs(dataset[:,x])))
    return data_norm

  def get_random_splited_data(self, mat, tag, test_size=0.2, random_state=2020):
    split_param = {'test_size': test_size, 'random_state': random_state, 'stratify': tag}
    tr_mat, ts_mat, tr_tag, ts_tag = train_test_split(mat, tag, **split_param)
    return tr_mat, ts_mat, tr_tag, ts_tag

  def get_random_ind_tensor(self, arr, tag, test_size=0.2, random_state=2020):
    split_param = {'test_size': test_size, 'random_state': random_state, 'stratify': tag}
    tr_ind, ts_ind, _, _ = train_test_split(arr, tag, **split_param)
    val_size = int(len(tr_ind) * 0.2)
    val_ind = tr_ind[:val_size]
    tr_ind = tr_ind[val_size:]
    tr_ind, val_ind, ts_ind = torch.LongTensor(tr_ind),torch.LongTensor(val_ind),torch.LongTensor(ts_ind)
    return tr_ind, val_ind, ts_ind




  


