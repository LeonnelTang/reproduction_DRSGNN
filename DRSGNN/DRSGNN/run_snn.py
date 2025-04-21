# coding=gbk
import datareader, sharedutils, os
import numpy as np
import pandas as pd
from model_lif_fc import model_lif_fc
from model_lif_fc_with_val import model_lif_fc_with_val


def snn_run(dataname,if_with_val, **new_conf):
  conf, cnf = sharedutils.read_config(), {}
  cnf.update(conf['shared_conf'])
  cnf.update(conf['snn'][dataname])
  # may be some new params
  cnf.update(new_conf)
  cnf['log_dir'] = conf['snn']['log_dir']
  if cnf['v_reset'] == -100: cnf['v_reset'] = None
  print("batch_size:", cnf["batch_size"])
  # rd = datareader.ReadData("/dataset/zh/dataset/datafromgg")
  rd = datareader.ReadData("./dataset/datafromgg")

  data_fixed = rd.read_raw_data(dataname)
  data =data_fixed
  PE_files_path = os.path.join('./PE',dataname)
  if not os.path.exists(PE_files_path):
      os.makedirs(PE_files_path)
  PE_file_path = PE_files_path +'/RWPE.npy'
  mat, tag, PE = rd.conv_graph(data_fixed,PE_file_path,pos_enc_init='rand_walk')
  PE_dim = PE.shape[1]
  try:
      PE.shape[0]==mat.shape[0]
  except:
      print("load false PE")

  tr_ind, val_ind, ts_ind = data.split_nodes().train_nodes, \
  data.split_nodes().val_nodes, data.split_nodes().test_nodes


  print("train, valiadation,test's shape:", len(tr_ind), len(val_ind), len(ts_ind))
  tr_val_ind = np.hstack((tr_ind,val_ind))


  tr_val_mat = mat[tr_val_ind]
  tr_val_tag = tag[tr_val_ind]
  tr_mat = mat[tr_ind]
  tr_PE = PE[tr_ind]
  tr_tag = tag[tr_ind]
  val_mat = mat[val_ind]
  val_PE = PE[val_ind]
  val_tag = tag[val_ind]
  ts_mat = mat[ts_ind]
  ts_PE = PE[ts_ind]
  ts_tag = tag[ts_ind]
  k = pd.DataFrame(mat)
  u = k.describe()


  self_sample = False

  if self_sample==True:
    train_data_loader, val_data_loader, test_data_loader = rd.sample_numpy2dataloader(20,data_fixed, mat, tag, batch_size=cnf["batch_size"])
  else:
    train_data_loader, val_data_loader, test_data_loader = rd.tr_ts_val_numpy2dataloader_LSPE(tr_mat, ts_mat, val_mat,tr_PE, ts_PE, val_PE, tr_tag,
                                                                                     ts_tag, val_tag,
                                                                                     batch_size=cnf["batch_size"])




  print("train, valiadation,test's batch num:", len(train_data_loader), len(val_data_loader), len(test_data_loader))

  n_nodes, n_feat, n_flat = mat.shape[0], (mat.shape[1]+PE.shape[1]), 1
  print("dataset: %s, num_node_classes: %d" % (dataname, data.graph.num_classes))
  if if_with_val=="no":
    print('mode: with no valcode ')
    ret = model_lif_fc(device=cnf["device"], dataset_dir=cnf["dataset_dir"],
                       dataname=dataname, batch_size=cnf["batch_size"],
                       learning_rate=cnf["learning_rate"], T=cnf["T"], tau=cnf["tau"],
                       v_reset=cnf["v_reset"], v_threshold=cnf["v_threshold"],
                       train_epoch=cnf["train_epoch"], log_dir=cnf["log_dir"], n_labels=data.graph.num_classes,
                       n_dim0=n_nodes, n_dim1=n_flat, n_dim2=n_feat, train_data_loader=train_data_loader,
                       val_data_loader=val_data_loader, test_data_loader=test_data_loader, PE_dim=PE_dim)
  elif if_with_val=="yes":
    print('mode: with valcode ')
    ret = model_lif_fc_with_val(device=cnf["device"], dataset_dir=cnf["dataset_dir"],
                       dataname=dataname, batch_size=cnf["batch_size"],
                       learning_rate=cnf["learning_rate"], T=cnf["T"], tau=cnf["tau"],
                       v_reset=cnf["v_reset"], v_threshold=cnf["v_threshold"],
                       train_epoch=cnf["train_epoch"], log_dir=cnf["log_dir"], n_labels=data.graph.num_classes,
                       n_dim0=n_nodes, n_dim1=n_flat, n_dim2=n_feat, train_data_loader=train_data_loader,
                       val_data_loader=val_data_loader, test_data_loader=test_data_loader, PE_dim=PE_dim)


  return ret


def model_startup(dataname, runs,if_with_val, **new_conf):
  #borrowed from https://github.com/ZulunZhu/SpikingGCN
  scores = []

  submits = []

  conc = False
  if conc:
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(10)
    for run in range(runs):
      obj = pool.submit(snn_run, dataname, **new_conf)
      submits.append(obj)
    pool.shutdown(wait=True)
    for sub in submits:
      scores.append(sub.result())
  else:
    for run in range(runs):
      score,result_msg = snn_run(dataname,if_with_val, **new_conf)
      scores.append(score)
  return np.mean(scores), np.std(scores),result_msg

def search_params(dataname, runs, log_dir,if_with_val):

  params_set = {"T": np.array([ 32,64,128,256,512]),
      "learning_rate": np.array([0.0015, 0.002, 0.0025,0.003,0.01, 0.015, 0.02, 0.025, 0.03]),}


  best_score, std, best_params = sharedutils.grid_search(dataname, runs, params_set,model_startup,if_with_val)
  msg = "sgc; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
  print(msg)



  sharedutils.add_log(os.path.join(log_dir, "snn_search.log"), msg)


if __name__ == '__main__':
  print('* Set parameters in models_conf.json, such as device": "cuda:0"')
  do_search_params = "no"
  dataname="coauthor_phy"
  runs = 3
  if_with_val = "no" #or "yes"

  # do the parameter search for each dataset
  if do_search_params == "yes":
    if if_with_val == "no":
      allconfs = sharedutils.read_config("./models_conf.json")
      search_params(dataname, runs, allconfs["snn"]["log_dir"], if_with_val)

    if if_with_val == "yes":
      allconfs = sharedutils.read_config("./models_conf.json")
      search_params(dataname, runs, allconfs["snn"]["log_dir"], if_with_val)


  else:
    me, st, result_msg = model_startup(dataname, runs, if_with_val)
    print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))