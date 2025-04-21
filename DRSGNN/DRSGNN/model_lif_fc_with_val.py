import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import encoding, functional
from os.path import join as pjoin
import sharedutils
from leanable_thre import LIFSpike

import logging
import os

# log_filename = os.path.join(os.getcwd(), "training.log")
#
# logger = logging.getLogger("train_logger")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fh = logging.FileHandler(log_filename, mode="w")
#     fh.setLevel(logging.INFO)
#
#     formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
#     fh.setFormatter(formatter)
#
#     logger.addHandler(fh)
#
# logger.propagate = False
#
# logger.info("Logger setup complete. Logging to training.log.")


def model_lif_fc_with_val(dataname, dataset_dir, device, batch_size,
                 learning_rate, T, tau, v_threshold, v_reset, train_epoch, log_dir, 
                 n_labels, n_dim0, n_dim1, n_dim2, train_data_loader,
                 val_data_loader, test_data_loader,PE_dim):
    # init
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_dim1 * n_dim2, n_labels, bias=False),
        nn.Dropout(p=0.6),
        LIFSpike(n_labels)
    )
    net = net.to(device)



    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_val_accuracy = 0
    model_pth = pjoin('./tmpdir/snn/',dataname,'best_snn.model')
    val_accs, train_accs = [], []


    for epoch in range(train_epoch):
        net.train()
        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 
            
        for rind, (img, label) in enumerate(train_data_loader):
            img = img.to(device)
            label = label.long().to(device)
            label_one_hot = F.one_hot(label, n_labels).float()
            optimizer.zero_grad()


            for t in range(T):
                if t == 0: out_spikes_counter = net(encoder(img[:,:n_dim2]).float())
                else: out_spikes_counter += net(encoder(img[:,:n_dim2]).float())

            out_spikes_counter_frequency = out_spikes_counter / T

            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            
            train_accs.append(accuracy)

            train_times += 1
        scheduler.step()

        net.eval()
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            for img, label in val_data_loader:
                img = img.to(device)
                n_imgs = img.shape[0]
                out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
                for t in range(T):
                    out_spikes_counter += net(encoder(img[:,:n_dim2]).float())

                correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            val_accuracy = correct_sum / test_sum
            val_accs.append(val_accuracy)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                torch.save(net, model_pth)
        # print(f'Epoch {epoch}: device={device}, dataset_dir={dataset_dir}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, log_dir={log_dir}, max_train_accuracy={train_accs[-1]:.4f},max_val_accuracy={max_val_accuracy:.4f}, train_times={train_times}', end="\r")
        logger.info(
            f'Epoch {epoch}: device={device}, dataset_dir={dataset_dir}, batch_size={batch_size}, '
            f'learning_rate={learning_rate}, T={T}, log_dir={log_dir}, '
            f'max_train_accuracy={train_accs[-1]:.4f}, max_val_accuracy={max_val_accuracy:.4f}, '
            f'train_times={train_times}'
        )

    # test
    best_snn = torch.load(model_pth)
    best_snn.eval()
    best_snn.to(device)
    max_test_accuracy = 0.0
    result_sops, result_num_spikes_1, result_num_spikes_2 = 0, 0, 0
    with torch.no_grad():
        test_sum, correct_sum = 0, 0
        for img, label in test_data_loader:
            img = img.to(device)
            n_imgs = img.shape[0]
            out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
            denominator = n_imgs * len(test_data_loader)
            for t in range(T):
                enc_img = encoder(img[:,:n_dim2]).float()
                out_spikes_counter += best_snn(enc_img)
                # pre spikes
                result_num_spikes_1 += torch.sum(enc_img) / denominator
            # post spikes
            result_num_spikes_2 += torch.sum(out_spikes_counter) / denominator
            correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            test_sum += label.numel()

            functional.reset_net(best_snn)

        test_accuracy = correct_sum / test_sum
        max_test_accuracy = max(max_test_accuracy, test_accuracy)
    result_msg = f'testset\'acc: device={device}, dataset={dataname}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, max_test_accuracy={max_test_accuracy:.4f}'
    result_msg += f", num_s1: {int(result_num_spikes_1)}, num_s2: {int(result_num_spikes_2)}"
    result_msg += f", num_s_per_node: {int(result_num_spikes_1)+int(result_num_spikes_2)}"
    sharedutils.add_log(pjoin(log_dir, "snn_search.log"), result_msg)
    print(result_msg)

    logger.info(result_msg)

    return max_test_accuracy, result_msg