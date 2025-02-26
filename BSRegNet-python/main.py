import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from dataset import PDBSNetData
from model import PDBSNet
from utils import setup_seed, TensorToHSI, get_auc, compute_diff1, compute_diff2

diff1 = True


class Trainer(object):
    """
    Trains a model
    """

    def __init__(self,
                 opt,
                 model,
                 criterion,
                 optimizer,
                 dataloader,
                 device,
                 model_path: str,
                 logs_path: str,
                 save_freq: int = 1000,
                 scheduler=None):
        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training sets provided in `dataloader`
        using `criterion` and `optimizer`.
        Saves model weight snapshots every `save_freq` epochs and saves the
        weights at the end of training.
        Parameters
        ----------
        model : torch model object, with callable `forward` method.
        criterion : callable taking inputs and targets, returning loss.
        optimizer : torch.optim optimizer.
        dataloader : train dataloaders.
        model_path : string. output path for model.
        logs_path : string. output path for log.
        save_freq : integer. Number of epochs between model checkpoints. Default = 50.
        scheduler : learning rate scheduler.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.model_path = model_path
        self.logs_path = logs_path
        self.save_freq = save_freq
        self.scheduler = scheduler
        self.opt = opt

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.log_output = open(f"{self.logs_path}/log.txt", 'w')

        self.writer = SummaryWriter(logs_path)

        print(self.opt)
        print(self.opt, file=self.log_output)

    def train_epoch(self) -> None:
        # Run a train phase for each epoch
        self.model.train(True)
        loss_train = []

        loader_train = self.dataloader.to(self.device)

        # forward net
        outputs = self.model(loader_train)

        # backward net
        self.optimizer.zero_grad()

        if diff1 == True:
            loss = self.criterion(outputs, loader_train) + 2.0 * compute_diff1(outputs) + 0.0 * compute_diff2(outputs)
        else:
            loss = self.criterion(outputs, loader_train)

        loss.backward()
        self.optimizer.step()

        # get losses
        loss_train = loss.item()

        print("Train Loss:" + str(round(loss_train, 4)))

        print("Train Loss:" + str(round(loss_train, 4)), file=self.log_output)

        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'Loss_train': np.mean(loss_train)
        }
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.epoch + 1)

        # Saving model
        if ((self.epoch + 1) % self.save_freq == 0):
            torch.save(self.model.state_dict(), os.path.join(self.model_path,
                                                             'PDBSNet' + '_' + self.opt.dataset + '_' + str(
                                                                 self.epoch + 1) + '.pkl'))

    def train(self) -> nn.Module:
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs))
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs), file=self.log_output)
            print('-' * 50)
            # run training epoch
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
        return self.model


def train_model(opt):
    DB = opt.dataset

    expr_dir = os.path.join('./checkpoints/', DB)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    prefix = 'PDBSNet' + '_epoch_' + str(opt.epochs) + '_learning_rate_' + str(
        opt.learning_rate) + '_factor_train_' + str(opt.factor_train) + '_gpu_ids_' + str(opt.gpu_ids)

    trainfile = os.path.join(expr_dir, prefix)
    if not os.path.exists(trainfile):
        os.makedirs(trainfile)

    # Device
    device = torch.device('cuda:{}'.format(opt.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')

    # Directories for storing model and output samples
    model_path = os.path.join(trainfile, 'model')

    logs_path = os.path.join(trainfile, './logs')

    setup_seed(opt.seed)

    loader_train, band = PDBSNetData(opt)
    net = PDBSNet(factor=opt.factor_train, nch_in=band, nch_out=band, nch_ker=opt.nch_ker, nblk=opt.nblk).to(device)

    # Define Optimizers and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
    scheduler_net = None

    if opt.lossm.lower() == 'l1':
        criterion = nn.L1Loss().to(device)  # Regression loss: L1
    elif opt.lossm.lower() == 'l2':
        criterion = nn.MSELoss().to(device)  # Regression loss: L2

    if torch.cuda.is_available():
        print('Model moved to CUDA compute device.')
    else:
        print('No CUDA available, running on CPU!')

    # Training
    t_begin = time.time()
    trainer = Trainer(opt,
                      net,
                      criterion,
                      optimizer,
                      loader_train,
                      device,
                      model_path,
                      logs_path,
                      scheduler=scheduler_net)
    trainer.train()
    t_end = time.time()
    print('Time of training-{}'.format((t_end - t_begin)))


def predict(opt):
    DB = opt.dataset

    expr_dir = os.path.join('./checkpoints/', DB)
    prefix = 'PDBSNet' + '_epoch_' + str(opt.epochs) + '_learning_rate_' + str(
        opt.learning_rate) + '_factor_train_' + str(opt.factor_train) + '_gpu_ids_' + str(opt.gpu_ids)

    trainfile = os.path.join(expr_dir, prefix)

    model_path = os.path.join(trainfile, 'model')

    expr_dirs = os.path.join('./result/', DB)
    if not os.path.exists(expr_dirs):
        os.makedirs(expr_dirs)

    log_output = open(f"{expr_dirs}/log.txt", 'w')

    # model_weights = os.path.join(model_path, 'PDBSNet' + '_' + opt.dataset + '_350.pkl')
    model_weights = os.path.join(model_path, 'PDBSNet' + '_' + opt.dataset + '_' + str(opt.epochs) + '.pkl')

    # test datalodar
    data_dir = '../BS_data/'
    image_file = data_dir + opt.dataset + '.mat'

    input_data = sio.loadmat(image_file)
    # image = input_data['nor_hsi']
    image = input_data['data']
    image = image.astype(np.float32)
    gt = input_data['map']
    gt = gt.astype(np.float32)

    image = ((image - image.min()) / (image.max() - image.min()))

    band = image.shape[2]

    test_data = np.expand_dims(image, axis=0)
    loader_test = torch.from_numpy(test_data.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

    # Device
    device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

    net = PDBSNet(factor=opt.factor_test, nch_in=band, nch_out=band, nch_ker=opt.nch_ker, nblk=opt.nblk).to(device)
    net.load_state_dict(torch.load(model_weights, map_location='cuda:0'))

    t_begin = time.time()

    net.eval()

    img_old = loader_test
    test_data = loader_test.to(device)

    img_new = net(test_data)

    HSI_old = TensorToHSI(img_old)
    HSI_new = TensorToHSI(img_new)

    auc, detectmap = get_auc(HSI_old, HSI_new, gt)

    t_end = time.time()

    print("AUC: " + str(auc))
    print("AUC: " + str(auc), file=log_output)

    print('Time of testing-{}s'.format((t_end - t_begin)))
    print('Time of testing-{}s'.format((t_end - t_begin)), file=log_output)

    sio.savemat(os.path.join(expr_dirs, 'detectmap.mat'), {'detectmap': detectmap})
    sio.savemat(os.path.join(expr_dirs, 'reconstructed_data.mat'), {'reconstructed_data': HSI_new})


def get_default_arguments(i):
    args = argparse.Namespace()

    # 设置所有参数的默认值
    args.seed = 1
    args.gpu_ids = 0
    args.command = 'predict'  # 'train' 或者 'predict'，根据需要设置
    args.factor_train = 3
    args.factor_test = 3
    args.nch_ker = 64
    args.nblk = 6
    args.lossm = 'l1'
    args.learning_rate = 1e-4
    args.weight_decay = 1e-5
    args.epochs = 3000
    if i == 0:
        args.dataset = 'AVIRIS1_BS'
    if i == 1:
        args.dataset = 'Airport1_BS'
    if i == 2:
        args.dataset = 'Urban1_BS'
    if i == 3:
        args.dataset = 'AVIRIS2_BS'
    if i == 4:
        args.dataset = 'Urban3_BS'
    if i == 5:
        args.dataset = 'HYDICE_BS'
    if i == 6:
        args.dataset = 'Beach4_BS'
    return args


def main():
    for i in [0, 1, 3]:
        opt = get_default_arguments(i)
        if opt.command.lower() == 'train':
            train_model(opt)
        elif opt.command.lower() == 'predict':
            predict(opt)


if __name__ == '__main__':
    main()
