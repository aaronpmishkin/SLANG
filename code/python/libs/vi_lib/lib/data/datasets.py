# @Author: amishkin
# @Date:   18-08-01
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-23

import os
import warnings
import torch
import numpy as np
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import sklearn.model_selection as modsel


###########################################
## PyTorch Dataset for presplit datasets ##
###########################################

from lib.data.classification.axa import Axa
from lib.data.classification.usps import Usps
from lib.data.classification.usps_3vs5 import Usps_3vs5


#################################################
## PyTorch Dataset for randomly split datasets ##
#################################################

from lib.data.classification.australian import Australian

###################################################
## PyTorch Dataset for locally presplit datasets ##
###################################################

from lib.data.classification.australian_presplit import AustralianPresplit
from lib.data.classification.breast_cancer_presplit import BreastCancerPresplit
from lib.data.classification.colon_cancer_presplit import ColonCancerPresplit
from lib.data.classification.covtype_presplit import CovtypePresplit

###############################################
## PyTorch Dataset for presplit UCI datasets ##
###############################################

from lib.data.regression.boston_presplit import BostonPresplit
from lib.data.regression.concrete_presplit import ConcretePresplit
from lib.data.regression.energy_presplit import EnergyPresplit
from lib.data.regression.kin8nm_presplit import Kin8nmPresplit
from lib.data.regression.naval_presplit import NavalPresplit
from lib.data.regression.powerplant_presplit import PowerplantPresplit
from lib.data.regression.wine_presplit import WinePresplit
from lib.data.regression.yacht_presplit import YachtPresplit

from lib.data.regression.cov_viz_1d import CovViz1D

DEFAULT_DATA_FOLDER = "~/data"


################################################
## Construct class for dealing with data sets ##
################################################

class Dataset():
    def __init__(self, data_set, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        if data_set == "mnist":
            self.train_set = dset.MNIST(root = data_folder,
                                        train = True,
                                        transform = transforms.ToTensor(),
                                        download = True)

            self.test_set = dset.MNIST(root = data_folder,
                                       train = False,
                                       transform = transforms.ToTensor())

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        if data_set == "mnist_val":
            full_train_set = dset.MNIST(root = data_folder,
                                        train = True,
                                        transform = transforms.ToTensor(),
                                        download = True)

            self.train_set = torch.utils.data.Subset(full_train_set, np.arange(start=0, stop=50000))

            self.test_set = torch.utils.data.Subset(full_train_set, np.arange(start=50000, stop=60000))

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "smnist":
            self.test_set = dset.MNIST(root = data_folder,
                                       train = True,
                                       transform = transforms.ToTensor(),
                                       download = True)

            self.train_set = dset.MNIST(root = data_folder,
                                        train = False,
                                        transform = transforms.ToTensor())

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "usps":
            self.train_set = Usps(root = data_folder,
                                  train = True)
            self.test_set = Usps(root = data_folder,
                                 train = False)

            self.task = "classification"
            self.num_features = 16 * 16
            self.num_classes = 10

        elif data_set == "usps_3vs5":
            self.train_set = Usps_3vs5(root = data_folder,
                                  train = True)
            self.test_set = Usps_3vs5(root = data_folder,
                                 train = False)

            self.task = "classification"
            self.num_features = 16 * 16
            self.num_classes = 2

        elif data_set in ("a1a","a2a","a3a","a4a","a5a","a6a","a7a","a8a","a9a"):
            self.train_set = Axa(root = data_folder,
                                 data_set = data_set,
                                 train = True)
            self.test_set = Axa(root = data_folder,
                                data_set = data_set,
                                train = False)

            self.task = "classification"
            self.num_features = 123
            self.num_classes = 2

        elif data_set == "australian_presplit":
            self.train_set = AustralianPresplit(root = data_folder,
                                                train = True)
            self.test_set = AustralianPresplit(root = data_folder,
                                               train = False)

            self.task = "classification"
            self.num_features = 14
            self.num_classes = 2

        elif data_set == "australian":
            data = Australian(root = data_folder,
                                                train = True)

            splits = random_split(data, [345, 345])
            self.train_set = splits[0]
            self.test_set = splits[1]

            self.task = "classification"
            self.num_features = 14
            self.num_classes = 2

        elif data_set == "breastcancer_presplit":
            self.train_set = BreastCancerPresplit(root = data_folder,
                                                  train = True)
            self.test_set = BreastCancerPresplit(root = data_folder,
                                                 train = False)

            self.task = "classification_presplit"
            self.num_features = 10
            self.num_classes = 2

        elif data_set == "coloncancer_presplit":
            self.train_set = ColonCancerPresplit(root = data_folder,
                                                 train = True)
            self.test_set = ColonCancerPresplit(root = data_folder,
                                                train = False)

            self.task = "classification"
            self.num_features = 2000
            self.num_classes = 2

        elif data_set == "covtype_presplit":
            self.train_set = CovtypePresplit(root = data_folder,
                                             train = True)
            self.test_set = CovtypePresplit(root = data_folder,
                                            train = False)

            self.task = "classification"
            self.num_features = 54
            self.num_classes = 2

        elif data_set in ["boston" + str(i) for i in range(20)]:
            self.train_set = BostonPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = BostonPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 13
            self.num_classes = None

        elif data_set in ["concrete" + str(i) for i in range(20)]:
            self.train_set = ConcretePresplit(root = data_folder,
                                              data_set = data_set,
                                              train = True)
            self.test_set = ConcretePresplit(root = data_folder,
                                             data_set = data_set,
                                             train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["energy" + str(i) for i in range(20)]:
            self.train_set = EnergyPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = EnergyPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["kin8nm" + str(i) for i in range(20)]:
            self.train_set = Kin8nmPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = Kin8nmPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["naval" + str(i) for i in range(20)]:
            self.train_set = NavalPresplit(root = data_folder,
                                           data_set = data_set,
                                           train = True)
            self.test_set = NavalPresplit(root = data_folder,
                                          data_set = data_set,
                                          train = False)

            self.task = "regression"
            self.num_features = 16
            self.num_classes = None

        elif data_set in ["powerplant" + str(i) for i in range(20)]:
            self.train_set = PowerplantPresplit(root = data_folder,
                                                data_set = data_set,
                                                train = True)
            self.test_set = PowerplantPresplit(root = data_folder,
                                               data_set = data_set,
                                               train = False)

            self.task = "regression"
            self.num_features = 4
            self.num_classes = None

        elif data_set in ["wine" + str(i) for i in range(20)]:
            self.train_set = WinePresplit(root = data_folder,
                                          data_set = data_set,
                                          train = True)
            self.test_set = WinePresplit(root = data_folder,
                                         data_set = data_set,
                                         train = False)

            self.task = "regression"
            self.num_features = 11
            self.num_classes = None

        elif data_set in ["yacht" + str(i) for i in range(20)]:
            self.train_set = YachtPresplit(root = data_folder,
                                           data_set = data_set,
                                           train = True)
            self.test_set = YachtPresplit(root = data_folder,
                                          data_set = data_set,
                                          train = False)

            self.task = "regression"
            self.num_features = 6
            self.num_classes = None

        elif data_set.startswith("cov_viz_1d"):
            self.train_set = CovViz1D(root=data_folder, data_set=data_set)
            self.test_set = self.train_set
            warnings.warn("CovViz does not have a test set - the test set mirrors the training set.")
            self.task = "regression"
            self.num_features = 1
            self.num_classes = None

        else:
            RuntimeError("Unknown data set")

    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(dataset = self.train_set,
                                  batch_size = batch_size,
                                  shuffle = shuffle)
        return train_loader

    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(dataset = self.test_set,
                                 batch_size = batch_size,
                                 shuffle = shuffle)
        return test_loader

    def load_full_train_set(self, use_cuda=torch.cuda.is_available()):

        full_train_loader = DataLoader(dataset = self.train_set,
                                       batch_size = len(self.train_set),
                                       shuffle = False)

        x_train, y_train = next(iter(full_train_loader))

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_full_test_set(self, use_cuda=torch.cuda.is_available()):

        full_test_loader = DataLoader(dataset = self.test_set,
                                      batch_size = len(self.test_set),
                                      shuffle = False)

        x_test, y_test = next(iter(full_test_loader))

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test



#######################################################################
## Construct class for dealing with data sets using cross-validation ##
#######################################################################

class DatasetCV():
    def __init__(self, data_set, n_splits=3, seed=None, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        self.n_splits = n_splits
        self.seed = seed
        self.current_split = 0

        if data_set == "mnist":
            self.data = dset.MNIST(root = data_folder,
                                   train = True,
                                   transform = transforms.ToTensor(),
                                   download = True)

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "smnist":
            self.data = dset.MNIST(root = data_folder,
                                   train = False,
                                   transform = transforms.ToTensor(),
                                   download = True)

            self.task = "classification"
            self.num_features = 28 * 28
            self.num_classes = 10

        elif data_set == "usps":
            self.data = Usps(root = data_folder,
                             train = True)

            self.task = "classification"
            self.num_features = 16 * 16
            self.num_classes = 10

        elif data_set == "usps_3vs5":
            self.data = Usps_3vs5(root = data_folder,
                                  train = True)

            self.task = "classification"
            self.num_features = 16 * 16
            self.num_classes = 2

        elif data_set in ("a1a","a2a","a3a","a4a","a5a","a6a","a7a","a8a","a9a"):
            self.data = Axa(root = data_folder,
                            data_set = data_set,
                            train = True)

            self.task = "classification"
            self.num_features = 123
            self.num_classes = 2

        elif data_set == "australian_presplit":
            self.data = AustralianPresplit(root = data_folder,
                                           train = True)

            self.task = "classification"
            self.num_features = 14
            self.num_classes = 2

        elif data_set == "breastcancer_presplit":
            self.data = BreastCancerPresplit(root = data_folder,
                                             train = True)

            self.task = "classification_presplit"
            self.num_features = 10
            self.num_classes = 2

        elif data_set == "coloncancer_presplit":
            self.data = ColonCancerPresplit(root = data_folder,
                                            train = True)

            self.task = "classification"
            self.num_features = 2000
            self.num_classes = 2

        elif data_set == "covtype_presplit":
            self.data = CovtypePresplit(root = data_folder,
                                        train = True)

            self.task = "classification"
            self.num_features = 54
            self.num_classes = 2

        elif data_set in ["boston" + str(i) for i in range(20)]:
            self.data = BostonPresplit(root = data_folder,
                                       data_set = data_set,
                                       train = True)

            self.task = "regression"
            self.num_features = 13
            self.num_classes = None

        elif data_set in ["concrete" + str(i) for i in range(20)]:
            self.data = ConcretePresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["energy" + str(i) for i in range(20)]:
            self.data = EnergyPresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["kin8nm" + str(i) for i in range(20)]:
            self.data = Kin8nmPresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["naval" + str(i) for i in range(20)]:
            self.data = NavalPresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 16
            self.num_classes = None

        elif data_set in ["powerplant" + str(i) for i in range(20)]:
            self.data = PowerplantPresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 4
            self.num_classes = None

        elif data_set in ["wine" + str(i) for i in range(20)]:
            self.data = WinePresplit(root = data_folder,
                                         data_set = data_set,
                                         train = True)

            self.task = "regression"
            self.num_features = 11
            self.num_classes = None

        elif data_set in ["yacht" + str(i) for i in range(20)]:
            self.data = YachtPresplit(root = data_folder,
                                      data_set = data_set,
                                      train = True)

            self.task = "regression"
            self.num_features = 6
            self.num_classes = None

        else:
            RuntimeError("Unknown data set")

        # Store CV splits
        cv = modsel.KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = cv.split(range(len(self.data)))

        self.split_idx_val = []
        for (_, idx_val) in splits:
            self.split_idx_val.append(idx_val)

    def get_full_data_size(self):
        return len(self.data)

    def get_current_split(self):
        return self.current_split

    def set_current_split(self, split):
        if split >= 0 and split <= self.n_splits-1:
            self.current_split = split
        else:
            RuntimeError("Split higher than number of splits")

    def _get_current_val_idx(self):
        return self.split_idx_val[self.current_split]

    def _get_current_train_idx(self):
        return np.setdiff1d(range(len(self.data)), self.split_idx_val[self.current_split])

    def get_current_val_size(self):
        return len(self._get_current_val_idx())

    def get_current_train_size(self):
        return len(self.data) - len(self._get_current_val_idx())

    def get_current_train_loader(self, batch_size, shuffle=True):
        train_set = torch.utils.data.Subset(self.data, self._get_current_train_idx())
        train_loader = DataLoader(dataset = train_set,
                                  batch_size = batch_size,
                                  shuffle = shuffle)
        return train_loader

    def get_current_val_loader(self, batch_size, shuffle=True):
        val_set = torch.utils.data.Subset(self.data, self._get_current_val_idx())
        val_loader = DataLoader(dataset = val_set,
                                batch_size = batch_size,
                                shuffle = shuffle)
        return val_loader

    def load_current_train_set(self, use_cuda=torch.cuda.is_available()):
        x_train, y_train = self.data.__getitem__(self._get_current_train_idx())

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_current_val_set(self, use_cuda=torch.cuda.is_available()):
        x_test, y_test = self.data.__getitem__(self._get_current_val_idx())

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test
