import os
import torch
import numpy as np
import torch.utils.data as data

class Usps_3vs5(data.Dataset):

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if self.train:
            self.training_file_x = os.path.join(self.root, "usps_resampled/usps_resampled_X_train.csv")
            self.training_file_y = os.path.join(self.root, "usps_resampled/usps_resampled_y_train.csv")
            X_train = np.loadtxt(self.training_file_x).T
            y_train = np.loadtxt(self.training_file_y).T
            y_train = np.where(y_train==1)[1]

            three_indices = y_train == 3
            five_indices = y_train == 5
            y_train[three_indices] = 0
            y_train[five_indices] = 1

            y_train = y_train[three_indices | five_indices]
            X_train = X_train[three_indices | five_indices]
            self.train_data, self.train_labels = torch.FloatTensor(X_train), torch.LongTensor(y_train)
        else:
            self.test_file_x = os.path.join(self.root, "usps_resampled/usps_resampled_X_test.csv")
            self.test_file_y = os.path.join(self.root, "usps_resampled/usps_resampled_y_test.csv")
            X_test = np.loadtxt(self.test_file_x).T
            y_test = np.loadtxt(self.test_file_y).T
            y_test = np.where(y_test==1)[1]

            three_indices = y_test == 3
            five_indices = y_test == 5
            y_test[three_indices] = 0
            y_test[five_indices] = 1
            y_test = y_test[three_indices | five_indices]
            X_test = X_test[three_indices | five_indices]
            self.test_data, self.test_labels = torch.FloatTensor(X_test), torch.LongTensor(y_test)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            x, y = self.train_data[index], self.train_labels[index]
        else:
            x, y = self.test_data[index], self.test_labels[index]

        return x, y

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
