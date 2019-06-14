from copy import deepcopy

# from keras.datasets import mnist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


class Task:
    def __init__(self, x_train, y_train, x_test, y_test, batch_size, use_cuda=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size

        data = data_utils.TensorDataset(self.x_train, self.y_train)
        self.train_loader = data_utils.DataLoader(data,
            batch_size=self.batch_size, shuffle=True)

        if use_cuda:
            self.x_train, self.y_train = (self.x_train.cuda(),
                self.y_train.cuda())
            self.x_test, self.y_test = (self.x_test.cuda(), self.y_test.cuda())


class PermutedMnistGenerator():
  def __init__(self, max_iter=10):
    # (self.X_train, self.Y_train), (self.X_test, self.Y_test) = \
    # mnist.load_data()


    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=60000, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=10000, shuffle=True)

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor()
    #                    ])),
    #     batch_size=60000, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor()
    #                    ])),
    #     batch_size=10000, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=60000, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=10000, shuffle=True)


    # There should be only 1 batch 
    for x, y in train_loader:
        self.X_train = x.numpy().reshape(-1, 784) / 255.
        self.Y_train = y.numpy()

    for x, y in test_loader:
        self.X_test = x.numpy().reshape(-1, 784) / 255.
        self.Y_test = y.numpy()

    print(f'shapes: {self.X_train.shape}, {self.Y_train.shape}')

    del train_loader
    del test_loader

    # self.X_train = self.X_train.reshape(-1, 784).astype(np.float32) / 255.
    # self.X_test = self.X_test.reshape(-1, 784).astype(np.float32) / 255.
    # self.Y_train = self.Y_train.astype(np.int)
    # self.Y_test = self.Y_test.astype(np.int)
    self.max_iter = max_iter
    self.cur_iter = 0

  def get_N(self):
    return self.X_train.shape[0]

  def get_dims(self):
    # Get data input and output dimensions
    return self.X_train.shape[1], 10

  def next_task(self):
    if self.cur_iter >= self.max_iter:
        raise Exception('Number of tasks exceeded!')
    else:
        # np.random.seed(self.cur_iter)
        np.random.seed(self.cur_iter)
        perm_inds = [i for i in range(self.X_train.shape[1])]
        np.random.shuffle(perm_inds)

        # Retrieve train data
        next_x_train = deepcopy(self.X_train)
        next_x_train = next_x_train[:,perm_inds]
        # next_y_train = np.eye(10)[self.Y_train].astype(np.int)
        next_y_train = deepcopy(self.Y_train)

        # Retrieve test data
        next_x_test = deepcopy(self.X_test)
        next_x_test = next_x_test[:,perm_inds]
        # next_y_test = np.eye(10)[self.Y_test].astype(np.int)
        next_y_test = deepcopy(self.Y_test)

        self.cur_iter += 1

        return next_x_train, next_y_train, next_x_test, next_y_test


class PermutedMnistTasks:
    def __init__(self, no_tasks, batch_size, use_cuda):
        self.tasks = []
        self.use_cuda = use_cuda

        generator = PermutedMnistGenerator()
        for task_id in range(no_tasks):
            x_train, y_train, x_test, y_test = generator.next_task()

            x_train, y_train, x_test, y_test = (
                torch.from_numpy(x_train), torch.from_numpy(y_train),
                torch.from_numpy(x_test), torch.from_numpy(y_test))
            self.tasks.append(Task(x_train, y_train, x_test, y_test,
                batch_size, use_cuda))

        self.input_size, self.output_size = generator.get_dims()
        self.no_datapoints = generator.get_N()

    # def get_tasks(self):
    #     return self.tasks

    # def plot_contours(self, x, y, no_classes, task_id, pred_func, filename,
    #     x2_list=None):
    #     no_mesh_points = 2000

    #     fig = plt.figure()
    #     plt.title('2D Classification')
    #     ax = plt.gca()

    #     x_min, x_max = x.numpy().min() - 1., x.numpy().max() + 1.
    #     for xx in x2_list:
    #         x_min = min(x_min, np.asarray(xx).min() - 1.)
    #         x_max = max(x_max, np.asarray(xx).max() + 1.)

    #     ax.set_xticks(range(int(np.floor(x_min - 1.)), int(np.ceil(x_max + 1.)), 4))
    #     ax.set_yticks(range(int(np.floor(x_min - 1.)), int(np.ceil(x_max + 1.)), 4))
    #     ax.set_xlabel(r'$x_1$')
    #     ax.set_ylabel(r'$x_2$')

    #     x1, x2 = np.meshgrid(np.linspace(x_min, x_max, no_mesh_points),
    #         np.linspace(x_min, x_max, no_mesh_points))
    #     new_points = np.asarray([x1.ravel(), x2.ravel()]).T
    #     preds = F.softmax(pred_func(torch.tensor(new_points, dtype=torch.float32), task_id),
    #         dim=1)
    #     np.set_printoptions(threshold=np.nan)
    #     preds1 = preds.detach().numpy()[:, 0].reshape(x1.shape)
    #     levels = [0.3, 0.5, 0.9]
    #     ct = plt.contour(x1, x2, preds1, levels, colors='r')
    #     plt.clabel(ct, inline=True, fontsize=10, fmt='%2.1f', colors='k')

    #     no_points = int(len(x) / no_classes)
    #     if x2_list is None:
    #         colors = ['r', 'b']
    #         for i in range(no_classes):
    #             x1, x2 = x.numpy()[i * no_points : (i + 1) * no_points, :].T
    #             plt.scatter(x1, x2, c=colors[i])
    #     else:
    #         x1, x2 = x.numpy()[:no_points, :].T
    #         plt.scatter(x1, x2, c='r')
    #         for i, px in enumerate(x2_list):
    #             x1, x2 = px.T
    #             alpha = (i + 1) * 1. / (len(x2_list))
    #             if i < len(x2_list) - 1:
    #                 alpha *= .5
    #             plt.scatter(x1, x2, c='b', alpha=alpha)

    #     if filename is not None:    
    #         fig.savefig(filename)
