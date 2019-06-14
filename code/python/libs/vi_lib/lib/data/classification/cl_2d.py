import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


class Task:
    def __init__(self, means, covs, labels, no_datapoints, batch_size, use_cuda=False):
        self.means = means
        self.covs = covs
        self.labels = labels
        self.no_datapoints = no_datapoints
        self.batch_size = batch_size

        features_list = []
        targets_list = []
        for label, mean, cov in zip(labels, means, covs):
            features = self.sample(mean, cov, torch.Size([self.no_datapoints]))
            targets = torch.tensor(np.repeat(label, no_datapoints))
            features_list.append(features)
            targets_list.append(targets)

        self.x_train = torch.cat(features_list, dim=0)
        self.y_train = torch.cat(targets_list, dim=0)

        data = data_utils.TensorDataset(self.x_train, self.y_train)
        self.train_loader = data_utils.DataLoader(data,
            batch_size=self.batch_size, shuffle=True)

        if use_cuda:
            self.x_train, self.y_train = (self.x_train.cuda(),
                self.y_train.cuda())

        features_list = []
        targets_list = []
        # Create test datasets
        for label, mean, cov in zip(labels, means, covs):
            cur_label = label
            features = self.sample(mean, cov, torch.Size([no_datapoints]))
            targets = torch.tensor(np.repeat(cur_label, no_datapoints))
            features_list.append(features)
            targets_list.append(targets)

        self.x_test = torch.cat(features_list, dim=0)
        self.y_test = torch.cat(targets_list, dim=0)

    def sample(self, mean, cov, no_samples=500):
        return torch.distributions.multivariate_normal.MultivariateNormal(
            mean, cov).sample(no_samples)


class ClassificationTasks2D:
    def __init__(self, no_classes, no_datapoints, batch_size, use_cuda):
        self.tasks = []
        self.no_classes = no_classes
        self.no_datapoints = no_datapoints
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        # Generate 2d classification tasks.
        # One task is classification of data coming from two Gaussian distributions
        # One distribution is always the same. Mean of another one is rotated 
        # by (task id * 360 / number of tasks) to create a new distribution.
        no_tasks = 5
        mean1 = [1., 1.]
        mean2 = [1., 7.]
        cov1 = [[2., 0.], [0., 2.]]
        cov2 = [[1., .3], [.3, 2.]]
        for task_id in range(no_tasks):
            # Rotate the second Gaussian mean
            # by alpha = (360 / no_tasks) degrees clock-wise
            alpha = task_id * 2. * np.pi / no_tasks
            mean_x = (np.cos(alpha) * (mean2[0] - mean1[0]) +
                np.sin(alpha) * (mean2[1] - mean1[1]) + mean1[0])
            mean_y = (- np.sin(alpha) * (mean2[0] - mean1[0]) +
                np.cos(alpha) * (mean2[1] - mean1[1]) + mean1[1])
            self.tasks.append(Task(
                means=[torch.tensor(mean1), torch.tensor([mean_x, mean_y])],
                covs=[torch.tensor(cov1), torch.tensor(cov2)],
                labels=list(range(self.no_classes)),
                no_datapoints=self.no_datapoints, batch_size=self.batch_size,
                use_cuda=self.use_cuda))

    def get_tasks(self):
        return self.tasks

    def plot_contours(self, x, y, no_classes, task_id, pred_func, filename,
        x2_list=None):
        no_mesh_points = 2000

        fig = plt.figure()
        plt.title('2D Classification')
        ax = plt.gca()

        x_min, x_max = x.numpy().min() - 1., x.numpy().max() + 1.
        for xx in x2_list:
            x_min = min(x_min, np.asarray(xx).min() - 1.)
            x_max = max(x_max, np.asarray(xx).max() + 1.)

        ax.set_xticks(range(int(np.floor(x_min - 1.)), int(np.ceil(x_max + 1.)), 4))
        ax.set_yticks(range(int(np.floor(x_min - 1.)), int(np.ceil(x_max + 1.)), 4))
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

        x1, x2 = np.meshgrid(np.linspace(x_min, x_max, no_mesh_points),
            np.linspace(x_min, x_max, no_mesh_points))
        new_points = np.asarray([x1.ravel(), x2.ravel()]).T
        preds = F.softmax(pred_func(
            x=torch.tensor(new_points, dtype=torch.float32), task_id=task_id),
            dim=1)
        np.set_printoptions(threshold=np.nan)
        preds1 = preds.detach().numpy()[:, 0].reshape(x1.shape)
        levels = [0.3, 0.5, 0.9]
        ct = plt.contour(x1, x2, preds1, levels, colors='r')
        plt.clabel(ct, inline=True, fontsize=10, fmt='%2.1f', colors='k')

        no_points = int(len(x) / no_classes)
        if x2_list is None:
            colors = ['r', 'b']
            for i in range(no_classes):
                x1, x2 = x.numpy()[i * no_points : (i + 1) * no_points, :].T
                plt.scatter(x1, x2, c=colors[i])
        else:
            x1, x2 = x.numpy()[:no_points, :].T
            plt.scatter(x1, x2, c='r')
            for i, px in enumerate(x2_list):
                x1, x2 = px.T
                alpha = (i + 1) * 1. / (len(x2_list))
                if i < len(x2_list) - 1:
                    alpha *= .5
                plt.scatter(x1, x2, c='b', alpha=alpha)

        if filename is not None:    
            fig.savefig(filename)
