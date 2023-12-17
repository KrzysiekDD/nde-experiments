""" DataLoader and Datasets for MNIST-type and dynamical system datasets """
# import torchvision
# from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import ToTensor, Normalize
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dx
import utils
from jax import config

config.update("jax_enable_x64", True)

# TODO remove this later
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Definitions of dynamical systems to be learned from data
# If a custom system is not specified, the default system is number 1
def dyn_sys_1(t, y, args):
    """
    Simplest system
    :param t:
    :param y:
    :param args:
    :return:
    """
    dy_1dt = -(y[1] / (1 + y[1]))
    dy_2dt = y[0] / (1 + y[0])

    return jnp.stack([dy_1dt, dy_2dt], axis=-1)


def dyn_sys_2(t, y, args):
    """
    Dynamical system to be learned from data
    :param t:
    :param y:
    :param args:
    :return:
    """
    dy_1dt = y[0] + y[1] / (1 + jnp.sin(y[1])) + jnp.exp(-y[0])
    dy_2dt = y[1] - (y[0] / (1 + jnp.cos(y[0]))) + jnp.exp(-y[1])

    return jnp.stack([dy_1dt, dy_2dt], axis=-1)


def dyn_sys_3(t, y, args):
    """
    Dynamical system to be learned from data
    :param t:
    :param y:
    :param args:
    :return:
    """
    alpha = 1.0
    beta = 0.1
    delta = 0.075
    gamma = 1.5
    dy_1dt = alpha * y[0] - beta * y[0] * y[1]
    dy_2dt = delta * y[0] * y[1] - gamma * y[1]
    return jnp.stack([dy_1dt, dy_2dt], axis=-1)


def dyn_sys_4(t, y, args):
    """
    Dynamical system to be learned from data
    :param t:
    :param y:
    :param args:
    :return:
    """
    # a, b, c, d, e, f = 2.0, 5.0, 3.5, 1.0, 4.0, 3.0
    # dy_1dt = a * (y[1] / (1 + y[1])) - b * (y[0] / (1 + y[0]))
    # dy_2dt = c * (y[2] / (1 + y[2])) - d * (y[1] / (1 + y[1]))
    # dy_3dt = e * (y[0] / (1 + y[0])) - f * (y[2] / (1 + y[2]))

    dy_1dt = jnp.sin(t) * (y[1] / (1 + y[1]))
    dy_2dt = -2 * jnp.sin(t) * (y[0] / (1 + y[0]))
    dy_3dt = -2 * jnp.sin(t) * (y[1] / (1 + y[1]))

    return jnp.stack([dy_1dt, dy_2dt, dy_3dt], axis=-1)


def dyn_sys_5(t, y, args):
    """
    Dynamical system to be learned from data
    :param t:
    :param y:
    :param args:
    :return:
    """
    dy_1dt = y[1] / (2 + y[1])
    dy_2dt = -(2 * y[0] / (3 + y[0]))
    dy_3dt = -(4 * y[1] / (5 + y[1]))
    return jnp.stack([dy_1dt, dy_2dt, dy_3dt], axis=-1)


# @dataclass
# class MNISTData:
#     """"""
#     batch_size: int = field(default=None)
#     train_dataset: Dataset = field(default=None)
#     test_dataset: Dataset = field(default=None)
#     train_loader: DataLoader = field(default=None)
#     test_loader: DataLoader = field(default=None)
#     mnist_type: str = field(default="MNIST")
#     data_dir: str = field(default="data")
#
#     def __post_init__(self):
#         if self.batch_size is None:
#             self.batch_size = 64
#
#         normalization_transform = torchvision.transforms.Compose(
#             [ToTensor(), Normalize((0.5,), (0.5,))]
#         )
#
#         dataset_type = getattr(torchvision.datasets, self.mnist_type)
#
#         if self.train_dataset is None:
#             self.train_dataset = dataset_type(
#                 self.data_dir,
#                 train=True,
#                 download=True,
#                 transform=normalization_transform,
#             )
#             self.train_loader = DataLoader(
#                 self.train_dataset, batch_size=self.batch_size, shuffle=True
#             )
#
#         if self.test_dataset is None:
#             self.test_dataset = dataset_type(
#                 self.data_dir,
#                 train=False,
#                 download=True,
#                 transform=normalization_transform,
#             )
#             self.test_loader = DataLoader(
#                 self.test_dataset, batch_size=self.batch_size, shuffle=True
#             )


@dataclass
class NonlinearDSData:
    """
    Generates solutions for the specified dynamical system.
    Only required param is JAX's PRNGKey
    """

    key: jax._src.typing.Array
    dataset_size: int = field(default=None)
    batch_size: int = field(default=None)
    solver: dx.AbstractSolver = field(default=None)
    max_solver_steps: int = field(default=None)
    solver_step_size_controller: dx.AbstractStepSizeController = field(default=None)
    dynamical_system_number: int = field(default=None)
    custom_dynamical_system: str = field(default=None)
    t_k: float = field(default=None)
    ts_len: int = field(default=None)
    min_val: float = field(default=None)
    max_val: float = field(default=None)

    def __post_init__(self):
        """
        Set default values of parameters not provided at instantiation
        :return:
        """
        if self.dataset_size is None:
            self.dataset_size = 256

        if self.batch_size is None:
            self.batch_size = 32

        if self.solver is None:
            self.solver = dx.Tsit5()
            # For stiff problems use Kvaerno family along with PID step size controller
            # self.solver = dx.Kvaerno5()

        if self.max_solver_steps is None:
            self.max_solver_steps = 4096

        if self.solver_step_size_controller is None:
            self.solver_step_size_controller = dx.ConstantStepSize()
            # For stiff problems use Kvaerno family along with PID step size controller
            # self.solver_step_size_controller = dx.PIDController(rtol=1e-3, atol=1e-6)

        if self.dynamical_system_number is None:
            self.dynamical_system_number = 1
        # TODO Add custom system parsing
        if self.custom_dynamical_system is not None:
            self.dynamical_system = "tbd"
            self.number_of_states = "tbd"

        if self.t_k is None:
            self.t_k = 10

        if self.ts_len is None:
            self.ts_len = 100

        if self.min_val is None:
            self.min_val = -0.6

        if self.max_val is None:
            self.max_val = 1.0

        if self.custom_dynamical_system is None:
            match self.dynamical_system_number:
                case 1:
                    self.dynamical_system = dyn_sys_1
                    self.number_of_states = 2
                case 2:
                    self.dynamical_system = dyn_sys_2
                    self.number_of_states = 2
                case 3:
                    self.dynamical_system = dyn_sys_3
                    self.number_of_states = 2
                case 4:
                    self.dynamical_system = dyn_sys_4
                    self.number_of_states = 3
                case 5:
                    self.dynamical_system = dyn_sys_5
                    self.number_of_states = 3

        # if self.ts_len <= self.max_solver_steps:
        #     self.max_solver_steps = 4 * self.ts_len

    def generate_data(self):
        """
         Generates the actual training data
        :return:
        """
        ts = jnp.linspace(0, self.t_k, self.ts_len)
        keys_split = jrandom.split(self.key, self.dataset_size)
        ys = jax.vmap(lambda key: self.generate_solution(ts, key=key))(keys_split)

        return ts, ys

    def generate_solution(self, ts, *, key):
        """
        Generates one solution from random initial conditions.
        This function is vmapped as each call requires a different PRNGKey
        :return:
        """
        # TODO add minval, maxval to the argparse probably
        # if self.dynamical_system_number == 5:
        #     # This is for the ROBER problem only
        #     base_iv = jnp.array([1.0, 0.0, 0.0])
        #     epsilon = 0.05
        #     perturbations = jrandom.uniform(key, (3,), minval=-epsilon, maxval=epsilon)
        #
        #     y_0 = base_iv# + perturbations
        # else:
        y_0 = jrandom.uniform(
            key, (self.number_of_states,), minval=self.min_val, maxval=self.max_val
        )

        # The solution cannot have more data points than the maximum number of solver steps
        # This is the case for explicit RK, for more complex solvers this might be an even
        # stronger constraint, e.g. max_solver_steps = 8 * ts_len
        solution = dx.diffeqsolve(
            dx.ODETerm(self.dynamical_system),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y_0,
            max_steps=4 * self.ts_len,
            stepsize_controller=self.solver_step_size_controller,
            saveat=dx.SaveAt(ts=ts),
        )

        ys = solution.ys
        return ys

    @staticmethod
    def dataloader(arrays, batch_size, *, key):
        """
         Torch-like dataloader tweaked a bit to be compatible with JAX
        :param arrays:
        :param batch_size:
        :param key:
        :return:
        """
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)

        while True:
            permutation = jrandom.permutation(key, indices)
            (key,) = jrandom.split(key, 1)
            start = 0
            end = batch_size

            while end < dataset_size:
                batch_permutation = permutation[start:end]
                yield tuple(array[batch_permutation] for array in arrays)
                start = end
                end = start + batch_size


@dataclass
class ChaoticDSData:
    """Generates solutions for Lorenz attractor with specified coefficients"""

    key: jax._src.typing.Array
    dataset_size: int = field(default=None)
    batch_size: int = field(default=None)
    solver: dx.AbstractSolver = field(default=None)
    max_solver_steps: int = field(default=None)
    solver_step_size_controller: dx.AbstractStepSizeController = field(default=None)
    custom_params: list[float] = field(default=(10.0, 28.0, 8.0 / 3))
    t_k: float = field(default=None)
    ts_len: int = field(default=None)
    min_val: float = field(default=None)
    max_val: float = field(default=None)
    number_of_states: int = field(default=3)

    def __post_init__(self):
        if self.dataset_size is None:
            self.dataset_size = 256

        if self.batch_size is None:
            self.batch_size = 32

        if self.solver is None:
            self.solver = dx.Tsit5()

        if self.max_solver_steps is None:
            self.max_solver_steps = 4096

        if self.solver_step_size_controller is None:
            self.solver_step_size_controller = dx.ConstantStepSize()

        if self.t_k is None:
            self.t_k = 10

        if self.ts_len is None:
            self.ts_len = 100

        if self.min_val is None:
            self.min_val = -0.6

        if self.max_val is None:
            self.max_val = 1.0

    def generate_data(self):
        """
         Generates the actual training data
        :return:
        """
        ts = jnp.linspace(0, self.t_k, self.ts_len)
        keys_split = jrandom.split(self.key, self.dataset_size)
        ys = jax.vmap(lambda key: self.generate_solution(ts, key=key))(keys_split)

        return ts, ys

    def generate_solution(self, ts, *, key):
        """
        Generates one solution from random initial conditions.
        This function is vmapped as each call requires a different PRNGKey
        :return:
        """
        y_0 = jrandom.uniform(
            key, (self.number_of_states,), minval=self.min_val, maxval=self.max_val
        )

        def lorenz_system(t, y, args):
            _x, _y, _z = y
            sigma = 10.0
            rho = 28.0
            beta = 8 / 3

            dx_dt = sigma * (_y - _x)
            dy_dt = _x * (rho - _z) - _y
            dz_dt = _x * _y - beta * _z
            return jnp.stack([dx_dt, dy_dt, dz_dt], axis=-1)

        solution = dx.diffeqsolve(
            dx.ODETerm(lorenz_system),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y_0,
            max_steps=4 * self.ts_len,
            stepsize_controller=self.solver_step_size_controller,
            saveat=dx.SaveAt(ts=ts),
        )

        ys = solution.ys
        return ys

    @staticmethod
    def dataloader(arrays, batch_size, *, key):
        """
         Torch-like dataloader tweaked a bit to be compatible with JAX
        :param arrays:
        :param batch_size:
        :param key:
        :return:
        """
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)

        while True:
            permutation = jrandom.permutation(key, indices)
            (key,) = jrandom.split(key, 1)
            start = 0
            end = batch_size

            while end < dataset_size:
                batch_permutation = permutation[start:end]
                yield tuple(array[batch_permutation] for array in arrays)
                start = end
                end = start + batch_size


if __name__ == "__main__":
    # Downloads the specified MNIST dataset and saves it to "data" folder
    # in the current dir
    # mnist_data = MNISTData()
    # fashion_mnist_data = MNISTData(mnist_type="FashionMNIST")

    key_gen = utils.key_generator(0)

    ##### 1st
    # nl_ds_data = NonlinearDSData(dynamical_system_number=1, key=next(key_gen), t_k=10, ts_len=1000, min_val=-0.6, max_val=1.0)
    # ts, ys = nl_ds_data.generate_data()
    # utils.plot_2_trajectories(ts, ys, ys)

    ##### 2nd
    # nl_ds_data_2 = NonlinearDSData(dynamical_system_number=2, key=next(key_gen), t_k=20, ts_len=10000, min_val=-1., max_val=2.0)
    # ts2, ys2 = nl_ds_data_2.generate_data()
    # utils.plot_2_trajectories(ts2, ys2, ys2)

    ##### 3rd
    # nl_ds_data_3 = NonlinearDSData(
    #     dynamical_system_number=3,
    #     key=next(key_gen),
    #     t_k=20,
    #     ts_len=10000,
    #     # solver=dx.Kvaerno5(),
    #     # solver_step_size_controller=dx.PIDController(rtol=1e-3, atol=1e-6),
    #     min_val=-1.0,
    #     max_val=2.0,
    # )  # , max_solver_steps=2*10000)
    # ts3, ys3 = nl_ds_data_3.generate_data()
    # print(ys3[0, 0, 0])
    # print(ys3[0, 0, 1])
    # utils.plot_2_trajectories(ts3, ys3, ys3)
    # utils.plot_2d(ts3, ys3, ys3)

    ##### 4th
    nl_ds_data_4 = NonlinearDSData(
        dynamical_system_number=4,
        key=next(key_gen),
        t_k=50,
        ts_len=10000,
        min_val=-1,
        max_val=2,
    )
    ts4, ys4 = nl_ds_data_4.generate_data()
    plt.plot(ts4, ys4[0, :, 0], c="#e31212", linestyle="--", label=r"Test $y_1$")
    plt.plot(ts4, ys4[0, :, 1], c="#09d635", linestyle="--", label=r"Test  $y_2$")
    plt.plot(ts4, ys4[0, :, 2], c="#0e0eb0", linestyle="--", label=r"Test  $y_3$")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    plt.close()

    y1 = ys4[0, :, 0]
    y2 = ys4[0, :, 1]
    y3 = ys4[0, :, 2]

    # y1_model = y_model[:, 0]
    # y2_model = y_model[:, 1]
    # y3_model = y_model[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # Create a 3D subplot

    ax.plot(y1, y2, y3, c="#de7710", linestyle="--", label="Test")
    # ax.plot(y1_model, y2_model, y3_model, c="#de7710", label="Model", alpha=ALPHA)

    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_zlabel("$y_3$")
    ax.set_title("3D Phase Space Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # utils.plot_3_trajectories(ts4, ys4, ys4)
    # utils.plot_3d(ts4, ys4, ys4)

    ##### 5th
    # nl_ds_data_5 = NonlinearDSData(
    #     dynamical_system_number=5,
    #     key=next(key_gen),
    #     t_k=500,
    #     solver=dx.Kvaerno5(),
    #     solver_step_size_controller=dx.PIDController(rtol=1e-6, atol=1e-8),
    #     ts_len=10000,
    # )
    # ts5, ys5 = nl_ds_data_5.generate_data()
    # utils.plot_3_trajectories(ts5, ys5, ys5)
    # utils.plot_3d(ts5, ys5, ys5)

    ##### Lorenz
    # chaotic_data = ChaoticDSData(
    #     key=next(key_gen), t_k=100, ts_len=30000, min_val=-0.8, max_val=1.2
    # )
    # tsc, ysc = chaotic_data.generate_data()
    # utils.plot_3_trajectories(tsc, ysc, ysc)
    # utils.plot_3d(tsc, ysc, ysc)

#### GRAVEYARD ####
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
#
# train_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
#
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1, )).item()
#     img, label = train_data[sample_idx]  # This is both X and Y data
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="rainbow")
#
# plt.show()
#
# #### CUSTOM dataset must implement init, len and getitem
# import os
# import pandas as pd
# from torchvision.io import read_image
#
#
# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, item):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[item, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[item, 1]
#
#         if self.transform:
#             image = self.transform(image)
#
#         if self.target_transform:
#             label = self.target_transform(label)
#
#         return image, label
#
#
# #### DATALOADERS
# from torch.utils.data import DataLoader
#
# train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#
#
# # display image and label
# train_feat, train_label = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_feat.size()}")
# print(f"Labels batch shape: {train_label.size()}")
#
# # view the first example, squeeze removes batch dimensions
# img = train_feat[0].squeeze()
# label = train_label[0]
# plt.imshow(img, cmap="rainbow")
# plt.show()
# print(f"label : {label}")
# x1 = ys4[:, 0]
# y1 = ys4[:, 1]
# z1 = ys4[:, 2]
#
# x2 = ys4[:, 3]
# y2 = ys4[:, 4]
# z2 = ys4[:, 5]
#
# x3 = ys4[:, 6]
# y3 = ys4[:, 7]
# z3 = ys4[:, 8]
#
# # Create three 3D subplots
# fig = plt.figure(figsize=(12, 4))
#
# # Plot the ys4 of the first body
# ax1 = fig.add_subplot(131, projection='3d')
# ax1.plot(x1, y1, z1, label='Body 1')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('Body 1 Position')
#
# # Plot the ys4 of the second body
# ax2 = fig.add_subplot(132, projection='3d')
# ax2.plot(x2, y2, z2, label='Body 2', color='orange')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('Body 2 Position')
#
# # Plot the ys4 of the third body
# ax3 = fig.add_subplot(133, projection='3d')
# ax3.plot(x3, y3, z3, label='Body 3', color='green')
# ax3.set_xlabel('X')
# ax3.set_ylabel('Y')
# ax3.set_zlabel('Z')
# ax3.set_title('Body 3 Position')
#
# plt.tight_layout()
# plt.show()
# # Assuming y is a 9-dimensional vector [x1, y1, z1, x2, y2, z2, x3, y3, z3]
#
# # Extract ys4
# x1, y1, z1, x2, y2, z2, x3, y3, z3 = y
# G = 1.0
# # Calculate distances between bodies
# r12 = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
# r13 = jnp.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)
# r23 = jnp.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)
#
# # Calculate accelerations
# ax1 = G * (x2 - x1) / r12 ** 3 + G * (x3 - x1) / r13 ** 3
# ay1 = G * (y2 - y1) / r12 ** 3 + G * (y3 - y1) / r13 ** 3
# az1 = G * (z2 - z1) / r12 ** 3 + G * (z3 - z1) / r13 ** 3
#
# ax2 = G * (x1 - x2) / r12 ** 3 + G * (x3 - x2) / r23 ** 3
# ay2 = G * (y1 - y2) / r12 ** 3 + G * (y3 - y2) / r23 ** 3
# az2 = G * (z1 - z2) / r12 ** 3 + G * (z3 - z2) / r23 ** 3
#
# ax3 = G * (x1 - x3) / r13 ** 3 + G * (x2 - x3) / r23 ** 3
# ay3 = G * (y1 - y3) / r13 ** 3 + G * (y2 - y3) / r23 ** 3
# az3 = G * (z1 - z3) / r13 ** 3 + G * (z2 - z3) / r23 ** 3
#
# return jnp.array([ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3])
# dy_1dt = 1 + 1*jnp.sin(y[1]-y[0])
# dy_2dt = 1 + 1*jnp.sin(y[0]-y[1]) + 1*y[1]-y[2]
# dy_3dt = 1 + 1*jnp.sin(y[1]-y[2])
