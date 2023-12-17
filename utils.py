""" Utility functions used in multiple places """
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# to make the models predictions transparent on plots
ALPHA: float = 0.5


def key_generator(seed: int, num_keys: int = 10) -> iter:
    """
    Generator for less explicit handling of the PRNG state. Assign to a variable once,
    then simply call next() on that variable to obtain a new key.
    :param seed:
    :param num_keys:
    :return:
    """
    key = jrandom.PRNGKey(seed)
    keys_split = jrandom.split(key, num_keys)

    keys_list = [keys_split[i] for i in range(num_keys)]
    return iter(keys_list)


def key_split_generator(key, num_keys: int) -> iter:
    """
    Similar to key_generator, but takes an existing key instead of seed as input.
    :param key:
    :param num_keys:
    :return:
    """
    keys_split = jrandom.split(key, num_keys)

    keys_list = [keys_split[i] for i in range(num_keys)]
    return iter(keys_list)


def parse_custom_system():
    """
    Parse custom system given by a list of strings. i-th element of the list is
    the equation for the state dx_i(t)/dt
    """
    ...


def plot_2_trajectories(ts, ys, y_model, save=False):
    """
    Plot state trajectories for 2d system
    :param ts:
    :param ys:
    :param y_model:
    :param save:
    :return:
    """
    plt.plot(ts, ys[0, :, 0], c="#e31212", linestyle="--", label=r"Test $y_1$")
    plt.plot(ts, ys[0, :, 1], c="#09d635", linestyle="--", label=r"Test  $y_2$")

    plt.plot(ts, y_model[:, 0], c="#e31212", label=r"Model $y_1$", alpha=ALPHA)
    plt.plot(ts, y_model[:, 1], c="#09d635", label=r"Model $y_2$", alpha=ALPHA)

    plt.legend()
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig(f"neural_ode{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")

    plt.show()
    plt.close()


def plot_3_trajectories(ts, ys, y_model, save=False):
    """
    Plot state trajectories for 2d system
    :param ts:
    :param ys:
    :param y_model:
    :param save:
    :return:
    """
    plt.plot(ts, ys[0, :, 0], c="#e31212", linestyle="--", label=r"Test $y_1$")
    plt.plot(ts, ys[0, :, 1], c="#09d635", linestyle="--", label=r"Test  $y_2$")
    plt.plot(ts, ys[0, :, 2], c="#0e0eb0", linestyle="--", label=r"Test  $y_3$")

    plt.plot(ts, y_model[:, 0], c="#e31212", label=r"Model $y_1$", alpha=ALPHA)
    plt.plot(ts, y_model[:, 1], c="#09d635", label=r"Model $y_2$", alpha=ALPHA)
    plt.plot(ts, y_model[:, 2], c="#0e0eb0", label=r"Model  $y_3$", alpha=ALPHA)

    plt.legend()
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig(f"neural_ode{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")

    plt.show()
    plt.close()


def plot_2d(ts, ys, y_model, save=False):
    """
    Phase portrait for 2d system
    :param ts:
    :param ys:
    :param y_model:
    :return:
    """
    plt.plot(ys[0, :, 0], ys[0, :, 1], c="#de7710", linestyle="--", label="Test")
    plt.plot(y_model[:, 0], y_model[:, 1], c="#de7710", label="Model", alpha=ALPHA)

    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(
            f"neural_ode_portrait{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )

    plt.show()


def plot_3d(ts, ys, y_model, save=False):
    """
    Phase portrait for 3d system
    :param ts:
    :param ys:
    :param y_model:
    :return:
    """
    y1 = ys[0, :, 0]
    y2 = ys[0, :, 1]
    y3 = ys[0, :, 2]

    y1_model = y_model[:, 0]
    y2_model = y_model[:, 1]
    y3_model = y_model[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # Create a 3D subplot

    ax.plot(y1, y2, y3, c="#de7710", linestyle="--", label="Test")
    ax.plot(y1_model, y2_model, y3_model, c="#de7710", label="Model", alpha=ALPHA)

    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_zlabel("$y_3$")
    ax.legend()

    if save:
        plt.savefig(
            f"neural_ode_portrait{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        )

    plt.tight_layout()
    plt.show()


def normalized_mse(ys, y_model, num_dims):
    mse_list = []

    for i in range(num_dims):
        y_true = ys[0, :, i]
        y_pred = y_model[:, i]

        mse = jnp.mean((y_true - y_pred) ** 2)
        var = jnp.var(y_true)
        percentage_error = mse / var * 100
        mse_list.append(percentage_error)

    normalized_mse_value = jnp.mean(jnp.array(mse_list))
    return mse_list, normalized_mse_value
