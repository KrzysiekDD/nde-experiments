""" Approximation of 5 nonlinear dynamical systems """
import time
import argparse

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx
import equinox.nn as nn
import diffrax as dx
from data_loader import NonlinearDSData
import utils

# Enable float64's, which are turned off by default
from jax import config

config.update("jax_enable_x64", True)


class NODEVectorField(eqx.Module):
    """
    Neural network inside the ODESolve - the so-called vector field
    """

    mlp: nn.MLP

    def __init__(self, system_dims, network_width, network_depth, *, key, **kwargs):
        """
        Defines the vector field structure. Here it's just a Multi-Layer Perceptron

        As a reminder, models inheriting from eqx.Module  overwrite __call__,
        not forward like in the torch nn.Module

        :param system_dims:
        :param network_width:
        :param network_depth:
        :param key:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.mlp = nn.MLP(
            in_size=system_dims,
            out_size=system_dims,
            width_size=network_width,
            depth=network_depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, x, args):
        """
        t is not used explicitly here, but it is used inside diffeqsolve implicitly
        :param t:
        :param x:
        :param args:
        :return:
        """
        return self.mlp(x)


class NODE(eqx.Module):
    """
    Whole model. For this use-case, we don't insert NODE into another network, as
    we are modelling time-series data. The neural ODE can do that by itself.
    """

    # Instance attributes of eqx.Module need to have type annotations
    vector_field: NODEVectorField
    backprop_method: dx.AbstractAdjoint
    solver: dx.AbstractSolver
    rtol: float
    atol: float

    def __init__(
        self,
        system_dims,
        network_width,
        network_depth,
        backprop_method=dx.RecursiveCheckpointAdjoint(),
        solver=dx.Tsit5(),
        rtol=1e-6,
        atol=1e-6,
        # solver=dx.Kvaerno5(),
        *,
        key,
        **kwargs,
    ):
        """
        Initialize the vector field

        :param system_dims:
        :param network_width:
        :param network_depth:
        :param key:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.vector_field = NODEVectorField(
            system_dims, network_width, network_depth, key=key
        )
        self.backprop_method = backprop_method
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def __call__(self, ts, y_0):
        """
        Forward prop is just integration of the ODE (specified by a NN)
        :param ts:
        :param y_0:
        :return:
        """
        system_solution = dx.diffeqsolve(
            dx.ODETerm(self.vector_field),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y_0,
            # max_steps=8192,
            stepsize_controller=dx.PIDController(rtol=self.rtol, atol=self.atol),
            saveat=dx.SaveAt(ts=ts),
            adjoint=self.backprop_method,
        )

        ys = system_solution.ys
        return ys


if __name__ == "__main__":
    """Argparse, Training loop, results"""
    parser = argparse.ArgumentParser(
        description=" Nonlinear dynamics training and evaluation script "
    )
    # Dataset related variables
    parser.add_argument("-ds", "--dataset_size", type=int, default=256)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-it", "--integration_time", type=int, default=10)
    parser.add_argument("-is", "--integration_steps", type=int, default=1000)
    parser.add_argument("--min_val", type=float, default=-0.6)
    parser.add_argument("--max_val", type=float, default=1.0)
    # Model is trained first only on a fraction of the full sequence, thus
    # more than one learning strategy can be specified
    parser.add_argument(
        "-lr", "--learning_rates", nargs="+", type=float, default=[3e-3, 3e-3]
    )
    parser.add_argument(
        "-sp", "--steps_per_strategy", nargs="+", type=int, default=[500, 500]
    )
    parser.add_argument(
        "-lp", "--len_per_strategy", nargs="+", type=float, default=[0.1, 1.0]
    )
    # Script can use either a predefined systems or run a custom system
    parser.add_argument(
        "-f", "--function_number", type=int, default=1, choices=[1, 2, 3, 4, 5]
    )
    parser.add_argument("-c", "--custom_dynamics", action="store_true")
    # Hyperparameters of the vector field
    parser.add_argument("--num_dims", type=int, default=2)
    parser.add_argument("--nn_width", type=int, default=64)
    parser.add_argument("--nn_depth", type=int, default=2)
    parser.add_argument("-rt", "--nn_rtol", type=float, default=1e-3)
    parser.add_argument("-at", "--nn_atol", type=float, default=1e-6)
    parser.add_argument(
        "-bm",
        "--backprop_method",
        type=int,
        default=1,
        choices=[1, 2],
    )
    # Miscellaneous arguments
    parser.add_argument("--random_seed", type=int, default=2112)
    parser.add_argument(
        "-p", "--plot_results", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()

    # num_dims = args.num_dims
    # nn_width = args.nn_width
    # nn_depth = args.nn_depth
    # random_seed = args.random_seed
    #
    # key_gen = key_generator(random_seed)
    # vf = NODEVectorField(num_dims, nn_width, nn_depth, key=next(key_gen))
    # node = NODE(num_dims, nn_width, nn_depth, key=next(key_gen))

    # data = NonlinearDSData(key=next(key_gen))

    # ts, ys = data.generate_data()

    def main(
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        integration_time=args.integration_time,
        integration_steps=args.integration_steps,
        min_val=args.min_val,
        max_val=args.max_val,
        learning_rates=args.learning_rates,
        steps_per_strategy=args.steps_per_strategy,
        len_per_strategy=args.len_per_strategy,
        function_number=args.function_number,
        custom_dynamics=args.custom_dynamics,
        num_dims=args.num_dims,
        nn_width=args.nn_width,
        nn_depth=args.nn_depth,
        nn_rtol=args.nn_rtol,
        nn_atol=args.nn_atol,
        backprop_method=args.backprop_method,
        random_seed=args.random_seed,
        plot_results=args.plot_results,
        log_every=args.log_every,
    ):
        """
        Entrypoint of the script. Defines training loop and plots results
        :param dataset_size:
        :param batch_size:
        :param integration_time:
        :param integration_steps:
        :param min_val:
        :param max_val:
        :param learning_rates:
        :param steps_per_strategy:
        :param len_per_strategy:
        :param function_number:
        :param custom_dynamics:
        :param num_dims:
        :param nn_width:
        :param nn_depth:
        :param nn_rtol:
        :param nn_atol:
        :param backprop_method:
        :param random_seed:
        :param plot_results:
        :param log_every:
        :return:
        """
        key_gen = utils.key_generator(random_seed)

        if custom_dynamics:
            # TODO add parsing of a custom system, it should be in utils
            custom_system = "tbd"
        else:
            custom_system = None

        data = NonlinearDSData(
            dataset_size=dataset_size,
            batch_size=batch_size,
            dynamical_system_number=function_number,
            custom_dynamical_system=custom_system,
            t_k=integration_time,
            ts_len=integration_steps,
            min_val=min_val,
            max_val=max_val,
            key=next(key_gen),
        )
        ts, ys = data.generate_data()

        if (function_number == 4 or function_number == 5) and not custom_dynamics:
            num_dims = 3

        _, sequence_len, _num_dims = ys.shape

        assert _num_dims == num_dims

        if backprop_method == 1:
            adjoint_method = dx.RecursiveCheckpointAdjoint()
        elif backprop_method == 2:
            adjoint_method = dx.BacksolveAdjoint()

        node_model = NODE(
            num_dims,
            nn_width,
            nn_depth,
            atol=nn_atol,
            rtol=nn_rtol,
            backprop_method=adjoint_method,
            key=next(key_gen),
        )

        # 'filter' in the decorator means the model will be broken down into two parts:
        # 1. Only differentiable parameters
        # 2. Static parameters not meant to be differentiated
        @eqx.filter_value_and_grad
        def loss_and_gradients(model, t_i, y_i):
            """
            Calculates the loss value, as well as array of gradients w.r.t
             model parameters
            :param model:
            :param t_i:
            :param y_i:
            :return:
            """
            # vmap magic
            # jk, this vectorizes over the time dimension and the sequence dimension
            # of the solution, rather trivial compared to other vmap magics
            y_pred = jax.vmap(model, in_axes=(None, 0))(t_i, y_i[:, 0])

            # MSE loss
            return jnp.mean((y_i - y_pred) ** 2)

        @eqx.filter_jit
        def training_step(t_i, y_i, model, optimizer_state):
            """

            :param t_i:
            :param y_i:
            :param model:
            :param optimizer_state:
            :return:
            """
            loss, grads = loss_and_gradients(model, t_i, y_i)
            updates, optimizer_state = optimizer.update(grads, optimizer_state)
            model = eqx.apply_updates(model, updates)

            return loss, model, optimizer_state

        train_start = time.time()
        for learning_rate, num_steps, solution_len in zip(
            learning_rates, steps_per_strategy, len_per_strategy
        ):
            optimizer = optax.adabelief(learning_rate)
            optimizer_state = optimizer.init(
                eqx.filter(node_model, eqx.is_inexact_array)
            )

            ts_train = ts[: int(solution_len * sequence_len)]
            ys_train = ys[:, : int(solution_len * sequence_len)]

            for step, (y_i,) in zip(
                range(num_steps),
                data.dataloader((ys_train,), batch_size, key=next(key_gen)),
            ):
                start = time.time()
                loss, node_model, optimizer_state = training_step(
                    ts_train, y_i, node_model, optimizer_state
                )
                end = time.time()

                if (step % log_every) == 0 or step == num_steps - 1:
                    print(
                        f"{step}:    training_loss: {loss}, computation_time_per_step: {end - start}"
                    )

            if plot_results:
                y_model = node_model(ts, ys[0, 0])
                if plot_results:
                    if num_dims == 2:
                        utils.plot_2_trajectories(ts, ys, y_model)
                        utils.plot_2d(ts, ys, y_model)
                    elif num_dims == 3:
                        utils.plot_3_trajectories(ts, ys, y_model)
                        utils.plot_3d(ts, ys, y_model)

        train_end = time.time()
        training_time = train_end - train_start  # in seconds
        h = int(training_time // 3600)
        m = int((training_time % 3600) // 60)
        s = int(training_time % 60)
        print(f"Training time: {h} hours {m} minutes {s} seconds")

        if plot_results:
            y_model = node_model(ts, ys[0, 0])
            if plot_results:
                if num_dims == 2:
                    print(
                        f"Normalized MSE: {utils.normalized_mse(ys, y_model, num_dims)[1]}"
                    )
                    utils.plot_2_trajectories(ts, ys, y_model)
                    utils.plot_2d(ts, ys, y_model)
                elif num_dims == 3:
                    print(
                        f"Normalized MSE: {utils.normalized_mse(ys, y_model, num_dims)[1]}"
                    )
                    utils.plot_3_trajectories(ts, ys, y_model)
                    utils.plot_3d(ts, ys, y_model)

        return ts, ys, node_model

        # # Plotting y1 and y2 against time
        # plt.figure(figsize=(12, 6))
        #
        # # Subplot for time series
        # plt.subplot(1, 2, 1)
        # plt.plot(ts, ys[0, :, 0], c="dodgerblue", label=r"Real $x_1$")
        # plt.plot(ts, ys[0, :, 1], c="g", label=r"Real $x_1$")
        # model_y = node_model(ts, ys[0, 0])
        # plt.plot(ts, model_y[:, 0], c="crimson", label="Model y1")
        # plt.plot(ts, model_y[:, 1], c="y", label="Model y2")
        # plt.xlabel("Time")
        # plt.ylabel("States")
        # plt.legend()
        # plt.title("Time Series")
        #
        # # Subplot for phase space trajectory
        # plt.subplot(1, 2, 2)
        # plt.plot(ys[0, :, 0], ys[0, :, 1], c="dodgerblue", label="Real Trajectory")
        # plt.plot(
        #     model_y[:, 0], model_y[:, 1], c="crimson", label="Model Trajectory"
        # )
        # plt.xlabel("$y_1(t)$")
        # plt.ylabel("$y_2(t)$")
        # plt.title("Phase Space Trajectory")
        # plt.legend()
        #
        # plt.tight_layout()
        # # plt.savefig("neural_ode_combined.png")
        # plt.show()

        #     plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        #     plt.plot(ts, ys[0, :, 1], c="g", label="Real 2")
        #     model_y = node_model(ts, ys[0, 0])
        #     plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        #     plt.plot(ts, model_y[:, 1], c="y", label="Model 2")
        #     plt.legend()
        #     plt.tight_layout()
        #     # plt.savefig("neural_ode.png")
        #     plt.show()

    ts, ys, model = main()
