# nde-experiments
Experiments with neural ordinary differential equations conducted for engineering thesis.


## Environment setup
Use a venv with jax installed, then install required libraries inside of it,
or create a new venv and install everything required there. On Ubuntu 22.04 it looks something like this:


```
python3 -m venv venv && source ./venv/bin/activate && pip3 install -r requirements.txt
```


## Reproducing experiments
When JAX and other modules are installed, you can start training your neural ODE's for dynamical system approximation. For a nonlinear system:

```
python3 nonlinear_system_approx.py -ds 512 -it 10 -is 1000 --min_val -0.6 --max_val 1. -f 1  --nn_width 128 --nn_depth 4 -bm 1
```


For the lorenz system (This fails 99.99% of the time):
```
python3 chaotic_system_approx.py -ds 4096 -bs 64 -it 100 -is 30000 --min_val -0.8 --max_val 1.2 -lr 0.001 0.001 0.0003 -sp 500 500 500 -lp 0.15 0.5 1.0 --nn_width 64 --nn_depth 2 -at 0.000001 -rt 0.00000001 -bm 1
```
