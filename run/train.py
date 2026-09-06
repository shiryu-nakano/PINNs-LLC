"""Train the spatially-varying-diffusion PINN (c_net + D_net).

This is the scripted equivalent of PINNs_SVD.ipynb's training section, with
paths (data / checkpoints) made explicit and configurable instead of
downloading to and saving in the current working directory.

Usage:
    python run/train.py --case 1
    python run/train.py --config run/configs/example.yaml
    python run/train.py --case 2 --n-iter 1000 --ckpt-dir /path/to/ckpts

Paths default to the shared data server (see src/config.py); override with
--data-dir/--ckpt-dir or a --config file if running somewhere else.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint import build_ckpt_steps, save_ckpt, save_context, save_llc_eval_set  # noqa: E402
from src.config import TrainConfig, build_config  # noqa: E402
from src.data import download_data, load_dataset  # noqa: E402
from src.model import NeuralNet, mean_squared_error, relative_error  # noqa: E402


def make_train_step(c_net, D_net, optimizer, scale_c: float, scale_d, known_diffusion: bool):
    @tf.function
    def compute_gradients(x_data_tf, y_data_tf, t_data_tf, c_data_tf, d_data_tf,
                           x_eqns_tf, y_eqns_tf, t_eqns_tf, dt):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_eqns_tf, y_eqns_tf])
            dt = tf.cast(dt, tf.float32)

            c_data_pu_1 = c_net(x_data_tf, y_data_tf, t_data_tf)[0]
            c_eqns_pu_1 = c_net(x_eqns_tf, y_eqns_tf, t_eqns_tf)[0]
            c_eqns_pu_2 = c_net(x_eqns_tf, y_eqns_tf, t_eqns_tf + dt)[0]
            D_pred_ = D_net(x_eqns_tf, y_eqns_tf)[0]

            c_x = tape.gradient([c_eqns_pu_2], [x_eqns_tf])[0]
            c_y = tape.gradient([c_eqns_pu_2], [y_eqns_tf])[0]
            D_x = tape.gradient([D_pred_], [x_eqns_tf])[0]
            D_y = tape.gradient([D_pred_], [y_eqns_tf])[0]
            c_xx = tape.gradient([c_x], [x_eqns_tf])[0]
            c_yy = tape.gradient([c_y], [y_eqns_tf])[0]

            f_c = D_pred_ * (c_xx + c_yy) + D_x * c_x + D_y * c_y
            c_eqns_pi_1 = c_eqns_pu_2 - dt * f_c

            loss_data = mean_squared_error(c_data_pu_1 / scale_c, c_data_tf / scale_c)
            loss_consistency = mean_squared_error(c_eqns_pu_1 / scale_c, c_eqns_pi_1 / scale_c)

            if known_diffusion:
                D_pred_data = D_net(x_data_tf, y_data_tf)[0]
                loss_diff = mean_squared_error(D_pred_data / scale_d, d_data_tf / scale_d)
            else:
                loss_diff = tf.constant(0.0)

            total_loss = loss_data + loss_consistency

        variables = c_net.get_trainable_variables() + D_net.get_trainable_variables()
        gradients = tape.gradient(total_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss_data, loss_diff, loss_consistency

    return compute_gradients


def train(cfg: TrainConfig) -> None:
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    data_path = download_data(cfg.data_dir, cfg.case)
    print(f"[data] using {data_path}")

    ds = load_dataset(
        data_path,
        n_eqns=cfg.n_eqns,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
        known_diffusion=cfg.known_diffusion,
    )

    ckpt_dir = cfg.run_ckpt_dir()
    print(f"[ckpt] writing to {ckpt_dir}")

    c_net = NeuralNet(ds.x_data, ds.y_data, ds.t_data, layers=cfg.layers)
    D_net = NeuralNet(ds.x_data, ds.y_data, layers=cfg.layers_d)

    scale_c = float(np.std(ds.c_data))
    scale_d = float(np.std(ds.d_data)) if cfg.known_diffusion else None

    save_context(
        ckpt_dir,
        C_star=ds.C_star, Diff_star=ds.Diff_star, X_star=ds.X_star, t_star=ds.t_star,
        idx_test=ds.idx_test, idx_x=ds.idx_x, dt=ds.dt,
        scale_c=scale_c, scale_d=scale_d,
        c_net=c_net, D_net=D_net, layers=cfg.layers, layers_d=cfg.layers_d,
    )
    save_llc_eval_set(
        ckpt_dir,
        x_data=ds.x_data, y_data=ds.y_data, t_data=ds.t_data,
        c_data=ds.c_data, d_data=ds.d_data,
        dt=ds.dt, scale_c=scale_c, scale_d=scale_d,
        n_llc_data=cfg.n_llc_data, n_llc_eqns=cfg.n_llc_eqns,
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((ds.x_data, ds.y_data, ds.t_data, ds.c_data, ds.d_data))
        .shuffle(buffer_size=10)
        .repeat()
        .batch(cfg.batch_size, drop_remainder=True)
    )
    train_iterator = iter(train_dataset)

    eqns_dataset = (
        tf.data.Dataset.from_tensor_slices((ds.x_eqns, ds.y_eqns, ds.t_eqns))
        .shuffle(buffer_size=10)
        .repeat()
        .batch(cfg.batch_size, drop_remainder=True)
    )
    eqns_iterator = iter(eqns_dataset)

    optimizer = tf.keras.optimizers.Adam()
    compute_gradients = make_train_step(c_net, D_net, optimizer, scale_c, scale_d, cfg.known_diffusion)

    ckpt_steps = build_ckpt_steps(cfg.n_iter, cfg.ckpt_every, cfg.ckpt_extra_steps)
    if 0 in ckpt_steps:
        save_ckpt(ckpt_dir, 0, c_net, D_net)

    start_time = time.time()
    running_time = 0.0
    it = 0

    while it < cfg.n_iter:
        lr = cfg.eta_min + 0.5 * (cfg.eta_max - cfg.eta_min) * (1 + math.cos(math.pi * it / cfg.n_iter))
        optimizer.learning_rate.assign(lr)

        x_data_tf, y_data_tf, t_data_tf, c_data_tf, d_data_tf = next(train_iterator)
        x_eqns_tf, y_eqns_tf, t_eqns_tf = next(eqns_iterator)

        loss_data, loss_diff, loss_consistency = compute_gradients(
            x_data_tf, y_data_tf, t_data_tf, c_data_tf, d_data_tf,
            x_eqns_tf, y_eqns_tf, t_eqns_tf, ds.dt,
        )

        it += 1

        if it % cfg.log_every == 0:
            elapsed = time.time() - start_time
            running_time += elapsed / 3600.0
            print(
                f"It: {it}, Loss_data: {loss_data.numpy():.3e}, Loss_f: {loss_consistency.numpy():.3e}, "
                f"Loss_diff: {loss_diff.numpy():.3e}, Time: {elapsed:.2f}s, "
                f"Running Time: {running_time:.2f}h, Learning Rate: {lr:.1e}"
            )
            start_time = time.time()

        if it % cfg.eval_every == 0:
            c_val = c_net(ds.x_test, ds.y_test, ds.t_test)[0]
            msg = f"  eval @ it={it}: rel_err_c={relative_error(c_val, ds.c_test).numpy():.3e}"
            if cfg.known_diffusion:
                D_val = D_net(ds.x_test, ds.y_test)[0]
                msg += f", rel_err_d={relative_error(D_val, ds.d_test).numpy():.3e}"
            print(msg)

        if it in ckpt_steps:
            save_ckpt(ckpt_dir, it, c_net, D_net)


def main(argv=None) -> None:
    cfg = build_config(argv)
    train(cfg)


if __name__ == "__main__":
    main()
