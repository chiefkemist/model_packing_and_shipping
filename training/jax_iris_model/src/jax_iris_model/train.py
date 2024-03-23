#!/usr/bin/env python
# coding: utf-8

from jax_iris_model.data import X_train, X_test, y_train, y_test


import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state


from jax_iris_model.model import IrisNN


def create_train_state(rng_key, learning_rate, input_shape):
    model = IrisNN()
    params = model.init(rng_key, jnp.ones(input_shape))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, X, y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, X)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


from tqdm import tqdm

# Training loop
rng_key = random.PRNGKey(0)
state = create_train_state(rng_key, 0.001, (1, 4))
for epoch in tqdm(range(100)):
    state = train_step(state, jnp.array(X_train), jnp.array(y_train))

import pickle

with open("jax_model_iris.pkl", "wb") as file:
    pickle.dump(state.params, file)
