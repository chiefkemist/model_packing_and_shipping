#!/usr/bin/env python3

from fastapi import FastAPI
import jax.numpy as jnp
import pickle
from jax_iris_model.model import IrisNN
from jax_iris_model.jax_iris import X_test

app = FastAPI()

# Load the JAX model parameters
with open("jax_model_iris.pkl", "rb") as f:
    jax_params = pickle.load(f)


@app.get("/predict/{index}")
async def predict(index: int):
    # Assuming index is used to select a sample from X_test for simplicity
    input_data = jnp.array(X_test[index : index + 1], jnp.float32)
    preds = IrisNN().apply({"params": jax_params}, input_data)
    return {"prediction": preds.tolist()}
