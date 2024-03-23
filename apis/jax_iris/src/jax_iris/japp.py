#!/usr/bin/env python3

from fastapi import FastAPI
import jax
import jax.numpy as jnp
import pickle
from jax_iris_model.model import IrisNN
from jax_iris_model.data import iris, X_test


app = FastAPI()

# Load the JAX model parameters
with open("jax_model_iris.pkl", "rb") as f:
    jax_params = pickle.load(f)


@app.get("/predict/{index}")
async def predict(index: int):
    # Assuming index is used to select a sample from X_test for simplicity
    input_data = jnp.array(X_test[index : index + 1], jnp.float32)
    preds = IrisNN().apply({"params": jax_params}, input_data)
    probs = jax.nn.softmax(preds)
    top_k_values, top_k_indices = jax.lax.top_k(probs, k=3)

    results = []
    # Format and collect the results
    for i in range(len(input_data)):
        for j in range(3):
            class_index = top_k_indices[i, j]
            probability = top_k_values[i, j]
            class_name = iris.target_names[class_index]
            results.append(dict(name=class_name, score=f"{probability*100:.2f}%"))

    return {"prediction": results}
