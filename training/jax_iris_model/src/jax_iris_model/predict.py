#!/usr/bin/env python3

import pickle

import jax
import jax.numpy as jnp

from jax_iris_model.data import iris, X
from jax_iris_model.model import IrisNN

# Load the saved JAX model parameters
with open("jax_model_iris.pkl", "rb") as f:
    params = pickle.load(f)

# Initialize and predict
model = IrisNN()
input_data = X[:5]  # Taking first 5 samples for prediction
predictions = model.apply({"params": params}, jnp.array(input_data))

# Convert logits to probabilities
probs = jax.nn.softmax(predictions)

# Get top 3 predictions
top_k_values, top_k_indices = jax.lax.top_k(probs, k=3)

for i in range(len(input_data)):
    print(f"Sample {i+1} predictions:")
    for j in range(3):
        class_index = top_k_indices[i, j]
        probability = top_k_values[i, j]
        class_name = iris.target_names[class_index]
        print(f"\t{class_name} ({probability*100:.2f}%)")
