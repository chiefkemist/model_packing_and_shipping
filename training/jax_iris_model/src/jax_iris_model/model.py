#!/usr/bin/env python3


from flax import linen as nn


class IrisNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        return x
