import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from jax.random import PRNGKey, normal, bernoulli

class RBM(nn.Module):
    num_visible: int  # 可視層のユニット数
    num_hidden: int   # 隠れ層のユニット数

    def setup(self):
        # 重みとバイアスを初期化
        self.W = self.param('W', normal, (self.num_visible, self.num_hidden))  # 重み行列
        self.b = self.param('b', lambda rng: jnp.zeros(self.num_visible))      # 可視層のバイアス
        self.c = self.param('c', lambda rng: jnp.zeros(self.num_hidden))      # 隠れ層のバイアス

    def free_energy(self, v):
        """
        自由エネルギーの計算
        """
        vbias_term = jnp.dot(v, self.b)
        hidden_term = jnp.sum(jnp.log1p(jnp.exp(jnp.dot(v, self.W) + self.c)), axis=-1)
        return -vbias_term - hidden_term

    def sample_hidden(self, v, rng):
        """
        可視層から隠れ層をサンプリング
        """
        activations = jnp.dot(v, self.W) + self.c
        probs = jax.nn.sigmoid(activations)
        return bernoulli(rng, probs), probs

    def sample_visible(self, h, rng):
        """
        隠れ層から可視層をサンプリング
        """
        activations = jnp.dot(h, self.W.T) + self.b
        probs = jax.nn.sigmoid(activations)
        return bernoulli(rng, probs), probs

    def