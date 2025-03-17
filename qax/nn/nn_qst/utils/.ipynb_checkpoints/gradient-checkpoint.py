import jax.numpy as jnp
import numpy as np

def vector_to_grads(vec, parameters):
    r"""フラットなベクトル `vec` から、各パラメータに対応する勾配のリストを作成します。

    :param vec: モデルのパラメータに対応するフラットな勾配ベクトル。
                型は jax.numpy.ndarray と想定します。
    :param parameters: モデルのパラメータのイテレータ（またはリスト）。
                       各要素は jax.numpy.ndarray であり、形状はパラメータの形状を表すものとします。
    :returns: 各パラメータに対応する勾配のリスト。  
              リストの i 番目の要素は、`parameters[i]` と同じ形状の勾配となります。
    """
    if not isinstance(vec, jnp.ndarray):
        raise TypeError("expected jax.numpy.ndarray, but got: {}".format(type(vec)))

    pointer = 0
    grads = []
    for param in parameters:
        # パラメータの総要素数を計算
        num_param = np.prod(param.shape)
        # vec から対応する部分を抜き出し、パラメータと同じ形状にリシェイプする
        grad = jnp.reshape(vec[pointer:pointer + num_param], param.shape)
        grads.append(grad)
        pointer += num_param

    return grads
