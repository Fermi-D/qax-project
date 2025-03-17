import jax
import jax.numpy as jnp
import numpy as np

# 複素数を表すための「チャネル」を示すインデックス（0: 実部、1: 虚部）
I = jnp.array([0, 1])

def make_complex(x, y=None):
    """
    JAX 互換の複素数テンソルを作成する関数。

    ※ x と y は同じ形状である必要があります。
       また、x が複素数の numpy 配列の場合は、x.real, x.imag を用いて変換します。
    """
    if isinstance(x, np.ndarray):
        # numpy の複素配列の場合は、実部・虚部を抽出して再帰的に呼び出す
        return make_complex(jnp.array(x.real), jnp.array(x.imag))
    if y is None:
        y = jnp.zeros_like(x)
    # PyTorch の torch.cat(unsqueeze(...)) の代わりに jnp.stack を使用
    return jnp.stack([x, y], axis=0)

def numpy(x):
    """
    複素数 JAX テンソルを複素数 numpy 配列に変換します。

    :param x: 複素数テンソル (最初の軸が実部と虚部)
    :returns: 複素数 numpy 配列
    """
    return np.array(real(x)) + 1j * np.array(imag(x))

def real(x):
    """
    複素数テンソルの実部を返します。

    :param x: 複素数テンソル
    :returns: テンソル x の実部 (最初の軸が削除される)
    """
    return x[0, ...]

def imag(x):
    """
    複素数テンソルの虚部を返します。

    :param x: 複素数テンソル
    :returns: テンソル x の虚部 (最初の軸が削除される)
    """
    return x[1, ...]

def scalar_mult(x, y, out=None):
    """
    複素数同士（スカラー・ベクトル・行列）の積を計算します。

    :param x: 複素数テンソル
    :param y: 複素数テンソル
    :param out: （JAX ではサポートしていないため常に None としてください）
    :returns: x と y の積
    """
    if out is not None:
        raise RuntimeError("JAX 版では out 引数による上書きはサポートしていません。")
    r = jnp.multiply(real(x), real(y)) - jnp.multiply(imag(x), imag(y))
    i = jnp.multiply(real(x), imag(y)) + jnp.multiply(imag(x), real(y))
    return make_complex(r, i)

def matmul(x, y):
    """
    複素数行列・行列・ベクトル積を計算します。

    ※ 行列・ベクトル積の場合、第二引数がベクトルである必要があります。

    :param x: 複素数行列（またはバッチ）
    :param y: 複素数行列またはベクトル
    :returns: x と y の積
    """
    r = jnp.matmul(real(x), real(y)) - jnp.matmul(imag(x), imag(y))
    i = jnp.matmul(real(x), imag(y)) + jnp.matmul(imag(x), real(y))
    return make_complex(r, i)

def inner_prod(x, y):
    """
    複素数ベクトルの内積 (⟨x|y⟩) を返します。

    :param x: 複素数ベクトル
    :param y: 複素数ベクトル
    :raises ValueError: 入力の次元がサポート外の場合
    :returns: 内積（複素数テンソル）
    """
    if x.ndim == 2 and y.ndim == 2:
        # 例えば、各ベクトルが shape (2, N) の場合
        r = jnp.dot(real(x), real(y)) + jnp.dot(imag(x), imag(y))
        i = jnp.dot(real(x), imag(y)) - jnp.dot(imag(x), real(y))
        return make_complex(r, i)
    elif x.ndim == 1 and y.ndim == 1:
        # 複素スカラーは shape (2,) で表現
        r = (real(x) * real(y)) + (imag(x) * imag(y))
        i = (real(x) * imag(y)) - (imag(x) * real(y))
        return make_complex(r, i)
    else:
        raise ValueError("Unsupported input shapes!")

def outer_prod(x, y):
    """
    複素数ベクトルの外積 |x⟩⟨y| を返します。

    :param x: 複素数ベクトル（shape: (2, N)）
    :param y: 複素数ベクトル（shape: (2, M)）
    :raises ValueError: 入力が想定する次元でない場合
    :returns: 外積（複素数テンソル、shape: (2, N, M)）
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("An input is not of the right dimension.")
    # torch.ger の代わりに jnp.outer を利用
    r = jnp.outer(real(x), real(y)) + jnp.outer(imag(x), imag(y))
    i = jnp.outer(imag(x), real(y)) - jnp.outer(real(x), imag(y))
    return make_complex(r, i)

def einsum(equation, a, b, real_part=True, imag_part=True):
    """
    複素数に対応した einsum のラッパーです。

    :param equation: アインシュタイン記法の式（例："ij,jk->ik"）
    :param a: 複素数テンソル
    :param b: 複素数テンソル
    :param real_part: 結果の実部を計算するかどうか
    :param imag_part: 結果の虚部を計算するかどうか
    :returns: 結果のテンソル。両方 True なら複素数テンソル、それ以外は実テンソル。
    """
    if real_part:
        r = jnp.einsum(equation, real(a), real(b)) - jnp.einsum(equation, imag(a), imag(b))
    if imag_part:
        i = jnp.einsum(equation, real(a), imag(b)) + jnp.einsum(equation, imag(a), real(b))
    if real_part and imag_part:
        return make_complex(r, i)
    elif real_part:
        return r
    elif imag_part:
        return i
    else:
        return None

def conjugate(x):
    """
    引数の複素共役転置を返します。

    - スカラーまたはベクトルの場合は単に複素共役を取ります。
    - ランク2以上の場合は、まず複素共役を取った後、最初の2軸を入れ替えます。
    """
    if x.ndim < 3:
        return conj(x)
    else:
        # 汎用的に最初の2軸を交換する
        axes = list(range(x.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        return make_complex(jnp.transpose(real(x), axes=axes),
                            -jnp.transpose(imag(x), axes=axes))

def conj(x):
    """
    要素ごとの複素共役を返します。

    :param x: 複素数テンソル
    :returns: 複素共役テンソル
    """
    return make_complex(real(x), -imag(x))

def elementwise_mult(x, y):
    """
    scalar_mult のエイリアスです。
    """
    return scalar_mult(x, y)

def elementwise_division(x, y):
    """
    x を y で要素ごとに割ります。

    :param x: 複素数テンソル
    :param y: 複素数テンソル
    :raises ValueError: 形状が一致しない場合
    :returns: 要素ごとの除算結果（複素数テンソル）
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape!")
    y_star = conj(y)
    sqrd_abs_y = absolute_value(y) ** 2
    return elementwise_mult(x, y_star) / sqrd_abs_y

def absolute_value(x):
    """
    複素数テンソルの各要素について絶対値を返します。

    :param x: 複素数テンソル
    :returns: 実数テンソル（要素ごとの絶対値）
    """
    # 複素数 z の場合、|z| = sqrt( z * z* ) となるので、実部のみ取り出します。
    return jnp.sqrt(real(elementwise_mult(x, conj(x))))

def kronecker_prod(x, y):
    """
    2 つの複素数行列のテンソル積（Kronecker 積）を返します。

    入力はいずれも shape が (2, m, n) の複素数行列である必要があります。

    :param x: 複素数行列（shape: (2, m, n)）
    :param y: 複素数行列（shape: (2, p, q)）
    :raises ValueError: 入力の次元が 3 でない場合
    :returns: Kronecker 積（shape: (2, m*p, n*q)）
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("Inputs must be complex matrices!")
    # 各チャネル毎に kron を計算する（複素乗算の公式を利用）
    r = jnp.kron(real(x), real(y)) - jnp.kron(imag(x), imag(y))
    i = jnp.kron(real(x), imag(y)) + jnp.kron(imag(x), real(y))
    return make_complex(r, i)

def sigmoid(x, y):
    r"""
    複素数 z = x + iy に対する sigmoid 関数を計算します（各要素ごと）。

    sigmoid(z) = exp(z) / (1 + exp(z))

    :param x: z の実部
    :param y: z の虚部
    :returns: 複素 sigmoid (複素数テンソル)
    """
    # JAX の jnp.exp は複素数も扱えますので、dtype のキャストを行いながら計算します
    z = x.astype(jnp.complex64) + 1j * y.astype(jnp.complex64)
    out = jnp.exp(z) / (1 + jnp.exp(z))
    return make_complex(jnp.real(out), jnp.imag(out))

def scalar_divide(x, y):
    """
    x を y で割ります。形状が同じ場合は要素ごとに、y が複素スカラーの場合はスカラー除算を行います。

    :param x: 複素数テンソル（分子）
    :param y: 複素数テンソル（分母）
    :returns: x / y（複素数テンソル）
    """
    return scalar_mult(x, inverse(y))

def inverse(z):
    """
    複素数テンソル z の乗法逆元を返します（要素ごと）。

    :param z: 複素数テンソル
    :returns: 1 / z（複素数テンソル）
    """
    z_star = conj(z)
    denominator = real(scalar_mult(z, z_star))
    return z_star / denominator

def norm_sqr(x):
    """
    複素数テンソル x の二乗ノルム |x|^2 を返します。

    :param x: 複素数テンソル
    :returns: 実数テンソル
    """
    return real(inner_prod(x, x))

def norm(x):
    """
    複素数テンソル x のノルム |x| を返します。

    :param x: 複素数テンソル
    :returns: 実数テンソル
    """
    return jnp.sqrt(norm_sqr(x))
