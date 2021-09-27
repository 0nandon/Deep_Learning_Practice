import numpy as np
from dezero.core import Function, Variable, as_variable, as_array
from dezero import utils, cuda

# sin
class Sin(Function):
    def forward(self, x):
    	y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
	
# cos
class Cos(Function):
    def forward(self, x):
    	y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

# tanh
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

# reshape
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


# transpose
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
        super().__init__()

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        gx = transpose(gy, inv_axes)
        return gx

# sum
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__()

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

# 브로드 캐스트
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


class MatMul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

# MSE
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

# Linear transformation
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

# sigmoid
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (np.exp(-x) + 1)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5 # better expression
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

# exp
class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

# softmax
class Softmax(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        x -= x.max(axis=1, keepdims=True)
        y = xp.exp(x)
        return y / y.sum(axis=1, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

# softmax with cross-entropy loss
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        return -log_p.sum() / np.float32(N)
    

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)

        xp = cuda.get_array_module(t.data)
        """
        [another expression]
        y[np.arange(N), t.data] -= 1
        return y * gy
        """
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


class Relu:
    def forward(self, x):
        y = np.maximum(x, 0)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        return gy * mask


# ============================================== #
# slice function
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
        super().__init__()

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x = self.inputs[0]
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
        super().__init__()

    def forward(self, x):
        xp = cuda.get_array_module(x)
        gx = xp.zeros(self.in_shape)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, slices)


def get_item(x, slices):
    return GetItem(slices)(x)

# ============================================== #

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def sin(x):
    return Sin()(x)
	

def cos(x):
	  return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x, axes=None):
    return Transpose(axes)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def matmul(x, W):
    return MatMul()(x, W)


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def linear(x, W, b=None):
    return Linear()(x, W, b)


def sigmoid(x):
    return Sigmoid()(x)


def exp(x):
    return Exp()(x)


def softmax(x):
    return Softmax()(x)


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def relu(x):
    return Relu()(x)
