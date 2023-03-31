from .activation import Activation
from .binary import Binary
from .common import OpFactory
from .common import Operator  # noqa: F401
from .concat import Concat
from .conv import Conv
from .conv import TransposeConv
from .fullyconnected import FullyConnected
from .padding import Padding
from .pooling import Pooling
from .quantize import Quantize
from .reduce import Reduce
from .reshape import Reshape
from .resize import Resize
from .rsqrt import Rsqrt
from .slice import Slice
from .softmax import Softmax
from .split import Split
from .squared_difference import SquaredDifference
from .transpose import Transpose
from .unary import Unary


OpFactory.register(Activation)
OpFactory.register(Binary)
OpFactory.register(Concat)
OpFactory.register(Conv)
OpFactory.register(FullyConnected)
OpFactory.register(Padding)
OpFactory.register(Pooling)
OpFactory.register(Quantize)
OpFactory.register(Reduce)
OpFactory.register(Reshape)
OpFactory.register(Resize)
OpFactory.register(Rsqrt)
OpFactory.register(Slice)
OpFactory.register(Softmax)
OpFactory.register(Split)
OpFactory.register(SquaredDifference)
OpFactory.register(Transpose)
OpFactory.register(TransposeConv)
OpFactory.register(Unary)
