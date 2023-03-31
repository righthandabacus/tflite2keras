import copy
import logging

import numpy as np
from tflite import TensorType

from . import mapping
from .common import T2KBase

logger = logging.getLogger('t2k.tensor')


class Tensor(T2KBase):
    """Holds one tensor from TFLite file. A tensor is identified by an integer
    index in the graph.
    """

    def __init__(self, model, graph, index: int, layout=None, is_bias=False):
        super().__init__(model, graph, index)
        self.tflite = graph.Tensors(index) if index >= 0 else None
        self.shape = []
        self.dtype: TensorType = None
        self.data: np.ndarray = None

        # the defaults of quantization parameter
        self.scale = 1.0
        self.zero_point = 127

        self.layout = layout
        self.producers = []  # op that produce this tensor (value info only)
        self.consumers = []  # op that uses this tensor (value info only)

        # we only accept INT32 as quantized tensor type for bias
        self.is_bias = is_bias

        self.setInited()

    @property
    def isInitializer(self):
        """initializer: the weights and bias which self.data holds a numpy
        array; otherwise is just a KerasTensor or placeholder to denote the
        output shape of a layer
        """
        return self.data is not None

    def addProducer(self, op):
        assert len(self.producers) == 0
        self.producers.append(op)
        assert len(self.producers) == 1

    def removeProducer(self, op):
        assert len(self.producers) == 1
        assert self.producers[0] == op
        self.producers.remove(op)

    def replaceProducer(self, original, new):
        assert len(self.producers) == 1
        assert self.producers[0] == original
        self.producers[0] = new

    def addConsumer(self, op):
        assert op not in self.consumers
        self.consumers.append(op)

    def removeConsumer(self, op):
        assert op in self.consumers
        self.consumers.remove(op)

    def replaceConsumer(self, original, new):
        assert original in self.consumers
        for i, op in enumerate(self.consumers):
            if op is original:
                self.consumers[i] = new
                return

    @property
    def quantized(self):
        is_quant_dtype = ((self.dtype == TensorType.UINT8) or
                          ((self.dtype == TensorType.INT32) and self.is_bias))
        if self.tflite is None:
            return is_quant_dtype
        else:
            has_quant = self.tflite.Quantization() is not None
            return is_quant_dtype and has_quant

    def dequantize(self):
        if not self.quantized:
            return
        logger.debug("Dequantizing %s", self.shorty)
        if self.isInitializer:
            int32 = self.data.astype('int32')
            # both operands in subtract and multiply are numpy arrays, assume broadcast works
            shiftted = np.subtract(int32,
                                   np.expand_dims(self.zero_point,
                                                  axis=list(range(self.zero_point.ndim, int32.ndim))
                                                  )
                                   )
            fp32 = np.multiply(shiftted.astype('float32'),
                               np.expand_dims(self.scale,
                                              axis=list(range(self.scale.ndim, shiftted.ndim))
                                              )
                               )
            self.data = fp32
        self.dtype = TensorType.FLOAT32

    @property
    def isScalar(self):
        return (self.layout is None) and (len(self.shape) == 0) and (len(self.data) == 1)

    def asDtype(self, dtype: str):
        """remember the type in self.dtype and reset numpy type for self.data
        """
        self.dtype = mapping.DTYPE_NAME2TFLITE[dtype]
        if self.isInitializer:
            self.data = self.data.astype(dtype)

    def parse(self):
        if self.status.parsed:
            return
        tensor = self.tflite
        self.name = tensor.Name().decode('utf-8')
        logger.debug("Parsing %s...", self.name)
        self.shape = [int(i) for i in tensor.ShapeAsNumpy()]

        assert tensor.Type() in mapping.DTYPE_TFLITE2NAME
        self.dtype = tensor.Type()
        dtype = mapping.DTYPE_TFLITE2NAME[self.dtype]
        self.data = TensorFactory.getData(self.model, self.graph, self.index, dtype)

        if self.quantized:
            quant = tensor.Quantization()
            assert quant.ScaleAsNumpy().size > 0, "Scale cannot be empty"
            assert quant.ZeroPointAsNumpy().size > 0, "Zero-point cannot be empty"
            self.scale: np.ndarray = quant.ScaleAsNumpy()
            self.zero_point: np.ndarray = quant.ZeroPointAsNumpy()

        self.setParsed()

    def transform(self):
        """Apply layout transformation, e.g. NHWC -> NCHW"""
        assert self.status.parsed
        assert self.layout is not None
        if self.isInitializer:
            data = self.data.reshape(self.shape)
            self.shape = self.layout.transform(self.shape)
            self.data = data.transpose(self.layout.perm)
        else:
            self.shape = self.layout.transform(self.shape)

    def validate(self):
        """A tensor is valid if it has a name and a correct number of producers"""
        if self.isInitializer:
            assert len(self.producers) == 0, "Initializer should not have producer"
        else:
            assert len(self.producers) <= 1, "Tensor should not have more than 1 producer"
        assert len(self.name) > 0, "Tensor must have valid name"

    def convert(self):
        """Convert the arbitrary data structure at self.data into numpy array
        (in case it is a weight or bias) or Keras Input tensor at self.keras"""
        if self.status.converted:
            return
        logger.debug("Converting %s...", self.shorty)
        dtype = mapping.DTYPE_TFLITE2NAME[self.dtype]
        if self.isInitializer:
            if isinstance(self.data, np.ndarray):
                self.keras = self.data
            else:
                self.keras = np.asarray(self.data, dtype=dtype).reshape(self.shape)
        else:
            from tensorflow.keras.layers import Input
            shape = self.shape[1:]
            self.keras = Input(shape=shape, dtype=dtype, name=self.name)
        assert self.keras is not None
        self.setConverted()

    @property
    def shorty(self):
        return '<%s>(%s,%s)' % (self.name, mapping.DTYPE_TFLITE2NAME[self.dtype], self.shape)

    def __str__(self):
        producer_names = [op.shorty for op in self.producers]
        consumer_names = [op.shorty for op in self.consumers]
        return '%s: {%s} -> {%s}' % (self.shorty, producer_names, consumer_names)


class TensorFactory:
    """The registry holds all tensors in a SubGraph of TFLite by a name->Tensor map."""

    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.registry = dict()

    def get(self, index, layout=None, is_bias=False):
        """get a tflite tensor from graph based on id, remember in registry
        dict using name as key
        """
        tft = self.graph.Tensors(index)
        if index < 0:
            return tft  # this is dummy
        name = tft.Name().decode('utf-8')
        if name not in self.registry:
            t = Tensor(self.model, self.graph, index, layout, is_bias)
            self.registry[name] = t
        else:
            t = self.registry[name]
            if t.layout is None:
                t.layout = layout
        return t

    def getWithRef(self, ref, name, forceUnique=False):
        """Create a copy of the ref tensor.

        This is used to create helper tensors for activations, layout handling,
        quantization and so on. Some attributions will be removed.
        """
        if name not in self.registry:
            t = Tensor(self.model, self.graph, -1)
            t.name = name
            t.dtype = ref.dtype
            t.layout = copy.deepcopy(ref.layout)
            t.shape = copy.deepcopy(ref.shape)
            t.scale = copy.deepcopy(ref.scale)
            t.zero_point = copy.deepcopy(ref.zero_point)
            self.registry[name] = t
        else:
            assert not forceUnique
            t = self.registry[name]
        return t

    def createScalar(self, dtype, value):
        name = 'T2K_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    def createVector(self, ndarray):
        array2key = str(ndarray).replace(' ', '_')
        dtype = str(ndarray.dtype)
        name = 'T2K_Vector_' + dtype + '_' + array2key
        if name not in self.registry:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2TFLITE[dtype]
            t.data = ndarray.copy()
            t.shape = t.data.shape
            t.setParsed()
            self.registry[name] = t
        return self.registry[name]

    def createEmptyTensor(self):
        # Used for optional inputs that we need it to be empty.
        logger.warning("Empty tensor used, please double confirm your code path!")
        name = 'T2K_EmptyTensor'
        if name not in self.registry:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2TFLITE['float32']
            t.data = []
            t.shape = [0]
            t.setParsed()
            self.registry[name] = t
        return self.registry[name]

    def _createScalarCore(self, name, dtype, value):
        if name not in self.registry:
            t = Tensor(self.model, self.graph, -1, None)
            t.name = name
            t.dtype = mapping.DTYPE_NAME2TFLITE[dtype]
            # cannot use NDArray for cases such as min/max of ReLU6
            t.data = [value]
            t.setParsed()
            self.registry[name] = t
        return self.registry[name]

    def createQuantScale(self, tensor):
        value = tensor.scale
        assert isinstance(value, float) or (len(value) == 1)
        dtype = 'float32'
        name = 'T2K_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    def createQuantZeroPoint(self, tensor):
        value = tensor.zero_point
        assert isinstance(value, int) or (len(value) == 1)
        assert value >= 0 and value <= 255
        dtype = 'uint8'
        name = 'T2K_Scalar_' + dtype + '_' + str(value)
        return self._createScalarCore(name, dtype, value)

    @staticmethod
    def getData(model, graph, index, dtype):
        """Extract a tensor from graph and cast to a particular type
        """
        if dtype not in ['int32', 'float32', 'uint8']:
            logger.warning("Data type %s not supported/tested yet, "
                           "the generated model may contain error", dtype)
        assert index < graph.TensorsLength()
        t = graph.Tensors(index)
        bi = t.Buffer()
        shape = t.ShapeAsNumpy()
        assert bi < model.BuffersLength()
        raw = model.Buffers(bi).DataAsNumpy()
        if isinstance(raw, int) and raw == 0:
            return None
        data = np.frombuffer(raw, dtype=dtype)
        if len(shape) > 0:
            data = data.reshape(shape)
        return data.copy()
