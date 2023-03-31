import logging
import tflite

from .activation import handleFusedActivation
from .common import Operator
from .padding import PaddingMapping

logger = logging.getLogger('t2k.pooling')


class Pooling(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.AVERAGE_POOL_2D: 'AveragePool',
        tflite.BuiltinOperator.MAX_POOL_2D: 'MaxPool',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        # ceil_mod = 0

        self.setInited()

    @property
    def type(self):
        if self.status.uninitialized:
            return 'Pooling'
        else:
            op = self.tflite
            opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            assert opcode in self.TypeMapping
            return self.TypeMapping[opcode]

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping

        assert op.InputsLength() == 1
        assert op.OutputsLength() == 1

        self.parseInput(0)

        op_opt = op.BuiltinOptions()
        option = tflite.Pool2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.attrs['auto_pad'] = PaddingMapping[option.Padding()]
        self.attrs['kernel_shape'] = [option.FilterHeight(), option.FilterWidth()]
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]

        ot = self.parseOutput(0)

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create Conv2D layer but not set the weight (that would need the input shape provided)"""
        from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
        kerasattrs = {
            "pool_size": attrs["kernel_shape"],
            "strides": attrs["strides"],
            "padding": {"SAME_UPPER":"same", "VALID":"valid"}.get(attrs["auto_pad"]),
            "name": self.derive_name(),
        }
        if attrs["auto_pad"] is None:
            raise NotImplementedError("Padding other than `valid` and `same` are not implemented")
        if self.type == "AveragePool":
            layer = AveragePooling2D(**kerasattrs)
        elif self.type == "MaxPool":
            layer = MaxPooling2D(**kerasattrs)
        else:
            raise NotImplementedError("unknown pooling type")
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
