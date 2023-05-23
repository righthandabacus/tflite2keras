import logging
import tflite

from ..layout import Layout
from .activation import handleFusedActivation
from .common import Operator

logger = logging.getLogger('t2k.fullyconnected')


class FullyConnected(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.FULLY_CONNECTED: 'Gemm',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        # raw default values
        self.attrs['alpha'] = 1.0
        self.attrs['beta'] = 1.0
        # TFLite Fully Connected: A (M, K), B (N, K)
        self.attrs['transA'] = 0
        self.attrs['transB'] = 1

        self.setInited()

    @property
    def type(self):
        return 'Gemm'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping

        assert op.InputsLength() == 3, "TFLite Fullly Connected always has bias"
        assert op.OutputsLength() == 1

        # input
        self.parseInput(0)

        # weight
        wlayout = Layout('OI', 'IO')
        self.parseInput(1, wlayout)

        # bias
        self.parseInput(2, is_bias=True)

        # output
        ot = self.parseOutput(0)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.FullyConnectedOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        assert not option.KeepNumDims()
        assert option.WeightsFormat() is tflite.FullyConnectedOptionsWeightsFormat.DEFAULT

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create TransposeConv2D layer but not set the weight"""
        from tensorflow.keras.layers import Dense
        kerasattrs = {
            "units": self.inputs[1].data.shape[-1],
            "name": self.derive_name(),
        }
        if len(self.inputs) == 3 and (self.inputs[-1].data == 0).all():
            self.inputs = self.inputs[:-1]
        if len(self.inputs) == 2:
            kerasattrs["use_bias"] = False
        layer = Dense(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
