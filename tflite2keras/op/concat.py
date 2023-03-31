import logging
import tflite

from .activation import handleFusedActivation
from .common import Operator

logger = logging.getLogger('t2k.concat')


class Concat(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONCATENATION: 'Concat',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['axis'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Concat'

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping

        assert op.InputsLength() >= 1
        assert op.OutputsLength() == 1

        for i in range(op.InputsLength()):
            self.parseInput(i)

        op_opt = op.BuiltinOptions()
        option = tflite.ConcatenationOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)
        self.attrs['axis'] = option.Axis()

        self.parseOutput(0)

        handleFusedActivation(self, option, self.outputs[0])

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        logger.debug("Transforming %s...", self.shorty)
        layout = self.outputs[0].layout
        if layout is not None:
            axis = self.attrs['axis']
            axis = axis if axis >= 0 else (axis + len(layout.perm))
            self.attrs['axis'] = layout.perm.index(axis)

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create concat operation layer"""
        from tensorflow.keras.layers import Concatenate
        kerasattrs = {
            "axis": attrs["axis"],
            "name": self.derive_name(),
        }
        layer = Concatenate(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
