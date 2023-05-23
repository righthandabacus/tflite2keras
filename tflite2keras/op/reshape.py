import copy
import logging
import tflite
import numpy as np

from .. import mapping
from ..layout import Layout
from .common import Operator
from .transpose import Transpose

logger = logging.getLogger('t2k.reshape')


class Reshape(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.RESHAPE: 'Reshape',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.forFakeBroadcasting = False

        self.setInited()

    @property
    def type(self):
        return 'Reshape'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping

        assert op.InputsLength() >= 1
        assert op.OutputsLength() == 1

        # input
        self.parseInput(0)

        if op.InputsLength() == 1:
            # This path has not been tested by CI as we don't have a simple model for it.
            # See https://github.com/tensorflow/tensorflow/issues/45150
            op_opt = op.BuiltinOptions()
            option = tflite.ReshapeOptions()
            option.Init(op_opt.Bytes, op_opt.Pos)
            sp = option.NewShapeAsNumpy()
            sp = self.TFactory.createVector(sp.astype('int64'))
            sp.addConsumer(self)
            self.inputs.append(sp)

        if op.InputsLength() == 2:
            # shape
            st = self.parseInput(1)

            # TFLite shape is int32 data type, Keras is int32
            st.dtype = mapping.DTYPE_NAME2TFLITE['int32']
            if st.isInitializer:
                st.data = st.data.astype('int64')
            if len(st.shape) > 1:
                logger.warning("ONNXRuntime doesn't support 2+rank shape, "
                               "flatten if the shape is initialzier, ignore otherwise."
                               "https://github.com/zhenhuaw-me/tflite2onnx/issues/16")
                if st.isInitializer:
                    st.shape = (np.prod(np.array(st.shape)),)

        # output
        self.parseOutput(0)

        self.setParsed()

    def preserveInputSpatialSemantic(self):
        # https://github.com/zhenhuaw-me/tflite2onnx/issues/28
        # An example for inserting `Transpose` before `Reshape`
        #    ------
        #   | Conv |
        #    ------
        #      |
        #      |  to_transpose (Original input of `Reshape`)
        #      |  (e.g. NCHW)
        #   -------
        #  | Trans |  e.g. perm: (0, 2, 3, 1)
        #   -------
        #      |
        #      |  transposed (New created tensor)
        #      |  (e.g. NHWC)
        #   --------
        #  | Reshape |
        #   --------

        assert self.status.parsed
        to_transpose = self.inputs[0]

        transposed_name = 'T2K_Transposed_%s' % to_transpose.name
        transposed = self.TFactory.getWithRef(to_transpose, transposed_name, True)

        # Construct the layout from the original input of `Reshape`
        layout = Layout(to_transpose.layout.target, to_transpose.layout.source)
        transposed.shape = layout.transform(to_transpose.shape)
        transposed.setParsed()

        # Construct the additional transpose before `Reshape`
        trans = Transpose(self.TFactory, -1)
        trans.attrs['perm'] = layout.perm

        trans.inputs.append(to_transpose)
        to_transpose.replaceConsumer(self, trans)
        self.replaceInput(to_transpose, transposed)

        trans.outputs.append(transposed)
        transposed.addProducer(trans)
        transposed.addConsumer(self)
        trans.setParsed()

        self.pre.append(trans)

    def preserveOutputSpatialSemantic(self):
        # https://github.com/zhenhuaw-me/tflite2onnx/issues/28
        # An example for inserting `Transpose` after `Reshape`
        #   -------
        # | Reshape |
        #   -------
        #      |
        #      |  to_transpose (New created tensor)
        #      |  (e.g. NHWC)
        #   -------
        #  | Trans |  e.g. perm: (0, 3, 1, 2)
        #   -------
        #      |
        #      |  transposed (Original `Reshape` output)
        #      |  (e.g. NCHW)
        #    ------
        #   | Conv |
        #    ------

        assert self.status.parsed
        transposed = self.outputs[0]

        to_transpose_name = 'T2K_ToTranspose_%s' % transposed.name
        to_transpose = self.TFactory.getWithRef(transposed, to_transpose_name, True)

        # Construct a layout from the original output of `Reshape`
        layout = Layout(transposed.layout.target, transposed.layout.source)
        to_transpose.shape = layout.transform(transposed.shape)
        to_transpose.setParsed()

        # Construct the additional transpose after `Reshape`
        trans = Transpose(self.TFactory, -1)
        trans.attrs['perm'] = transposed.layout.perm

        trans.inputs.append(to_transpose)
        transposed.replaceProducer(self, trans)
        self.replaceOutput(transposed, to_transpose)

        trans.outputs.append(transposed)
        to_transpose.addProducer(self)
        to_transpose.addConsumer(trans)
        trans.setParsed()
        # Rename the new `Transpose` operator avoid the name conflict with 'Reshape'
        trans.name = 'T2K_Transpose_%s' % transposed.name

        self.post.append(trans)

    def propagatableTensors(self):
        return list()

    def transform(self):
        i = self.inputs[0]
        o = self.outputs[0]

        if self.forFakeBroadcasting:
            assert len(i.shape) != len(o.shape)
            shape_t = self.inputs[1]
            layout = copy.deepcopy(o.layout)
            if layout is None:
                return  # XXX can I?
                raise ValueError("Requires layout description for <%s>" % i.name)
            shape_t.data = np.array(layout.transform(shape_t.data))
        else:
            # Insert `Transpose` before/after `Reshape` to preserve spatial semantic
            if i.layout and (i.layout.source != i.layout.target):
                self.preserveInputSpatialSemantic()
            if o.layout and (o.layout.source != o.layout.target):
                self.preserveOutputSpatialSemantic()

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create Reshape layer but not set the weight"""
        from tensorflow.keras.layers import Reshape, Flatten
        kerasattrs = {
            "target_shape": self.inputs[1].data.tolist()[1:],  # ignore batch axis
            "name": self.derive_name(),
        }
        if kerasattrs["target_shape"] == [-1]:
            del kerasattrs["target_shape"]
            layer = Flatten(**kerasattrs)
        else:
            kerasattrs["target_shape"] = self.outputs[0].shape[1:]
            layer = Reshape(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
