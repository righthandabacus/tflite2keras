import logging
import tflite

from ..layout import Layout
from .activation import handleFusedActivation
from .common import Operator
from .padding import computePaddingSize

logger = logging.getLogger('t2k.conv')


class Conv(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.CONV_2D: 'Conv',
        tflite.BuiltinOperator.DEPTHWISE_CONV_2D: 'Conv',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        # ONNX: This attribute cannot be used simultaneously with `auto_pad` attribute.
        # re-initialize during self.parse(), as it needs the shape of input.
        # We prefer `auto_pad`, however ONNXRuntime doesn't support
        # `dilation` + `auto_pad`, such that we use `pads` to workaround it.
        self.attrs['pads'] = [0, 0, 0, 0]
        # XXX Not enabled as ONNXRuntime has limitation to infer pads for non-1 dilation
        # self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        self.attrs['dilations'] = []
        self.attrs['group'] = -1

        self.setInited()

    @property
    def type(self):
        return 'Conv'

    @property
    def isDepthwise(self):
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        return (opcode is tflite.BuiltinOperator.DEPTHWISE_CONV_2D)

    def parse(self):
        """Extract attributes from TFLite operator options"""
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping
        assert op.InputsLength() == 3, "TFLite Conv always has bias"
        assert op.OutputsLength() == 1

        # input
        it = self.parseInput(0)

        # weight
        wlayout = Layout('CHWM', 'HWMC') if self.isDepthwise else Layout('OHWI', 'HWIO')
        wt = self.parseInput(1, wlayout)

        # bias
        self.parseInput(2, is_bias=True)

        # output
        ot = self.parseOutput(0)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.DepthwiseConv2DOptions() if self.isDepthwise else tflite.Conv2DOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['dilations'] = [option.DilationHFactor(), option.DilationWFactor()]
        self.attrs['group'] = wt.shape[3] if self.isDepthwise else 1
        self.attrs['kernel_shape'] = wt.shape[1:3]
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]
        self.attrs['auto_pad'] = {0: "same", 1: "valid"}.get(option.Padding())
        if self.isDepthwise:
            assert option.DepthMultiplier() == 1
            self.attrs["depth_multiplier"] = option.DepthMultiplier()
        self.attrs['pads'] = computePaddingSize(option.Padding(), it.shape[1:3],
                                                self.attrs['kernel_shape'],
                                                self.attrs['strides'], self.attrs['dilations'])

        handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create Conv2D layer but not set the weight (that would need the input shape provided)"""
        from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
        kerasattrs = {
            "kernel_size": attrs["kernel_shape"],
            "strides": attrs["strides"],
            "padding": attrs["auto_pad"],
            "dilation_rate": attrs["dilations"],
            "name": self.derive_name(),
        }
        if attrs["auto_pad"] is None:
            raise NotImplementedError("Padding other than `valid` and `same` are not implemented")
        if len(self.inputs) == 2:
            kerasattrs["use_bias"] = False
        if self.isDepthwise:
            kerasattrs["depth_multiplier"] = attrs["depth_multiplier"]
            layer = DepthwiseConv2D(**kerasattrs)
        else:
            kerasattrs["filters"] = self.inputs[1].data.shape[-1]
            layer = Conv2D(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer


class TransposeConv(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE_CONV: 'ConvTranspose',
    }

    # FIXME: cases that untested yet (we are not fully understand the semantic gap)
    # 1. Special output shape for VALID padding
    # 2. Different input/output shape for SAME padding

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['dilations'] = [1, 1]  # TFLite TransposeConv doesn't have dilation
        self.attrs['group'] = 1  # TFLite TransposeConv doesn't have group
        self.attrs['kernel_shape'] = []
        # self.attrs['output_padding'] = []
        self.attrs['output_shape'] = []
        # pads are overwrited by output_shape
        # self.attrs['auto_pad'] = 'NOTSET'
        # self.attrs['pads'] = []
        self.attrs['strides'] = []

        self.setInited()

    @property
    def type(self):
        return 'ConvTranspose'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

        assert opcode in self.TypeMapping
        assert op.InputsLength() == 3
        assert op.OutputsLength() == 1

        # oshape
        osi = op.Inputs(0)
        oshape = self.TFactory.getData(self.model, self.graph, osi, 'int32')

        # X
        self.parseInput(2)

        # weight
        wlayout = Layout('OHWI', 'HWIO')
        wt = self.parseInput(1, wlayout)

        # FIXME: we don't have a model containing bias.

        # output
        ot = self.parseOutput(0)
        assert (ot.shape == oshape).all()

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.TransposeConvOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['auto_pad'] = {0: "same", 1: "valid"}.get(option.Padding())
        self.attrs['kernel_shape'] = wt.shape[1:3]
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]
        oslayout = Layout('NHWC', 'NHWC')
        self.attrs['output_shape'] = oslayout.transform(oshape)
        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create TransposeConv2D layer but not set the weight"""
        from tensorflow.keras.layers import Conv2DTranspose
        kerasattrs = {
            "filters": self.inputs[1].data.shape[0],
            "kernel_size": attrs["kernel_shape"],
            "strides": attrs["strides"],
            "padding": attrs["auto_pad"],
            "dilation_rate": attrs["dilations"],
            "name": self.derive_name(),
        }
        if attrs["auto_pad"] is None:
            raise NotImplementedError("Padding other than `valid` and `same` are not implemented")
        if len(self.inputs) == 2:
            kerasattrs["use_bias"] = False
        layer = Conv2DTranspose(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
