import logging
import tflite
import numpy as np

from .. import mapping
from .common import Operator

logger = logging.getLogger('t2k.resize')


class Resize(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: 'Resize',
        tflite.BuiltinOperator.RESIZE_BILINEAR: 'Resize',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        # Four choices:
        # half_pixel, pytorch_half_pixel, align_corners, asymmetric, tf_crop_and_resize
        self.attrs['coordinate_transformation_mode'] = 'half_pixel'
        # This attribute is valid only if "mode" is "cubic".
        # The coefficient 'a' used in cubic interpolation.
        # Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
        self.attrs['cubic_coeff_a'] = -0.75
        self.attrs['exclude_outside'] = 0
        self.attrs['extrapolation_value'] = 0.0
        # Three interpolation modes: nearest (default), linear and cubic.
        # The "linear" mode includes linear interpolation for 1D tensor
        # and N-linear interpolation for N-D tensor
        # (for example, bilinear interpolation for 2D tensor).
        # The "cubic" mode includes cubic interpolation for 1D tensor
        # and N-cubic interpolation for N-D tensor
        # (for example, bicubic interpolation for 2D tensor).
        self.attrs['mode'] = 'nearest'
        # Four modes: round_prefer_floor (default, as known as round half down),
        # round_prefer_ceil (as known as round half up), floor, ceil.
        # Only used by nearest interpolation.
        # It indicates how to get "nearest" pixel in input tensor from x_original,
        # so this attribute is valid only if "mode" is "nearest".
        self.attrs['nearest_mode'] = 'round_prefer_floor'

        self.setInited()

    @property
    def type(self):
        return 'Resize'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert opcode in self.TypeMapping

        assert op.InputsLength() == 2, "TFLite has only two inputs"
        assert op.OutputsLength() == 1

        im = self.parseInput(0)

        # ROI and Scale are not optional until Resize v13,
        # currently (v11) we create them as empty initializer.
        # After v13, we can try to not include them in graph
        empty_input = self.TFactory.createEmptyTensor()
        empty_input.addConsumer(self)
        self.inputs.append(empty_input)  # ROI
        self.inputs.append(empty_input)  # Scale

        # output size
        sz = self.parseInput(1)
        # TFLite sizes is (H_new, W_new) while ONNX needs (N, C, H_new,W_new)
        assert len(sz.data) == 2
        assert len(im.shape) == 4
        sz.shape = [len(im.shape)]
        sz.data = np.concatenate((np.array([im.shape[0], im.shape[-1]]), sz.data))
        sz.dtype = mapping.DTYPE_NAME2TFLITE['int32']

        # output
        self.parseOutput(0)

        # options
        if opcode is tflite.BuiltinOperator.RESIZE_BILINEAR:
            self.attrs['mode'] = 'linear'
            option = tflite.ResizeBilinearOptions()
        elif opcode is tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            self.attrs['mode'] = 'nearest'
            option = tflite.ResizeNearestNeighborOptions()
        else:
            assert False, "Unreachable path!"

        op_opt = op.BuiltinOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        if option.AlignCorners():
            self.attrs['coordinate_transformation_mode'] = 'align_corners'
        elif option.HalfPixelCenters():
            self.attrs['coordinate_transformation_mode'] = 'half_pixel'
        else:
            raise NotImplementedError("This path has not been tried")

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass

    def make_node(self, nodetype, inames, onames, **attrs):
        """Create upsampling layer"""
        from tensorflow.keras.layers import UpSampling2D
        opcode = self.model.OperatorCodes(self.tflite.OpcodeIndex()).BuiltinCode()
        assert self.outputs[0].shape[0] == self.inputs[0].shape[0]      # N of NHWC
        assert self.outputs[0].shape[1] % self.inputs[0].shape[1] == 0  # H of NHWC
        assert self.outputs[0].shape[2] % self.inputs[0].shape[2] == 0  # W of NHWC
        assert self.outputs[0].shape[3] == self.inputs[0].shape[3]      # C of NHWC
        h_factor = self.outputs[0].shape[1] // self.inputs[0].shape[1]
        w_factor = self.outputs[0].shape[2] // self.inputs[0].shape[2]
        kerasattrs = {
            "size": [h_factor, w_factor],
            "name": self.derive_name(),
        }
        if opcode is tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            kerasattrs["interpolation"] = "nearest"
        elif opcode is tflite.BuiltinOperator.RESIZE_BILINEAR:
            kerasattrs["interpolation"] = "bilinear"
        else:
            raise NotImplementedError("to be implemented")
        layer = UpSampling2D(**kerasattrs)
        logger.info("%s(%s)",
                    layer.__class__.__name__,
                    ", ".join(f"{k}={repr(v)}" for k, v in kerasattrs.items()))
        return layer
