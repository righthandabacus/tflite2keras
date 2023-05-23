from __future__ import annotations   # until Py311, treat annotations as strings

import copy
import logging
import pprint
from collections import defaultdict

import tflite
import tensorflow as tf
from tensorflow.keras.models import Model

from .tensor import TensorFactory
from .common import T2KBase
from .layout import Layout
from .op import OpFactory
from .quantize import handleQuantizationTensor, foldFP16QuantPattern

logger = logging.getLogger('t2k.graph')
pp = pprint.PrettyPrinter(indent=4, width=100, compact=True, sort_dicts=True)


class Graph(T2KBase):
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph):
        super().__init__(model, graph)

        self.ops = []   # the OP that has TFLite peer
        self.op_all = []  # includes helper OP

        self.inputs = []
        self.outputs = []
        self.initializer = set()   # tensors with data, i.e., weights and biases
        self.value_info = set()    # placeholder tensors

        self.tflite = graph
        # TFLite tensors = KerasTensor or weights. TensorFactory: Extract
        # tensors from TFLite based on indices from graph
        self.TFactory = TensorFactory(model, graph)
        # TFLite op = Keras layers. OpFactory: help extract layer-specific
        # attributes (e.g., strides in conv)
        self.OPCFactory = OpFactory(self.TFactory)

        self.setInited()

    def _collectOpAndTensor(self):
        """Upon return, this function makes:
            - all ops (i.e., layers) in list self.op_all
            - all weight tensors in set self.initializer
            - all dummy tensors (as layer input/output, only shape, no data) in set self.value_info
        """
        self.op_all.clear()

        # collect operators
        def _recursive(op):
            for cur_op in op.pre:
                _recursive(cur_op)
            self.op_all.append(op)
            for cur_op in op.post:
                _recursive(cur_op)
        for op in self.ops:
            _recursive(op)

        # collect tensors
        assert len(self.op_all) > 0
        self.initializer.clear()
        self.value_info.clear()
        for op in self.op_all:
            for t in op.inputs + op.outputs:
                if t.isInitializer:
                    self.initializer.add(t)
                else:
                    self.value_info.add(t)

    def parse(self):
        logger = logging.getLogger("t2k.graph.parse")
        # operators
        for i in range(self.graph.OperatorsLength()):
            logger.debug("Parsing operator: %d", i)
            op = self.OPCFactory.create(i)
            op.parse()
            self.ops.append(op)

        # inputs
        for i in range(self.graph.InputsLength()):
            logger.debug("Parsing input: %d", i)
            # FIXME: assert they have been created.
            index = self.graph.Inputs(i)
            t = self.TFactory.get(index)
            self.inputs.append(t)

        # outputs
        for i in range(self.graph.OutputsLength()):
            logger.debug("Parsing output: %d", i)
            index = self.graph.Outputs(i)
            t = self.TFactory.get(index)
            self.outputs.append(t)

        self._collectOpAndTensor()

        self.setParsed()

    def validate(self):
        """Validate ops and tensors in graph: They should fulfil some rules to
        make the graph valid
        """
        self._collectOpAndTensor()
        for op in self.op_all:
            op.validate()
        for t in self.initializer | self.value_info:
            t.validate()

    @staticmethod
    def toposort(op_input, op_output):
        """Return topological sorted order of output tensors

        Args:
            op_input: Dict of tensor name to list of consumer op names
            op_output: Dict of tensor name to producer op name

        Return:
            list of name of topologically sorted order of output tensors, which
            the first element is a sick of the DAG and last element is a source
        """
        # make a dict of opname -> output tensor name
        ops = {v: k for k, v in op_output.items()}
        # make a dict of input tenor name -> list of output tensor names; this is a DAG
        graph = {k: [ops[n] for n in v] for k, v in op_input.items()}

        # topological sort start with the source nodes and end with sinks
        def _tsort(loopover):
            "Recursive topological sorter"
            for n in loopover:
                if n in seen:
                    continue
                elif n not in graph:
                    seen.add(n)
                    yield n  # this is a sink
                else:
                    unseen = [m for m in graph[n] if m not in seen]
                    yield from _tsort(unseen)
                    seen.add(n)
                    yield n

        seen = set()
        sources = [x for x in op_input if x not in op_output]
        toposorted = list(_tsort(sources))
        return toposorted

    @staticmethod
    def _fix_transpose_bias(op_all, op_input, op_output):
        # Hack: Some TFLite model will use Add operation to insert bias to ConvTranspose
        consts = {}
        for name, op in op_all.items():
            if all(t.isInitializer for t in op.inputs):
                # all-constant ops, this is the bias to ConvTranspose
                op = op_all[name]
                assert op.type == "Reshape", "Only reshape constant op is expected"
                assert all(x==1 for x in op.keras.target_shape[:-1])  # expect shape (1,1,1,n)
                consts[op.outputs[0].name] = op.inputs[0]
        for opname in [o for k in consts for o in op_input[k]]:
            op = op_all[opname]
            assert op.type == "Add", "Only Add op expected to use constant tensor"
            assert len(op.inputs) == 2, "Only Add of two inputs are expected with constant tensor"
            other_in = [x for x in op.inputs if x.name not in consts][0]
            const_in = [x for x in op.inputs if x.name in consts][0]
            assert len(other_in.shape) == len(const_in.shape) == 4
            assert other_in.shape[0] == const_in.shape[0] == 1
            assert const_in.shape[1] == const_in.shape[2] == 1
            assert other_in.shape[3] == const_in.shape[3]
            assert consts[const_in.name].data.ndim == 1
            assert len(consts[const_in.name].data) == const_in.shape[3]
            assert len(op_input[const_in.name]) == 1
            producer = op_all[op_output[other_in.name]]
            assert producer.type == "ConvTranspose"
            assert len(producer.inputs) == 2, "Expecting a ConvTranspose with no bias"
            # append the bias to ConvTranspose, replace ConvTranspose output to Add's output
            producer.inputs.append(consts[const_in.name])
            producer.outputs = op.outputs
            op_output[producer.outputs[0].name] = producer.name
            # remove the Add operation from DAG
            del op_input[other_in.name]
            del op_input[const_in.name]
            # update Keras object
            config = producer.keras.get_config()
            config["use_bias"] = True
            producer.keras = producer.keras.from_config(config)

    @staticmethod
    def _fuse_activations(op_all, op_input, op_output):
        """Fuse activation layer to the previous layer, fusing only supported
        for Conv and FC layers
        """
        for oname, op in op_all.items():
            if op.type in ["Relu", "Tanh", "Sigmoid"]:
                assert len(op.inputs) == len(op.outputs) == 1
                # find the previous layer
                prev_name = op_output[op.inputs[0].name]
                prev_op = op_all[prev_name]
                # rebuild the previous layer with activation fused
                if prev_op.type not in ["Conv", "ConvTranspose", "Gemm"]:
                    continue
                config = prev_op.keras.get_config()
                config["activation"] = op.type.lower()
                prev_op.keras = prev_op.keras.from_config(config)
                # reset the output, and remove myself
                prev_op.outputs = op.outputs
                del op_input[op.inputs[0].name]
                op_output[op.outputs[0].name] = prev_op.name

    def convert(self, explicit_layouts=None, details=False):
        """Convert a TFLite graph into Keras model. Only a subset of TFLite model is supported.

        Args:
            explicit_layouts: A dict to map tensor name into a pair of TFLite
                              layout-Keras layout
            details: If true, return both the Keras model and the dict of ops and tensors

        Returns:
            If details is False, only the functional Keras model converted from
            TFLite. If details is True, a 3-tuple of Keras model, a dict of name
            to op objects, and a dict of name to KerasTensor
        """
        logger = logging.getLogger("t2k.graph.convert")

        # transforming tensor layout if explicit_layouts are specified for some layers
        explicit_layouts = explicit_layouts or {}
        t_all = [t for op in self.ops for t in op.inputs+op.outputs if t.name in explicit_layouts]
        for t in t_all:
            assert t.layout is None
            layouts = explicit_layouts[t.name]
            assert len(layouts) == 2
            t.layout = Layout(layouts[0], layouts[1])
        self._propagateLayout()
        self._collectOpAndTensor()

        # remove TFLite quantization on weight tensors
        foldFP16QuantPattern(self.ops)
        self._collectOpAndTensor()

        logger.debug("Translating quantization semantic...")
        for t in self.value_info | self.initializer:
            deqt = handleQuantizationTensor(self.TFactory, t)
            for i, o in enumerate(self.outputs):
                if o == t:
                    self.outputs[i] = deqt
        self._collectOpAndTensor()

        logger.debug("Graph:\n%s", str(self))
        self.validate()

        op_all = {}                   # op name -> op object
        op_input = defaultdict(list)  # TFLite tensor name -> list of consumer op name
        op_output = {}                # TFLite tensor name -> producer op name
        for op in self.op_all:
            # transform each TFLite op into a Keras layer object stored in op.keras
            # weights are still at op.inputs and not set to layers
            #
            # TFLite is optimized: If Dense layer is created with default
            # setting, bias is init with all zero. Converting such layer to
            # TFLite will have the bias removed as it has no effect. This may
            # distort the attributes to the Keras object
            op.convert()
            assert len(op.outputs) == 1, "FIXME what layer is this?"
            # remember this layer as consumer of input tensors
            op_all[op.name] = op
            self.TFactory.keras_names.add(op.keras.name)
            for op_in in op.inputs:
                if op_in.isInitializer:
                    continue
                op_input[op_in.name].append(op.name)
            # remember this layer as producer of output tensors
            for op_out in op.outputs:
                if op_out.isInitializer:
                    continue
                op_output[op_out.name] = op.name

        # Clean-up Keras design: Combine ConvTranpose bias logic and fuse activations
        self._fix_transpose_bias(op_all, op_input, op_output)
        self._fuse_activations(op_all, op_input, op_output)

        # populating KerasTensor from input until output
        # TODO code generation can be piggybacked here!
        topoorder = self.toposort(op_input, op_output)
        avail = {t.name: t.keras for t in self.inputs}
        for tname in reversed(topoorder):
            if tname in avail:
                # when tname is an input/output tensor
                continue
            # create output KerasTensor
            op = op_all[op_output[tname]]
            inputs = [x.name for x in op.inputs if not x.isInitializer]
            inputs = [avail[x] for x in inputs]
            avail[tname] = op.keras(inputs[0] if len(inputs) == 1 else inputs)
            # set weights to Keras layer object
            weights = [x.data for x in op.inputs if x.isInitializer]
            if weights and op.type not in ["Reshape", "Resize", "Pad", "Clip"]:
                op.keras.set_weights(weights)

        # at this point, we should expect the graph output is ready
        assert all(t.name in avail for t in self.outputs)
        inputs = [avail[t.name] for t in self.inputs]
        outputs = [avail[t.name] for t in self.outputs]
        self.keras = Model(inputs=inputs, outputs=outputs, name=self.name)
        self.setConverted()
        if details:
            return self.keras, op_all, avail
        else:
            return self.keras

    def _propagateLayout(self):        # noqa: C901
        """Populate layout transformations to tensors
        """
        logger = logging.getLogger("t2k.graph.prop")

        # collect tensors
        T_toWalk = set()
        T_wild = set()
        tensor_count = len(self.value_info) + len(self.initializer)
        for t in self.value_info | self.initializer:
            if t.layout is None:
                T_wild.add(t)
            else:
                T_toWalk.add(t)
        logger.debug("Propagation: %d tensors in total, %d to walk, %d at wild",
                     tensor_count, len(T_toWalk), len(T_wild))

        # propagrate layout across graph
        T_ignored = set()
        T_walked = set()
        while T_toWalk:
            T = T_toWalk.pop()
            logger.debug("Propagation: walking %s", T.shorty)
            for n in T.producers + T.consumers:
                for t in n.propagatableTensors():
                    if (t is T) or (t not in T_wild):
                        continue
                    assert t.layout is None
                    T_wild.remove(t)
                    if t.isScalar:
                        T_ignored.add(t)
                    else:
                        logger.debug("Propagation: propagated to %s", t.shorty)
                        t.layout = copy.deepcopy(T.layout)
                        T_toWalk.add(t)
            T_walked.add(T)
        logger.debug("Propagation: wild tensors %d, ignored tensors %d",
                     len(T_wild), len(T_ignored))

        # update tensor and operator
        for t in T_walked:
            t.transform()
        self._collectOpAndTensor()
        for op in self.op_all:
            op.transform()

    def _dump(self, tag: str, container, useShorty: bool):
        dump = ['[%s] %s' % (tag, e.shorty if useShorty else e) for e in container]
        return "\n".join(dump)

    @property
    def shorty(self):
        string = [
            self._dump('OP', self.op_all, True),
            self._dump('Input', self.inputs, True),
            self._dump('Output', self.outputs, True),
            self._dump('Initializer', self.initializer, True),
            self._dump('Value Info', self.value_info, True),
        ]
        return "\n".join(string)

    def __str__(self):
        string = [
            self._dump('OP', self.op_all, False),
            self._dump('Input', self.inputs, False),
            self._dump('Output', self.outputs, False),
            self._dump('Initializer', self.initializer, False),
            self._dump('Value Info', self.value_info, False),
        ]
        return "\n".join(string)
