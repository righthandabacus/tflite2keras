from __future__ import annotations   # until Py311, treat annotations as strings

import logging
import tflite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import TensorFactory

from ..common import T2KBase

logger = logging.getLogger('t2k.opcommon')


class Operator(T2KBase):
    TypeMapping = dict()

    def __init__(self, TFactory: TensorFactory, index):
        super().__init__(TFactory.model, TFactory.graph, index)
        self.TFactory = TFactory
        self.tflite = self.graph.Operators(index) if index >= 0 else None
        self.inputs = []
        self.outputs = []
        self.pre = []  # ops that before this op which to enable TFLite op
        self.post = []  # ops that after this op which to enable TFLite op
        self.attrs = dict()  # One dict to hold all Keras operator attributes

    @property
    def type(self):
        raise NotImplementedError("Method Operator.type() must be overrided!")

    def propagatableTensors(self):
        """Get all layout propagable tensors of this operator.

        When we propagate layouts across the graph:
        1. Some operators may stop the propagation
            a) An operator assumes layouts of its tensors, `Conv` for example.
               Such operator needs to define the layouts of its tensors explicitly.
            b) An operator breaks layout semantic, `Reshape` for example.
               Tensors connected to this operator should be propagated.
               And the operator may need special handling regarding layout.
        2. Others may not - propagatable:
            a) An operator that is transparent to layout, such as Add.
               Just propagate the layouts.
            b) Layout can propagate across tensors of an operator, but the operator
               itself has attribution that is sensitive to layout.
               Operator needs special handling after propagation.
        This is defined per operator.

        To handle this, we firstly propagate layouts of tensors across the graph,
        and then update attributes of operators accordingly.
        """
        raise NotImplementedError("Method %s.propagatableTensors() must be overrided!" % self.type)

    def transform(self):
        """Transform the operator attributions w.r.t. propagated layouts.

        The attributions could be a tensor that describing layout related things.
        Operators that defined as 1.a, 1.b and 2.b in `layoutPropagatable()`
        are such cases. But not all of them need special treatment.
        For example, `Conv` doesn't need additional processing after propagation.

        This must be called after the layouts have been propagated across graph.
        """
        raise NotImplementedError("Method %s.transform() must be overrided!" % self.type)

    @property
    def str(self):
        return '[' + self.name + '] (' + self.type + ')'

    def parseInput(self, index, layout=None, is_bias=False):
        ii = self.tflite.Inputs(index)
        if ii < 0:
            return
        it = self.TFactory.get(ii, layout, is_bias)
        it.parse()
        it.addConsumer(self)
        self.inputs.append(it)
        return it

    def parseOutput(self, index, layout=None):
        oi = self.tflite.Outputs(index)
        ot = self.TFactory.get(oi, layout)
        ot.parse()
        ot.addProducer(self)
        self.outputs.append(ot)
        return ot

    def replaceInput(self, original, new):
        logger.debug("Replacing %s input %s with %s", self.shorty, original.shorty, new.shorty)
        assert original in self.inputs
        for i, item in enumerate(self.inputs):
            if item is original:
                self.inputs[i] = new
                return

    def replaceOutput(self, original, new):
        logger.debug("Replacing %s output %s with %s", self.shorty, original.shorty, new.shorty)
        assert original in self.outputs
        for i, item in enumerate(self.outputs):
            if item is original:
                self.outputs[i] = new
                return

    def setParsed(self):
        """Name the operator (if not yet) and change to initialized.

        Assume that the outputs won't change after parsed.
        * If the operator is a helper in TFLITE2KERAS, it should have been named already.
        * If the operator is original in TFLite, using name of its first output tensor.
        """
        self.name = self.outputs[0].name if self.name is None else self.name
        super().setParsed()

    def validate(self):
        assert len(self.outputs) >= 1, "Operator should produce something"

    def convert(self):
        logger.debug("Converting %s...", self.shorty)
        for t in self.inputs + self.outputs:
            t.convert()
        self.attrs['name'] = self.name
        inames = [t.name for t in self.inputs]
        onames = [t.name for t in self.outputs]
        # [AT] all Operator class must implement make_node() method to return a
        #      Keras object. After all operators in the graph are set up, we
        #      can walk through the graph again to create the KerasTensor
        self.keras = self.make_node(self.type, inames, onames, **self.attrs)
        self.setConverted()

    @property
    def shorty(self):
        return '[%s](%s)' % (self.name, self.type)

    def __str__(self):
        inames = str([t.name for t in self.inputs])
        onames = str([t.name for t in self.outputs])
        return '%s attr%s: %s -> %s' % (self.shorty, self.attrs, inames, onames)

    def make_node(self, nodetype, inames, onames, **attrs):
        breakpoint()
        raise NotImplementedError("This should implement a Keras layer for %s" % nodetype)

    def derive_name(self):
        """Derive a name for this layer"""
        # breakdown "path/path/name;path/path/name" and count
        votes = {}
        for part in self.name.split(";"):
            votes[part] = votes.get(part, 0) + 1
            end = len(part)
            while True:
                end = part.rfind("/", 0, end)
                if end == -1:
                    end = len(part)
                substr = part[:end]
                votes[substr] = votes.get(substr, 0) + 1
                if end == len(part):
                    break
        # find the best name
        _, _, name = max((v,len(k),k) for k,v in votes.items())
        name = name.split("/")
        if len(name) > 1:
            name = name[-2]
        else:
            name = name[-1]
        name = name.replace(":","_")
        return name



class OpFactory:
    """The factory for creating operater converter objects."""

    registry = dict()

    @staticmethod
    def register(converter):
        opcs = converter.TypeMapping.keys()
        for opc in opcs:
            assert opc not in OpFactory.registry
            OpFactory.registry[opc] = converter

    def __init__(self, TFactory: TensorFactory):
        self.model = TFactory.model
        self.graph = TFactory.graph
        self.TFactory = TFactory

    def create(self, index):
        op = self.graph.Operators(index)
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if opcode not in OpFactory.registry:
            if opcode in tflite.BUILTIN_OPCODE2NAME:
                name = tflite.opcode2name(opcode)
                raise NotImplementedError("Unsupported TFLite OP: {} {}!".format(opcode, name))
            else:
                raise ValueError("Opcode {} is not a TFLite builtin operator!".format(opcode))

        op_converter = OpFactory.registry[opcode]
        return op_converter(self.TFactory, index)

    @staticmethod
    def dump():
        return "Registered OP converter: %d" % len(OpFactory.registry)

    def __str__(self):
        return OpFactory.dump()
