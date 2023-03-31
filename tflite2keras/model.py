"""Holding a TFLite model, which will extract the graphs stored in it. Only one
graph per model supported yet.
"""

import logging

import tflite
import tensorflow as tf

from .common import T2KBase
from .graph import Graph

logger = logging.getLogger('t2k.model')


class Model(T2KBase):
    """Everything helps to convert TFLite model to Keras model"""

    def __init__(self, model: tflite.Model):
        super().__init__(model)
        self.tflite = model
        self.graphs: list[Graph] = []
        self.setInited()

    def parse(self):
        logger.debug("Parsing the Model...")
        graph_count = self.model.SubgraphsLength()
        if (graph_count != 1):
            raise NotImplementedError("T2K supports one graph per model only, while TFLite has ",
                                      graph_count)
        # obtain a subgraph object
        tflg = self.model.Subgraphs(0)
        graph = Graph(self.model, tflg)
        self.graphs.append(graph)

        for g in self.graphs:
            g.parse()

        self.setParsed()

    def validate(self):
        pass

    def convert(self, explicit_layouts=None, details=False):
        """Convert the graph in this model into Keras model

        Args:
            explicit_layouts: A dict to map tensor name into a pair of TFLite
                              layout-Keras layout
            details: If true, return both the Keras model and the dict of ops and tensors

        Returns:
            If details is False, only the functional Keras model converted from
            TFLite. If details is True, a 3-tuple of Keras model, a dict of name
            to op objects, and a dict of name to KerasTensor
        """
        self.parse()
        logger.debug("Converting...")
        for g in self.graphs:
            ret = g.convert(explicit_layouts, details)

        self.keras = self.graphs[0].keras
        self.setConverted()
        return ret

    def save(self, path: str):
        logger.debug("saving model as %s", path)
        assert self.status.converted
        tf.keras.models.save_model(self.keras, path)

    @property
    def shorty(self):
        return "Model holder"

    def __str__(self):
        return self.shorty
