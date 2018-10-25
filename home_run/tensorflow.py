from .base import BaseServable
from functools import partial
import tensorflow as tf
import os


class TensorFlowServable(BaseServable):

    def _build(self):
        # Initialize the TF environment
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # Get the directory that contains the pb file
        workdir = None
        for f in self.dlhub['files']['other']:
            if f.endswith('.pb'):
                workdir = os.path.dirname(f)

        # Load in that directory
        self.model = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING],
                                                workdir)

        # Create methods for the other operations
        for name in self.servable['methods'].keys():
            if name is not "run":
                self._set_function(name, partial(self._call_graph, name))

    def _call_graph(self, method, inputs, **parameters):
        """Call a certain function on the current graph.

        Looks up the node names using the information stored in the servable metadata

        Args:
            method (str): Name of the method to run
            inputs: Inputs for the method (probably a Tensor or list of Tensors)
        Returns:
            Outputs of calling the functions
        """

        # Get the input and output names
        run_info = self.servable['methods'][method]
        input_nodes = run_info['method_details']['input_nodes']
        output_nodes = run_info['method_details']['output_nodes']

        # Make the input field dictionary
        if len(input_nodes) == 1:
            feed_dict = {input_nodes[0]: inputs}
        else:
            feed_dict = dict(zip(input_nodes, inputs))

        # Run the method
        return self.sess.run(output_nodes, feed_dict=feed_dict)

    def _run(self, inputs, **parameters):
        return self._call_graph('run', inputs, **parameters)
