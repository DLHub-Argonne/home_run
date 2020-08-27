from .base import BaseServable
from functools import partial
import tensorflow as tf
import logging
import os
logger = logging.getLogger(__name__)


class TensorFlowServable(BaseServable):

    def _build(self):
        if tf.__version__ < '2':
            self._build_tf1()
        else:
            self._build_tf2()

    def _build_tf1(self):
        # Initialize the TF environment
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        logger.info('Started session. Available devices: {}'.format(self.sess.list_devices()))

        # Get the directory that contains the pb file
        workdir = None
        for f in self.dlhub['files']['other']:
            if f.endswith('.pb'):
                workdir = os.path.dirname(f)

        # Load in that directory
        self.model = tf.saved_model.loader.load(self.sess,
                                                [tf.saved_model.tag_constants.SERVING],
                                                workdir)
        logger.info('Loaded model from {}'.format(workdir))

        # Create methods for the other operations
        for name in self.servable['methods'].keys():
            if name != "run":
                self._set_function(name, partial(self._call_tf1_graph, name))
                logger.info('Mapped method {}. Inputs {}. Outputs {}'.format(
                    name, self.servable['methods'][name]['method_details']['input_nodes'],
                    self.servable['methods'][name]['method_details']['output_nodes']
                ))
        logger.info('Mapped method {}. Inputs {}. Outputs {}.'.format(
            'run', self.servable['methods']['run']['method_details']['input_nodes'],
            self.servable['methods']['run']['method_details']['output_nodes']
        ))

    def _build_tf2(self):
        # Get the directory that contains the pb file
        workdir = None
        for f in self.dlhub['files']['other']:
            if f.endswith('.pb'):
                workdir = os.path.dirname(f)

        # Load in the model
        self.model = tf.saved_model.load(workdir)

        # Map functions to the shim
        for name, info in self.servable['methods'].items():
            # Determine the proper inputs to the function
            is_multiinput = info['input']['type'] == "tuple"
            is_multioutput = info['output']['type'] == "tuple"
            if name == "run":
                name = "__call__"  # It runs using the callable method
            func = partial(self._call_tf2_eager_function, name, is_multiinput, is_multioutput)

            # Register the function
            if name != "__call__":
                self._set_function(name, func)
            else:
                self.run_function = func

    def _call_tf1_graph(self, method, inputs, **parameters):
        """Call a certain function on the current graph. Used for tensorflow 1

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

        # Make sure
        if len(output_nodes) == 1:
            output_nodes = output_nodes[0]

        # Make the input field dictionary
        if len(input_nodes) == 1:
            feed_dict = {input_nodes[0]: inputs}
        else:
            feed_dict = dict(zip(input_nodes, inputs))

        # Run the method
        return self.sess.run(output_nodes, feed_dict=feed_dict)

    def _call_tf2_eager_function(self, method, is_multiinput, is_multioutput, inputs, **parameters):
        """Call a tf.function from the TF2 saved model object

        Args:
            method (str): Method to run
            is_multiinput: Whether the function takes >1 inputs
            is_multioutput: Whether the function takes >1 outputs
            inputs: Any inputs for the method
        Returns:
            Outputs from the function invocation
        """

        # Wrap inputs in a tuple for a single-input function
        if not is_multiinput:
            inputs = (inputs,)

        # Run the function
        result = getattr(self.model, method)(*inputs, **parameters)

        # Unpack the outputs
        if not is_multioutput:
            return result.numpy()
        else:
            return [x.numpy() for x in result]

    def _run(self, inputs, **parameters):
        if tf.__version__ < '2':
            return self._call_tf1_graph('run', inputs, **parameters)
        else:
            return self.run_function(inputs, **parameters)
