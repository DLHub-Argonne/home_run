from dlhub_toolbox.models.servables.tensorflow import TensorFlowModel
from home_run.tensorflow import TensorFlowServable
from unittest import TestCase
import tensorflow as tf
import numpy as np
import shutil
import os

tf_export_path = os.path.join(os.path.dirname(__file__), 'tf-model')


class TestTensorFlow(TestCase):

    maxDiff = 4096

    def setUp(self):
        # Clear existing model
        if os.path.isdir(tf_export_path):
            shutil.rmtree(tf_export_path)

    def make_model(self):
        """Example used in the dlhub_toolbox"""

        tf.reset_default_graph()

        with tf.Session() as sess:

            # Make two simple graphs, both of which will be served by TF
            x = tf.placeholder('float', shape=(None, 3), name='Input')
            z = tf.placeholder('float', shape=(), name='Multiple')
            y = x + 1
            len_fun = tf.reduce_sum(y - x)  # Returns the number of elements in the array
            scale_mult = tf.multiply(z, x)

            # Create the tool for saving the model to disk
            builder = tf.saved_model.builder.SavedModelBuilder(tf_export_path)

            #  Make descriptions for the inputs and outputs
            x_desc = tf.saved_model.utils.build_tensor_info(x)
            y_desc = tf.saved_model.utils.build_tensor_info(y)
            z_desc = tf.saved_model.utils.build_tensor_info(z)
            len_fun_desc = tf.saved_model.utils.build_tensor_info(len_fun)
            scale_mult_desc = tf.saved_model.utils.build_tensor_info(scale_mult)

            #  Make a signature for the functions to be served
            func_sig = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': x_desc},
                outputs={'y': y_desc},
                method_name='run'
            )
            len_sig = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': x_desc},
                outputs={'len': len_fun_desc},
                method_name='length'
            )
            mult_sig = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': x_desc, 'z': z_desc},
                outputs={'scale_mult': scale_mult_desc},
                method_name='scalar_multiply'
            )

            #  Add the functions and the  state of the graph to the builder
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: func_sig,
                    'length': len_sig,
                    'scalar_multiply': mult_sig
                })

            #  Save the function
            builder.save()

    def test_tensorflow(self):
        self.make_model()

        # Make the model description
        metadata = TensorFlowModel.create_model(tf_export_path).set_name('tf').set_title('TF')

        # Make the servable
        model = TensorFlowServable(**metadata.to_dict())

        # Test it out
        self.assertTrue(np.isclose([[3, 4, 5]], model.run([[2, 3, 4]])).all())
        self.assertTrue(np.isclose([[4, 8, 12]],
                                   model.scalar_multiply(([[1, 2, 3]], 4))).all())
        self.assertAlmostEqual(3, model.length([[1, 2, 3]])[0])
