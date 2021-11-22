from dlhub_sdk.models.servables.tensorflow import TensorFlowModel
from home_run.tensorflow import TensorFlowServable
import tensorflow as tf
import numpy as np


def _make_model_v1(tf_export_path):
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
        multiout_sig = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_desc, 'z': z_desc},
            outputs={'scale_mult': scale_mult_desc, 'y': y_desc},
            method_name='scalar_multiply'
        )

        #  Add the functions and the  state of the graph to the builder
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: func_sig,
                'length': len_sig,
                'scalar_multiply': mult_sig,
                'multioutput': multiout_sig
            })

        #  Save the function
        builder.save()


def _make_model_v2(tf_export_path):
    """Builds and saves a custom module"""
    class CustomModule(tf.Module):

        def __init__(self):
            super().__init__()
            self.m = tf.Variable([1.0, 1.0, 1.0], name='slope')

        @tf.function
        def __call__(self, x):
            y = self.m * x + 1
            return y

        @tf.function(input_signature=[tf.TensorSpec((None, 3), tf.float32)])
        def length(self, x):
            return tf.reduce_sum(self(x) - x, name='length')

        @tf.function(input_signature=[tf.TensorSpec((None, 3), tf.float32),
                                      tf.TensorSpec([], tf.float32)])
        def scalar_multiply(self, x, z):
            return tf.multiply(x, z, name='scale_mult')

        @tf.function(input_signature=[tf.TensorSpec((None, 3), tf.float32),
                                      tf.TensorSpec([], tf.float32)])
        def multioutput(self, x, z):
            return self.scalar_multiply(x, z), self(x)

    module = CustomModule()

    # Make a concrete version of __call__
    call = module.__call__.get_concrete_function(tf.TensorSpec((None, 3)))

    tf.saved_model.save(
        module, tf_export_path, signatures={
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: call,
            'length': module.length,
            'scalar_multiply': module.scalar_multiply,
            'multioutput': module.multioutput
        }
    )


def test_tensorflow(tmpdir):
    tf_export_path = str(tmpdir)
    if tf.__version__ < '2':
        _make_model_v1(tf_export_path)
    else:
        _make_model_v2(tf_export_path)

    # Make the model description
    metadata = TensorFlowModel.create_model(tf_export_path).set_name('tf').set_title('TF')

    # Make the servable
    model = TensorFlowServable(**metadata.to_dict())

    # Test it out
    print(model.run([[2, 3, 4]]))
    assert (1, 3) == np.shape(model.run([[2, 3, 4]])[0])
    assert np.isclose([[3, 4, 5]], model.run([[2, 3, 4]])[0]).all()
    assert np.isclose([[4, 8, 12]], model.scalar_multiply(([[1, 2, 3]], 4))[0]).all()
    assert np.isclose(3, model.length([[1, 2, 3]])[0]).all()

    # Test the multioutput
    output, _ = model.multioutput(([[1, 2, 3]], 4))
    assert isinstance(output, list)
    assert 2 == len(output)
    assert (1, 3) == np.shape(output[0])
    assert (1, 3) == np.shape(output[1])
