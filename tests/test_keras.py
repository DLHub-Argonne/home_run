from dlhub_toolbox.models.servables.keras import KerasModel
from keras.models import Sequential
from keras.layers import Dense
from unittest import TestCase
from tempfile import mkdtemp
import numpy as np
import shutil
import os

from home_run.keras import KerasServable


class KerasTest(TestCase):

    def test_keras(self):
        # Make a Keras model
        model = Sequential()
        model.add(Dense(16, input_shape=(1,), activation='relu', name='hidden'))
        model.add(Dense(1, name='output'))
        model.compile(optimizer='rmsprop', loss='mse')

        # Save it
        tempdir = mkdtemp()
        try:
            model_path = os.path.join(tempdir, 'model.hd5')
            model.save(model_path)

            # Make the model
            metadata = KerasModel.create_model(model_path, ["y"])
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = np.array([[1]])
            self.assertAlmostEqual(model.predict(x)[0],
                                   servable.run(x)[0])

        finally:
            shutil.rmtree(tempdir)


