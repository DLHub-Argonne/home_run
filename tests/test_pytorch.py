from dlhub_sdk.models.servables.pytorch import PyTorchModel
from unittest import TestCase
import torch
import os

from home_run.pytorch import PyTorchServable


class TestPyTorch(TestCase):

    def test_pytorch(self):

        # Create and save a tensor
        x = torch.tensor([0, 1, 2, 3, 4])

        files = {'test': 'tensor.pt'}

        torch.save(x, files['test'])

        try:

            # Test the regressor via pickle
            model = PyTorchServable.create_model(files['test'])\
                .set_title('Example')
            model.set_name('example')
            servable = PyTorchServable(**model.to_dict())
            self.assertAlmostEqual(servable.run([[1]])[0], 0)

        finally:
            for f in files.values():
                os.unlink(f[1])
