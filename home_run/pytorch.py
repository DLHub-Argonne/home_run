"""Shim for PyTorch models"""

from home_run.base import BaseServable
import logging
import torch

logger = logging.getLogger(__name__)


class TorchServable(BaseServable):
    """Servable for Torch models"""

    def _build(self):
        # TODO (wardlt): Make map location configurable
        self.model = torch.load(self.dlhub['files']['model'], map_location='cpu')
        self.model.eval()

        # Get the types for the layers
        self.input_type = self.servable['methods']['run']['input']
        self.is_multiinput = self.input_type['type'] == 'tuple'
        if self.is_multiinput:
            logger.info('Loading a multi-input model')
            self.input_type = [x['item_type']['type'] for x in self.input_type['element_types']]
        else:
            logger.info('Loading a single-input model')
            self.input_type = self.input_type['item_type']['type']

        self.is_multioutput = self.servable['methods']['run']['output']['type'] == 'tuple'

    def _run(self, inputs, **parameters):
        if self.is_multiinput:
            inputs = [
                torch.tensor(i).to(getattr(torch, dt)) for i, dt in zip(inputs, self.input_type)
            ]
        else:
            inputs = [torch.tensor(inputs).to(getattr(torch, self.input_type))]

        # Make the tensors and shape them to the correct type
        outputs = self.model(*inputs, **parameters)

        if self.is_multioutput:
            return [o.detach().numpy() for o in outputs]
        else:
            return outputs.detach().numpy()
