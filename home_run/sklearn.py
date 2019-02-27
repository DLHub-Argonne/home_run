from home_run.base import BaseServable
from sklearn.externals import joblib
import pickle as pkl
import logging
logger = logging.getLogger(__name__)


class ScikitLearnServable(BaseServable):
    """Class for running a scikit-learn model"""

    def _build(self):

        # Load in the model from disk
        serialization_method = self.servable['options']['serialization_method']
        model_path = self.dlhub['files']['model']
        if serialization_method == "pickle":
            self.model = pkl.load(open(model_path, 'rb'))
            logging.info('Loaded model using pickle from: '.format(model_path))
        elif serialization_method == "joblib":
            self.model = joblib.load(model_path)
            logging.info('Loaded model using joblib from: '.format(model_path))
        else:
            raise Exception('Unknown serialization method: ' + serialization_method)

        # Determine whether to call predict or predict_proba
        predict_method = self.servable['methods']['run']['method_details']['method_name']
        self.predict = getattr(self.model, predict_method)
        logging.info('Set predict method to invoke {}'.format(predict_method))

    def _run(self, inputs, **parameters):
        """Compute a prediction using an sklearn_model"""

        # Get the features
        predictions = self.predict(inputs, **parameters)

        # Add the predictions to the input, return new object
        return predictions
