import abc
import pickle as pkl
from sklearn.externals import joblib


class BaseHRModel:
    """Represents a single link a machine learning model workflow

    LW19Jan18: Consider how to document what the requirements for each `run` operation
    """

    @abc.abstractmethod
    def run(self, input):
        """Abstract function which takes a list of entries as input,
        performs some transformation on these entries and returns a list
        as output

        LW 18Jan18: Should we make this API support taking a generator and
        passing passing back an iterable? Would some kind of streaming
        infrastructure make sense?

        Args:
            input - [dict], entries to process (unserialized from JSON)
        Returns:
            output - [dict], processed entries
            """

        pass


class ScikitLearnModel(BaseHRModel):
    """Class for running a scikit-learn model"""

    def __init__(self, model_path, serialization_method='pickle'):
        """Initialize this step

        Args:
            model_path - str, path to a single pickle file containing the model
            """
        if serialization_method == "pickle":
            self.model = pkl.load(open(model_path, 'rb'))
        elif serialization_method == "joblib":
            self.model = joblib.load(model_path)
        else:
            raise Exception('Unknown serialization method: ' + serialization_method)

    def run(self, inputs):
        """Run the scikit learn model"""

        # Get the features
        X = [x['features'] for x in inputs]
        predictions = self.model.predict(X)
        
        # Add the predictions to the input, return new object
        for i, p in zip(inputs, predictions):
            i['prediction'] = p.tolist()
        return inputs
