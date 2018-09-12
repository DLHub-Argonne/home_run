from sklearn.linear_model import LinearRegression, LogisticRegression
from dlhub_toolbox.models.servables.sklearn import ScikitLearnModel
from sklearn.externals import joblib
from unittest import TestCase
from tempfile import mkstemp
import pickle as pkl
import numpy as np
import os

from home_run.sklearn import ScikitLearnServable


class TestScikitLearn(TestCase):

    def test_sklearn(self):
        # Make a test training set
        X = [[1], [2]]
        y = [0, 1]

        # Train a classifier and a regressor
        reg = LinearRegression().fit(X, y)
        clf = LogisticRegression().fit(X, y)

        # Save using joblib and pickle
        files = {
            'reg_pkl': mkstemp('.pkl'),
            'reg_jbl': mkstemp('.pkl'),
            'clf_pkl': mkstemp('.pkl')
        }

        # Close the tempfiles
        for f in files.values():
            os.close(f[0])

        try:
            # Save the files
            with open(files['reg_pkl'][1], 'wb') as fp:
                pkl.dump(reg, fp)
            with open(files['clf_pkl'][1], 'wb') as fp:
                pkl.dump(clf, fp)
            joblib.dump(reg, files['reg_jbl'][1])

            # Test the regressor via pickle
            model = ScikitLearnModel(files['reg_pkl'][1], 1,
                                     serialization_method='pickle').set_title('Example')
            model.set_name('example')
            servable = ScikitLearnServable(**model.to_dict())
            self.assertAlmostEqual(servable.run([[1]])[0], 0)

            # Test the regressor via joblib
            model = ScikitLearnModel(files['reg_jbl'][1], 1,
                                     serialization_method='joblib').set_title('Example')
            model.set_name('example')
            servable = ScikitLearnServable(**model.to_dict())
            self.assertAlmostEqual(servable.run([[1]])[0], 0)

            # Test the classifier
            model = ScikitLearnModel(files['clf_pkl'][1], 1, classes=['Yes', 'No'],
                                     serialization_method='pickle').set_title('Example')
            model.set_name('example')
            servable = ScikitLearnServable(**model.to_dict())
            self.assertTrue(np.isclose(servable.run([[1]]), clf.predict_proba([[1]])).all())

        finally:
            for f in files.values():
                os.unlink(f[1])
