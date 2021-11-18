from sklearn.linear_model import LinearRegression, LogisticRegression
from dlhub_sdk.models.servables.sklearn import ScikitLearnModel
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
import pickle as pkl
import numpy as np

from home_run.sklearn import ScikitLearnServable


def test_sklearn(tmpdir):
    # Make a test training set
    X = [[1], [2]]
    y = [0, 1]

    # Train a classifier and a regressor
    reg = LinearRegression().fit(X, y)
    clf = LogisticRegression().fit(X, y)

    # Save using joblib and pickle
    files = dict((f, str(tmpdir / (f + '.pkl'))) for f in ['reg_pkl', 'reg_jbl', 'clf_pkl'])

    # Save the files
    with open(files['reg_pkl'], 'wb') as fp:
        pkl.dump(reg, fp)
    with open(files['clf_pkl'], 'wb') as fp:
        pkl.dump(clf, fp)
    joblib.dump(reg, files['reg_jbl'])

    # Test the regressor via pickle
    model = ScikitLearnModel.create_model(files['reg_pkl'], 1, serialization_method='pickle')\
        .set_title('Example')
    model.set_name('example')
    servable = ScikitLearnServable(**model.to_dict())
    assert np.isclose(servable.run([[1]])[0], 0).all()

    # Test the regressor via joblib
    model = ScikitLearnModel.create_model(files['reg_jbl'], 1, serialization_method='joblib')\
        .set_title('Example')
    model.set_name('example')
    servable = ScikitLearnServable(**model.to_dict())
    assert np.isclose(servable.run([[1]])[0], 0).all()

    # Test the classifier
    model = ScikitLearnModel.create_model(files['clf_pkl'], 1, classes=['Yes', 'No'],
                                          serialization_method='pickle')\
        .set_title('Example')
    model.set_name('example')
    servable = ScikitLearnServable(**model.to_dict())
    assert np.isclose(servable.run([[1]])[0], clf.predict_proba([[1]])).all()
