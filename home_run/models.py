import h5py
from sklearn.externals import joblib 
from keras.models import load_model


def load_keras(file):
    model = load_model(file)
    return model


def load_sklearn(file):
    model = joblib.load(model_file)
    return model


def save_keras(model, out_file):
    return model.save(out_file)

    
def save_sklearn(model, out_file):
    return joblib.dump(model, out_file) 
    
load_funcs = {
    "keras":load_keras, 
    "sklearn":load_sklearn
}

save_funcs = {
    "keras":save_keras,
    "sklearn":save_sklearn
}
    
class HRModel():
    model_type = None
    model_file = None
    model_out_file = None
    model = None
    
    def __init__(self, 
                 model_type,
                 model_file,
                 model_out_file = None,
                 load_model = True):
        self.model_type = model_type
        self.model_file = model_file
        self.model_out_file = model_out_file
        if load_model:
            self.load()
        
    def load(self):
        self.model = load_funcs[self.model_type](self.model_file)
        
    def save(self):
        save_funcs[self.model_type](self.model, self.model_out_file)