import requests
import json
class Homerun(object):

    def __init__(self, 
                target=None,
                 **kwargs):
        self.target = target
        return

    def predict(self, data):
        res = requests.post(self.target, json=data)
        return res.json()