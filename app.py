"""Application file called by DLHub"""
from home_run import create_servable
import json


def run():
    """Creates the servable object"""

    with open('.dlhub/config') as fp:
        return create_servable(json.load(fp))
