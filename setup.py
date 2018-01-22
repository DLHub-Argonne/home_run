from setuptools import setup

setup(
    name='home_run',
    version='0.0.1',
    packages=['home_run'],
    install_requires=[
        "keras",
        "numpy",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "h5py"
    ],
    python_requires=">=3.6",
    license="Apache License, Version 2.0",
    url="https://github.com/Argonne-DLHub/home_run"
)