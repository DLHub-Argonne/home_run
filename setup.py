from setuptools import setup

setup(
    name='home_run',
    version='0.0.1',
    packages=['home_run'],
    install_requires=[
        "keras",
        "numpy",
        "scipy",
        "scikit-learn"
    ],
    python_requires=">=3.6",
    license="Apache License, Version 2.0",
    url="https://github.com/blaiszik/home_run"
)