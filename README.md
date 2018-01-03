

# DLHub home_run

WARNING: Not really meant to be used yet...

A package of useful tools for loading various ML models. 

## Installation
```
pip install -e .
```

```python
import pandas as pd
from home_run.client import HomeRun

hr = HomeRun("http://localhost:5000")
```

## Examples

### Iris Example from SKLearn
```python
data = {
    "batch":False,
    "input":[[ 8.0,  9,  7,  30],
             [ 4.6,  3.1,  1.5,  0.2]]
}
res = hr.predict(data)
df = pd.DataFrame(res)
```

### MNIST Example from Keras
TBD


