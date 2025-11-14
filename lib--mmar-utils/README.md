# mmar-utils

Common pure/IO utilities for multi-modal architectures team.

## Installation

```bash
pip install mmar-utils
```

## parallel_map
Mix of `joblib.Parallel` and `tqdm`.
### Similar libraries
#### joblib.Parallel: https://joblib.readthedocs.io/en/latest/parallel.html
- doesn't support showing progress out of the box

Syntax comparison ( assuming tqdm disabled )

```python
from math import sqrt, pow
from joblib import Parallel as P, delayed as d
from agi_med_utils import parallel_map

# ONE-ARG: THREADING
print(P(n_jobs=2)(map(d(sqrt), range(5))))
print(P(n_jobs=2)(d(sqrt)(i) for i in range(5)))
print(parallel_map(sqrt, range(5)))
# > [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# ONE-ARG: MULTIPROCESSING
print(P(n_jobs=2, backend='multiprocessing')(d(sqrt)(i) for i in range(3)))
print(parallel_map(sqrt, range(3), process=True))
# > [0.0, 1.0, 1.4142135623730951]

# MANY-ARGS: THREADING
pow_args = [(i, j) for i in range(1, 4) for j in range(1, 3)]
print(P(n_jobs=2)(d(pow)(i, j) for i, j in pow_args))
print(parallel_map(pow, pow_args, multiple_args=True))
# > [1.0, 1.0, 2.0, 4.0, 3.0, 9.0]

# KWARGS: THREADING
def ipow_kw(*, x, y):
    return 1 / pow(x, y)
ipow_kwargs = [{'x': i, 'y': j} for i, j in pow_args]
print(P(n_jobs=2)(d(ipow_kw)(**kw) for kw in ipow_kwargs))
print(parallel_map(ipow_kw, ipow_kwargs, kwargs_args=True))
# > [1.0, 1.0, 0.5, 0.25, 0.3333333333333333, 0.1111111111111111]

# KWARGS: MULTIPROCESSING
def ipow_kw_inc(*, x, y):
    return 1 / pow(x, y) + 1
print(P(n_jobs=2, backend='multiprocessing')(d(ipow_kw_inc)(**kw) for kw in ipow_kwargs))
print(parallel_map(ipow_kw_inc, ipow_kwargs, kwargs_args=True, process=True))
# > [2.0, 2.0, 1.5, 1.25, 1.3333333333333333, 1.1111111111111112]
```
#### pqdm: https://pqdm.readthedocs.io/en/latest/usage.html
## trace_with
Decorator for function which executes some callback on each function call.

### Similar libraries
#### functrace: https://github.com/idanhazan/functrace
- does not remember start datetime of call, which is critical
- supports selection of parameters to remember
- `functrace.TraceResult` object is not serializable out of the box
