Try out the speedups from using Jax:
```
python simglucose/speed_test.py
```

On my system I get:
```
JAX Time elapsed per ODE step: 5.838871002197266e-05
Numpy Time elapsed per ODE step: 0.04383730173110962
Speedup factor: 750.78
JAX Time elapsed per vmapped ODE step (100 patients): 0.0013469243049621582
Vmap speedup over sequential looping:  4.33
```
