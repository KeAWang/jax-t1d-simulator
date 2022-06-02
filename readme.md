Try out the speedups from using Jax:
```
python simglucose/speed_test.py
```

On my system I get:
```
JAX Time elapsed per ODE step: 1.2874603271484375e-05
Numpy Time elapsed per ODE step: 0.025090458393096923
Speedup factor: 1948.83
JAX Time elapsed per vmapped ODE step (100 patients): 0.0011016869544982911
Vmap speedup over sequential looping:  1.17
```
