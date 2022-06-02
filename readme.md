Try out the speedups from using Jax:
```
python simglucose/speed_test.py
```

On CPU I get:
```
JAX Time elapsed per ODE step: 1.2874603271484375e-05
Numpy Time elapsed per ODE step: 0.025090458393096923
Speedup factor: 1948.83
JAX Time elapsed per vmapped ODE step (100 patients): 0.0011016869544982911
Vmap speedup over sequential looping:  1.17
```

On a 1080ti GPU, I get:
```
JAX Time elapsed per ODE step: 4.86445426940918e-05
Numpy Time elapsed per ODE step: 0.04463413953781128
Speedup factor: 917.56
JAX Time elapsed per vmapped ODE step (100 patients): 0.0013061881065368653
Vmap speedup over sequential looping:  3.72
```

Thus GPU gives significant gains when vmapped over multiple patients as expected.
