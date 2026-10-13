# lambda of Eq. (2) on the validation worlds (val_0, val_1; OakD)

3216 frame-conditions per pipeline. Direction accuracy:

```
cond        clear  particle  phys10  phys20  phys5  phys50   all
pipeline                                                        
alg1-depth   39.2      31.9    41.4    39.2   21.8    39.2  35.4
alg1-rgb     52.6      21.5    29.3    45.5    9.1    50.9  34.8
fused-0      55.8      35.8    42.0    51.1   20.7    54.9  43.4
fused-10     55.8      36.0    42.0    51.1   20.7    54.9  43.4
fused-100    56.7      36.2    42.0    51.7   20.7    54.9  43.7
fused-1000   57.1      36.8    43.1    51.7   21.3    55.0  44.2
fused-30     56.2      36.2    42.0    51.1   20.7    54.9  43.5
fused-300    56.9      36.6    42.0    51.3   20.9    55.4  43.8
fused-3000   57.1      36.4    43.5    52.4   21.8    56.2  44.6
```

Chosen: **fused-3000** (highest overall accuracy among the fused pipelines; ties -> smaller lambda).
