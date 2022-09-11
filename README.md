# `helmhurts-python`
Solving the Helmholtz equation to model the amplitude field of wireless (in Python),
with Dirichlet conditions on the boundaries and some diffusive term for the walls.

## For example

The following floor plan:

![](test.png)

should be used as

```
$ python3 ./power_map.py test.png
f = 2.400 Ghz → λ = 0.1249m → k = 50.300
Computing ... done!
```

and gives the following output (spacial resolution: 1px → 2cm):

![](test_out.png)


## Sources

+ https://jasmcole.com/2014/08/25/helmhurts/ (the original article)
+ https://github.com/mwil/helmhurts
+ https://bthierry.pages.math.cnrs.fr/course-fem/projet/2017-2018/
+ https://dx.doi.org/10.1002/net.22116