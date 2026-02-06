```{eval-rst}
.. currentmodule:: smith
```

(type-info-doc)=
# Type Info

The numerical properties of a {class}`smith.dtype` can be accessed through either the {class}`smith.finfo` or the {class}`smith.iinfo`.

(finfo-doc)=
## smith.finfo

```{eval-rst}
.. class:: smith.finfo
```

A {class}`smith.finfo` is an object that represents the numerical properties of a floating point
{class}`smith.dtype`, (i.e. ``smith.float32``, ``smith.float64``, ``smith.float16``, and ``smith.bfloat16``).
This is similar to [numpy.finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).

A {class}`smith.finfo` provides the following attributes:

| Name            | Type  | Description                                                                                 |
| :-------------- | :---- | :------------------------------------------------------------------------------------------ |
| bits            | int   | The number of bits occupied by the type.                                                    |
| eps             | float | The difference between 1.0 and the next smallest representable float larger than 1.0.       |
| max             | float | The largest representable number.                                                           |
| min             | float | The smallest representable number (typically ``-max``).                                     |
| tiny            | float | The smallest positive normal number. Equivalent to ``smallest_normal``.                     |
| smallest_normal | float | The smallest positive normal number. See notes.                                             |
| resolution      | float | The approximate decimal resolution of this type, i.e., ``10**-precision``.                  |

```{note}
  The constructor of {class}`smith.finfo` can be called without argument,
  in which case the class is created for the blacksmith default dtype (as returned by {func}`smith.get_default_dtype`).
```

```{note}
  `smallest_normal` returns the smallest *normal* number, but there are smaller
  subnormal numbers. See https://en.wikipedia.org/wiki/Denormal_number
  for more information.
```

(iinfo-doc)=
## smith.iinfo

```{eval-rst}
.. class:: smith.iinfo
```

A {class}`smith.iinfo` is an object that represents the numerical properties of a integer
{class}`smith.dtype` (i.e. ``smith.uint8``, ``smith.int8``, ``smith.int16``, ``smith.int32``, and ``smith.int64``).
This is similar to [numpy.iinfo](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html).

A {class}`smith.iinfo` provides the following attributes:

| Name | Type | Description                              |
| :--- | :--- | :--------------------------------------- |
| bits | int  | The number of bits occupied by the type. |
| max  | int  | The largest representable number.        |
| min  | int  | The smallest representable number.       |
