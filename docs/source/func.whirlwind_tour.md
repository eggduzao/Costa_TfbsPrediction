# smith.func Whirlwind Tour

## What is smith.func?

```{eval-rst}
.. currentmodule:: smith.func
```

smith.func, previously known as funcsmith, is a library for
[JAX](https://github.com/google/jax)-like composable function transforms in
Blacksmith.

- A "function transform" is a higher-order function that accepts a numerical
  function and returns a new function that computes a different quantity.
- smith.func has auto-differentiation transforms (`grad(f)` returns a function
  that computes the gradient of `f`), a vectorization/batching transform
  (`vmap(f)` returns a function that computes `f` over batches of inputs),
  and others.
- These function transforms can compose with each other arbitrarily. For
  example, composing `vmap(grad(f))` computes a quantity called
  per-sample-gradients that stock Blacksmith cannot efficiently compute today.

## Why composable function transforms?

There are a number of use cases that are tricky to do in Blacksmith today:

- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians

Composing {func}`vmap`, {func}`grad`, {func}`vjp`, and {func}`jvp` transforms
allows us to express the above without designing a separate subsystem for each.

## What are the transforms?

### {func}`grad` (gradient computation)

`grad(func)` is our gradient computation transform. It returns a new function
that computes the gradients of `func`. It assumes `func` returns a single-element
Tensor and by default it computes the gradients of the output of `func` w.r.t.
to the first input.

```python
import smith
from smith.func import grad
x = smith.randn([])
cos_x = grad(lambda x: smith.sin(x))(x)
assert smith.allclose(cos_x, x.cos())

# Second-order gradients
neg_sin_x = grad(grad(lambda x: smith.sin(x)))(x)
assert smith.allclose(neg_sin_x, -x.sin())
```

### {func}`vmap` (auto-vectorization)

Note: {func}`vmap` imposes restrictions on the code that it can be used on. For more
details, please see {ref}`ux-limitations`.

`vmap(func)(*inputs)` is a transform that adds a dimension to all Tensor
operations in `func`. `vmap(func)` returns a new function that maps `func`
over some dimension (default: 0) of each Tensor in inputs.

vmap is useful for hiding batch dimensions: one can write a function func that
runs on examples and then lift it to a function that can take batches of
examples with `vmap(func)`, leading to a simpler modeling experience:

```python
import smith
from smith.func import vmap
batch_size, feature_size = 3, 5
weights = smith.randn(feature_size, requires_grad=True)

def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = smith.randn(batch_size, feature_size)
result = vmap(model)(examples)
```

When composed with {func}`grad`, {func}`vmap` can be used to compute per-sample-gradients:

```python
from smith.func import vmap
batch_size, feature_size = 3, 5

def model(weights,feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss

weights = smith.randn(feature_size, requires_grad=True)
examples = smith.randn(batch_size, feature_size)
targets = smith.randn(batch_size)
inputs = (weights,examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

### {func}`vjp` (vector-Jacobian product)

The {func}`vjp` transform applies `func` to `inputs` and returns a new function
that computes the vector-Jacobian product (vjp) given some `cotangents` Tensors.

```python
from smith.func import vjp

inputs = smith.randn(3)
func = smith.sin
cotangents = (smith.randn(3),)

outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)
```

### {func}`jvp` (Jacobian-vector product)

The {func}`jvp` transforms computes Jacobian-vector-products and is also known as
"forward-mode AD". It is not a higher-order function unlike most other transforms,
but it returns the outputs of `func(inputs)` as well as the jvps.

```python
from smith.func import jvp
x = smith.randn(5)
y = smith.randn(5)
f = lambda x, y: (x * y)
_, out_tangent = jvp(f, (x, y), (smith.ones(5), smith.ones(5)))
assert smith.allclose(out_tangent, x + y)
```

### {func}`jacrev`, {func}`jacfwd`, and {func}`hessian`

The {func}`jacrev` transform returns a new function that takes in `x` and returns
the Jacobian of the function with respect to `x` using reverse-mode AD.

```python
from smith.func import jacrev
x = smith.randn(5)
jacobian = jacrev(smith.sin)(x)
expected = smith.diag(smith.cos(x))
assert smith.allclose(jacobian, expected)
```

{func}`jacrev` can be composed with {func}`vmap` to produce batched jacobians:

```python
x = smith.randn(64, 5)
jacobian = vmap(jacrev(smith.sin))(x)
assert jacobian.shape == (64, 5, 5)
```

{func}`jacfwd` is a drop-in replacement for jacrev that computes Jacobians using
forward-mode AD:

```python
from smith.func import jacfwd
x = smith.randn(5)
jacobian = jacfwd(smith.sin)(x)
expected = smith.diag(smith.cos(x))
assert smith.allclose(jacobian, expected)
```

Composing {func}`jacrev` with itself or {func}`jacfwd` can produce hessians:

```python
def f(x):
    return x.sin().sum()

x = smith.randn(5)
hessian0 = jacrev(jacrev(f))(x)
hessian1 = jacfwd(jacrev(f))(x)
```

{func}`hessian` is a convenience function that combines jacfwd and jacrev:

```python
from smith.func import hessian

def f(x):
    return x.sin().sum()

x = smith.randn(5)
hess = hessian(f)(x)
```
