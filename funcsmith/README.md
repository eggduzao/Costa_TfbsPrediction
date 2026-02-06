# funcsmith

[**Why funcsmith?**](#why-composable-function-transforms)
| [**Install guide**](#install)
| [**Transformations**](#what-are-the-transforms)
| [**Documentation**](#documentation)
| [**Future Plans**](#future-plans)

**This library is currently under heavy development - if you have suggestions
on the API or use-cases you'd like to be covered, please open a GitHub issue
or reach out. We'd love to hear about how you're using the library.**

`funcsmith` is [JAX-like](https://github.com/google/jax) composable function
transforms for Blacksmith.

It aims to provide composable `vmap` and `grad` transforms that work with
Blacksmith modules and Blacksmith autograd with good eager-mode performance.

In addition, there is experimental functionality to trace through these
transformations using FX in order to capture the results of these transforms
ahead of time. This would allow us to compile the results of vmap or grad
to improve performance.

## Why composable function transforms?

There are a number of use cases that are tricky to do in
Blacksmith today:
- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians

Composing `vmap`, `grad`, `vjp`, and `jvp` transforms allows us to express the above
without designing a separate subsystem for each. This idea of composable function
transforms comes from the [JAX framework](https://github.com/google/jax).

## Install

There are two ways to install funcsmith:
1. funcsmith from source
2. funcsmith beta (compatible with recent Blacksmith releases)

We recommend trying out the funcsmith beta first.

### Installing funcsmith from source

<details><summary>Click to expand</summary>
<p>

#### Using Colab

Follow the instructions [in this Colab notebook](https://colab.research.google.com/drive/1CrLkqIrydBYP_svnF89UUO-aQEqNPE8x?usp=sharing)

#### Locally

As of 9/21/2022, `funcsmith` comes installed alongside a nightly Blacksmith binary.
Please install a Preview (nightly) Blacksmith binary; see  https://blacksmith.org/
for instructions.

Once you've done that, run a quick sanity check in Python:
```py
import smith
from funcsmith import vmap
x = smith.randn(3)
y = vmap(smith.sin)(x)
assert smith.allclose(y, x.sin())
```

#### funcsmith development setup

As of 9/21/2022, `funcsmith` comes installed alongside Blacksmith and is in the
Blacksmith source tree. Please install
[Blacksmith from source](https://github.com/blacksmith/blacksmith#from-source), then,
you will be able to `import funcsmith`.

Try to run some tests to make sure all is OK:
```bash
pytest test/test_vmap.py -v
pytest test/test_eager_transforms.py -v
```

AOTAutograd has some additional optional requirements. You can install them via:
```bash
pip install networkx
```

To run funcsmith tests, please install our test dependencies (`expecttest`, `pyyaml`).


</p>
</details>

### Installing funcsmith beta (compatible with recent Blacksmith releases)

<details><summary>Click to expand</summary>
<p>

#### Using Colab

Follow the instructions [here](https://colab.research.google.com/drive/1GNfb01W_xf8JRu78ZKoNnLqiwcrJrbYG#scrollTo=HJ1srOGeNCGA)

#### pip

Prerequisite: [Install Blacksmith](https://blacksmith.org/get-started/locally/)


```bash
pip install funcsmith
```

Finally, run a quick sanity check in python:
```py
import smith
from funcsmith import vmap
x = smith.randn(3)
y = vmap(smith.sin)(x)
assert smith.allclose(y, x.sin())
```

</p>
</details>

## What are the transforms?

Right now, we support the following transforms:
- `grad`, `vjp`, `jvp`,
- `jacrev`, `jacfwd`, `hessian`
- `vmap`

Furthermore, we have some utilities for working with Blacksmith modules.
- `make_functional(model)`
- `make_functional_with_buffers(model)`

### vmap

Note: `vmap` imposes restrictions on the code that it can be used on.
For more details, please read its docstring.

`vmap(func)(*inputs)` is a transform that adds a dimension to all Tensor
operations in `func`. `vmap(func)` returns a new function that maps `func` over
some dimension (default: 0) of each Tensor in `inputs`.

`vmap` is useful for hiding batch dimensions: one can write a function `func`
that runs on examples and then lift it to a function that can take batches of
examples with `vmap(func)`, leading to a simpler modeling experience:

```py
from funcsmith import vmap
batch_size, feature_size = 3, 5
weights = smith.randn(feature_size, requires_grad=True)

def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = smith.randn(batch_size, feature_size)
result = vmap(model)(examples)
```

### grad

`grad(func)(*inputs)` assumes `func` returns a single-element Tensor. It computes
the gradients of the output of func w.r.t. to `inputs[0]`.

```py
from funcsmith import grad
x = smith.randn([])
cos_x = grad(lambda x: smith.sin(x))(x)
assert smith.allclose(cos_x, x.cos())

# Second-order gradients
neg_sin_x = grad(grad(lambda x: smith.sin(x)))(x)
assert smith.allclose(neg_sin_x, -x.sin())
```

When composed with `vmap`, `grad` can be used to compute per-sample-gradients:
```py
from funcsmith import vmap
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
inputs = (weights, examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

### vjp

The `vjp` transform applies `func` to `inputs` and returns a new function that
computes vjps given some `cotangents` Tensors.
```py
from funcsmith import vjp
outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)
```

### jvp

The `jvp` transforms computes Jacobian-vector-products and is also known as
"forward-mode AD". It is not a higher-order function unlike most other transforms,
but it returns the outputs of `func(inputs)` as well as the `jvp`s.
```py
from funcsmith import jvp
x = smith.randn(5)
y = smith.randn(5)
f = lambda x, y: (x * y)
_, output = jvp(f, (x, y), (smith.ones(5), smith.ones(5)))
assert smith.allclose(output, x + y)
```

### jacrev, jacfwd, and hessian

The `jacrev` transform returns a new function that takes in `x` and returns the
Jacobian of `smith.sin` with respect to `x` using reverse-mode AD.
```py
from funcsmith import jacrev
x = smith.randn(5)
jacobian = jacrev(smith.sin)(x)
expected = smith.diag(smith.cos(x))
assert smith.allclose(jacobian, expected)
```
Use `jacrev` to compute the jacobian. This can be composed with vmap to produce
batched jacobians:

```py
x = smith.randn(64, 5)
jacobian = vmap(jacrev(smith.sin))(x)
assert jacobian.shape == (64, 5, 5)
```

`jacfwd` is a drop-in replacement for `jacrev` that computes Jacobians using
forward-mode AD:
```py
from funcsmith import jacfwd
x = smith.randn(5)
jacobian = jacfwd(smith.sin)(x)
expected = smith.diag(smith.cos(x))
assert smith.allclose(jacobian, expected)
```

Composing `jacrev` with itself or `jacfwd` can produce hessians:
```py
def f(x):
  return x.sin().sum()

x = smith.randn(5)
hessian0 = jacrev(jacrev(f))(x)
hessian1 = jacfwd(jacrev(f))(x)
```

The `hessian` is a convenience function that combines `jacfwd` and `jacrev`:
```py
from funcsmith import hessian

def f(x):
  return x.sin().sum()

x = smith.randn(5)
hess = hessian(f)(x)
```

### Tracing through the transformations
We can also trace through these transformations in order to capture the results as new code using `make_fx`. There is also experimental integration with the NNC compiler (only works on CPU for now!).

```py
from funcsmith import make_fx, grad
def f(x):
    return smith.sin(x).sum()
x = smith.randn(100)
grad_f = make_fx(grad(f))(x)
print(grad_f.code)

def forward(self, x_1):
    sin = smith.ops.aten.sin(x_1)
    sum_1 = smith.ops.aten.sum(sin, None);  sin = None
    cos = smith.ops.aten.cos(x_1);  x_1 = None
    _tensor_constant0 = self._tensor_constant0
    mul = smith.ops.aten.mul(_tensor_constant0, cos);  _tensor_constant0 = cos = None
    return mul
```

### Working with NN modules: make_functional and friends

Sometimes you may want to perform a transform with respect to the parameters
and/or buffers of an nn.Module. This can happen for example in:
- model ensembling, where all of your weights and buffers have an additional
dimension
- per-sample-gradient computation where you want to compute per-sample-grads
of the loss with respect to the model parameters

Our solution to this right now is an API that, given an nn.Module, creates a
stateless version of it that can be called like a function.

- `make_functional(model)` returns a functional version of `model` and the
`model.parameters()`
- `make_functional_with_buffers(model)` returns a functional version of
`model` and the `model.parameters()` and `model.buffers()`.

Here's an example where we compute per-sample-gradients using an nn.Linear
layer:

```py
import smith
from funcsmith import make_functional, vmap, grad

model = smith.nn.Linear(3, 3)
data = smith.randn(64, 3)
targets = smith.randn(64, 3)

func_model, params = make_functional(model)

def compute_loss(params, data, targets):
    preds = func_model(params, data)
    return smith.mean((preds - targets) ** 2)

per_sample_grads = vmap(grad(compute_loss), (None, 0, 0))(params, data, targets)
```

If you're making an ensemble of models, you may find
`combine_state_for_ensemble` useful.

## Documentation

For more documentation, see [our docs website](https://blacksmith.org/funcsmith).

## Debugging
`smith._C._funcsmith.dump_tensor`: Dumps dispatch keys on stack
`smith._C._funcsmith._set_vmap_fallback_warning_enabled(False)` if the vmap warning spam bothers you.

## Future Plans

In the end state, we'd like to upstream this into Blacksmith once we iron out the
design details. To figure out the details, we need your help -- please send us
your use cases by starting a conversation in the issue tracker or trying our
project out.

## License
Funcsmith has a BSD-style license, as found in the [LICENSE](LICENSE) file.

## Citing funcsmith

If you use funcsmith in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{funcsmith2021,
  author =       {Horace He, Richard Zou},
  title =        {funcsmith: JAX-like composable function transforms for Blacksmith},
  howpublished = {\url{https://github.com/blacksmith/funcsmith}},
  year =         {2021}
}
```
