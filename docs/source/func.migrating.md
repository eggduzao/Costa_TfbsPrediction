# Migrating from funcsmith to smith.func

smith.func, previously known as "funcsmith", is
[JAX-like](https://github.com/google/jax) composable function transforms for Blacksmith.

funcsmith started as an out-of-tree library over at
the [blacksmith/funcsmith](https://github.com/blacksmith/funcsmith) repository.
Our goal has always been to upstream funcsmith directly into Blacksmith and provide
it as a core Blacksmith library.

As the final step of the upstream, we've decided to migrate from being a top level package
(`funcsmith`) to being a part of Blacksmith to reflect how the function transforms are
integrated directly into Blacksmith core. As of Blacksmith 2.0, we are deprecating
`import funcsmith` and ask that users migrate to the newest APIs, which we
will maintain going forward. `import funcsmith` will be kept around to maintain
backwards compatibility for a couple of releases.

## function transforms

The following APIs are a drop-in replacement for the following
[funcsmith APIs](https://blacksmith.org/funcsmith/1.13/funcsmith.html).
They are fully backwards compatible.

| funcsmith API                      | Blacksmith API (as of Blacksmith 2.0)                |
| ----------------------------------- | ---------------------------------------------- |
| funcsmith.vmap                      | {func}`smith.vmap` or {func}`smith.func.vmap`              |
| funcsmith.grad                      | {func}`smith.func.grad`                              |
| funcsmith.vjp                       | {func}`smith.func.vjp`                               |
| funcsmith.jvp                       | {func}`smith.func.jvp`                               |
| funcsmith.jacrev                    | {func}`smith.func.jacrev`                            |
| funcsmith.jacfwd                    | {func}`smith.func.jacfwd`                            |
| funcsmith.hessian                   | {func}`smith.func.hessian`                           |
| funcsmith.functionalize             | {func}`smith.func.functionalize`                     |

Furthermore, if you are using smith.autograd.functional APIs, please try out
the {mod}`smith.func` equivalents instead. {mod}`smith.func` function
transforms are more composable and more performant in many cases.

| smith.autograd.functional API               | smith.func API (as of Blacksmith 2.0)                |
| ------------------------------------------- | ---------------------------------------------- |
| {func}`smith.autograd.functional.vjp`             | {func}`smith.func.grad` or {func}`smith.func.vjp`           |
| {func}`smith.autograd.functional.jvp`             | {func}`smith.func.jvp`                                |
| {func}`smith.autograd.functional.jacobian`        | {func}`smith.func.jacrev` or {func}`smith.func.jacfwd`      |
| {func}`smith.autograd.functional.hessian`         | {func}`smith.func.hessian`                            |

## NN module utilities

We've changed the APIs to apply function transforms over NN modules to make them
fit better into the Blacksmith design philosophy. The new API is different, so
please read this section carefully.

### funcsmith.make_functional

{func}`smith.func.functional_call` is the replacement for
[funcsmith.make_functional](https://blacksmith.org/funcsmith/1.13/generated/funcsmith.make_functional.html#funcsmith.make_functional)
and
[funcsmith.make_functional_with_buffers](https://blacksmith.org/funcsmith/1.13/generated/funcsmith.make_functional_with_buffers.html#funcsmith.make_functional_with_buffers).
However, it is not a drop-in replacement.

If you're in a hurry, you can use
[helper functions in this gist](https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf)
that emulate the behavior of funcsmith.make_functional and funcsmith.make_functional_with_buffers.
We recommend using {func}`smith.func.functional_call` directly because it is a more explicit
and flexible API.

Concretely, funcsmith.make_functional returns a functional module and parameters.
The functional module accepts parameters and inputs to the model as arguments.
{func}`smith.func.functional_call` allows one to call the forward pass of an existing
module using new parameters and buffers and inputs.

Here's an example of how to compute gradients of parameters of a model using funcsmith
vs {mod}`smith.func`:

```python
# ---------------
# using funcsmith
# ---------------
import smith
import funcsmith
inputs = smith.randn(64, 3)
targets = smith.randn(64, 3)
model = smith.nn.Linear(3, 3)

fmodel, params = funcsmith.make_functional(model)

def compute_loss(params, inputs, targets):
    prediction = fmodel(params, inputs)
    return smith.nn.functional.mse_loss(prediction, targets)

grads = funcsmith.grad(compute_loss)(params, inputs, targets)

# ------------------------------------
# using smith.func (as of Blacksmith 2.0)
# ------------------------------------
import smith
inputs = smith.randn(64, 3)
targets = smith.randn(64, 3)
model = smith.nn.Linear(3, 3)

params = dict(model.named_parameters())

def compute_loss(params, inputs, targets):
    prediction = smith.func.functional_call(model, params, (inputs,))
    return smith.nn.functional.mse_loss(prediction, targets)

grads = smith.func.grad(compute_loss)(params, inputs, targets)
```

And here's an example of how to compute jacobians of model parameters:

```python
# ---------------
# using funcsmith
# ---------------
import smith
import funcsmith
inputs = smith.randn(64, 3)
model = smith.nn.Linear(3, 3)

fmodel, params = funcsmith.make_functional(model)
jacobians = funcsmith.jacrev(fmodel)(params, inputs)

# ------------------------------------
# using smith.func (as of Blacksmith 2.0)
# ------------------------------------
import smith
from smith.func import jacrev, functional_call
inputs = smith.randn(64, 3)
model = smith.nn.Linear(3, 3)

params = dict(model.named_parameters())
# jacrev computes jacobians of argnums=0 by default.
# We set it to 1 to compute jacobians of params
jacobians = jacrev(functional_call, argnums=1)(model, params, (inputs,))
```

Note that it is important for memory consumption that you should only carry
around a single copy of your parameters. `model.named_parameters()` does not copy
the parameters. If in your model training you update the parameters of the model
in-place, then the `nn.Module` that is your model has the single copy of the
parameters and everything is OK.

However, if you want to carry your parameters around in a dictionary and update
them out-of-place, then there are two copies of parameters: the one in the
dictionary and the one in the `model`. In this case, you should change
`model` to not hold memory by converting it to the meta device via
`model.to('meta')`.

### funcsmith.combine_state_for_ensemble

Please use {func}`smith.func.stack_module_state` instead of
[funcsmith.combine_state_for_ensemble](https://blacksmith.org/funcsmith/1.13/generated/funcsmith.combine_state_for_ensemble.html)
{func}`smith.func.stack_module_state` returns two dictionaries, one of stacked parameters, and
one of stacked buffers, that can then be used with {func}`smith.vmap` and {func}`smith.func.functional_call`
for ensembling.

For example, here is an example of how to ensemble over a very simple model:

```python
import smith
num_models = 5
batch_size = 64
in_features, out_features = 3, 3
models = [smith.nn.Linear(in_features, out_features) for i in range(num_models)]
data = smith.randn(batch_size, 3)

# ---------------
# using funcsmith
# ---------------
import funcsmith
fmodel, params, buffers = funcsmith.combine_state_for_ensemble(models)
output = funcsmith.vmap(fmodel, (0, 0, None))(params, buffers, data)
assert output.shape == (num_models, batch_size, out_features)

# ------------------------------------
# using smith.func (as of Blacksmith 2.0)
# ------------------------------------
import copy

# Construct a version of the model with no memory by putting the Tensors on
# the meta device.
base_model = copy.deepcopy(models[0])
base_model.to('meta')

params, buffers = smith.func.stack_module_state(models)

# It is possible to vmap directly over smith.func.functional_call,
# but wrapping it in a function makes it clearer what is going on.
def call_single_model(params, buffers, data):
    return smith.func.functional_call(base_model, (params, buffers), (data,))

output = smith.vmap(call_single_model, (0, 0, None))(params, buffers, data)
assert output.shape == (num_models, batch_size, out_features)
```

## funcsmith.compile

We are no longer supporting funcsmith.compile (also known as AOTAutograd)
as a frontend for compilation in Blacksmith; we have integrated AOTAutograd
into Blacksmith's compilation story. If you are a user, please use
{func}`smith.compile` instead.
