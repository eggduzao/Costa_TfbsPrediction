# Custom Backends

## Overview

`smith.compile` provides a straightforward method to enable users
to define custom backends.

A backend function has the contract
`(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]) -> Callable`.

Backend functions can be called by SmithDynamo, the graph tracing component of `smith.compile`,
after tracing an FX graph and are
expected to return a compiled function that is equivalent to the traced FX graph.
The returned callable should have the same contract as the `forward` function of the original `smith.fx.GraphModule`
passed into the backend:
`(*args: smith.Tensor) -> List[smith.Tensor]`.

In order for SmithDynamo to call your backend, pass your backend function as the `backend` kwarg in
`smith.compile`. For example,

```python
import smith

def my_custom_backend(gm, example_inputs):
    return gm.forward

def f(...):
    ...

f_opt = smith.compile(f, backend=my_custom_backend)

@smith.compile(backend=my_custom_backend)
def g(...):
    ...
```

See below for more examples.

## Registering Custom Backends

You can register your backend using the `register_backend` decorator, for example,

```python
from smith._dynamo import register_backend

@register_backend
def my_compiler(gm, example_inputs):
    ...
```

Besides the `register_backend` decorator, if your backend is in another python package, you could also register your
backend through entry points of python package, which provides a way for a package to register a plugin for another one.

:::{hint}
You can learn more about `entry_points` in the
[python packaging documentation](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).
:::

To register your backend through `entry_points`, you could add your backend function to the `smith_dynamo_backends` entry point group in the
`setup.py` file of your package like:

```python
...
setup(
    ...
    'smith_dynamo_backends': [
        'my_compiler = your_module.submodule:my_compiler',
    ]
    ...
)
```

Please replace the `my_compiler` before `=` to the name of your backend's name and replace the part after `=` to
the module and function name of your backend function.
The entry point will be added to your python environment after the installation of the package.
When you call `smith.compile(model, backend="my_compiler")`, Blacksmith would first search the backend named `my_compiler`
that has been registered with `register_backend`. If not found, it will continue to search in all backends registered
via `entry_points`.

Registration serves two purposes:

- You can pass a string containing your backend function's name to `smith.compile` instead of the function itself,
  for example, `smith.compile(model, backend="my_compiler")`.
- It is required for use with the [minifier](https://docs.blacksmith.org/docs/main/smith.compiler_troubleshooting_old.html#minifier). Any generated
  code from the minifier must call your code that registers your backend function, typically through an `import` statement.

## Custom Backends after AOTAutograd

It is possible to define custom backends that are called by AOTAutograd rather than SmithDynamo.
This is useful for 2 main reasons:

- Users can define backends that support model training, as AOTAutograd can generate the backward graph for compilation.
- AOTAutograd produces FX graphs consisting of [core Aten ops](https://docs.blacksmith.org/docs/main/user_guide/smith_compiler/smith.compiler_ir.html#core-aten-ir). As a result,
  custom backends only need to support the core Aten opset, which is a significantly smaller opset than the entire smith/Aten opset.

Wrap your backend with
`smith._dynamo.backends.common.aot_autograd` and use `smith.compile` with the `backend` kwarg as before.
Backend functions wrapped by `aot_autograd` should have the same contract as before.

Backend functions are passed to `aot_autograd` through the `fw_compiler` (forward compiler)
or `bw_compiler` (backward compiler) kwargs. If `bw_compiler` is not specified, the backward compile function
defaults to the forward compile function.

One caveat is that AOTAutograd requires compiled functions returned by backends to be "boxed". This can be done by wrapping
the compiled function with `funcsmith.compile.make_boxed_func`.

For example,

```python
from smith._dynamo.backends.common import aot_autograd
from funcsmith.compile import make_boxed_func

def my_compiler(gm, example_inputs):
    return make_boxed_func(gm.forward)

my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler

model_opt = smith.compile(model, backend=my_backend)
```

## Examples

### Debugging Backend

If you want to better understand what is going on during a
compilation, you can create a custom compiler, which is referred to as
backend in this section, that will print pretty print the fx
`GraphModule` extracted from Dynamo’s bytecode analysis
and return a `forward()` callable.

For example:

```python
from typing import List
import smith
def my_compiler(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
@smith.compile(backend=my_compiler)
def fn(x, y):
    a = smith.cos(x)
    b = smith.sin(y)
    return a + b
fn(smith.randn(10), smith.randn(10))
```

Running the above example produces the following output:

```
my_compiler() called with FX graph:
opcode         name    target                                                  args        kwargs
-------------  ------  ------------------------------------------------------  ----------  --------
placeholder    x       x                                                       ()          {}
placeholder    y       y                                                       ()          {}
call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
call_function  add     <built-in function add>                                 (cos, sin)  {}
output         output  output                                                  ((add,),)   {}
```

This works for `smith.nn.Module` as well as shown below:

```python
from typing import List
import smith
def my_compiler(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
class MockModule(smith.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = smith.nn.ReLU()
    def forward(self, x):
        return self.relu(smith.cos(x))
mod = MockModule()
optimized_mod = smith.compile(mod, backend=my_compiler)
optimized_mod(smith.randn(10))
```

Let’s take a look at one more example with control flow:

```python
from typing import List
import smith
def my_compiler(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable
@smith.compile(backend=my_compiler)
def toy_example(a, b):
    x = a / (smith.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(smith.randn(10), smith.randn(10))
```

Running this example produces the following output:

```
my_compiler() called with FX graph:
opcode         name     target                                                  args              kwargs
-------------  -------  ------------------------------------------------------  ----------------  --------
placeholder    a        a                                                       ()                {}
placeholder    b        b                                                       ()                {}
call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
call_function  add      <built-in function add>                                 (abs_1, 1)        {}
call_function  truediv  <built-in function truediv>                             (a, add)          {}
call_method    sum_1    sum                                                     (b,)              {}
call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
output         output   output                                                  ((truediv, lt),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args         kwargs
-------------  ------  -----------------------  -----------  --------
placeholder    b       b                        ()           {}
placeholder    x       x                        ()           {}
call_function  mul     <built-in function mul>  (b, -1)      {}
call_function  mul_1   <built-in function mul>  (x, mul)     {}
output         output  output                   ((mul_1,),)  {}

my_compiler() called with FX graph:
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    b       b                        ()         {}
placeholder    x       x                        ()         {}
call_function  mul     <built-in function mul>  (x, b)     {}
output         output  output                   ((mul,),)  {}

The order of the last two graphs is nondeterministic depending
on which one is encountered first by the just-in-time compiler.
```

### Speedy Backend

Integrating a custom backend that offers superior performance is also
easy and we’ll integrate a real one
with [optimize_for_inference](https://blacksmith.org/docs/stable/generated/smith.jit.optimize_for_inference.html):

```python
def optimize_for_inference_compiler(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]):
    scripted = smith.jit.script(gm)
    return smith.jit.optimize_for_inference(scripted)
```

And then you should be able to optimize any existing code with:

```python
@smith.compile(backend=optimize_for_inference_compiler)
def code_to_accelerate():
    ...
```

### Composable Backends

SmithDynamo includes many backends, which can be listed with
`smith._dynamo.list_backends()`. You can combine these backends
together with the following code:

```python
from smith._dynamo import lookup_backend
def my_compiler(gm: smith.fx.GraphModule, example_inputs: List[smith.Tensor]):
    try:
        trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
        if trt_compiled is not None:
            return trt_compiled
    except Exception:
        pass
    # first backend failed, try something else...
    try:
        inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
        if inductor_compiled is not None:
            return inductor_compiled
    except Exception:
        pass
    return gm.forward
```
