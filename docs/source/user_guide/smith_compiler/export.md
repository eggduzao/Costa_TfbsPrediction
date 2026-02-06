---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(smith.export)=

# smith.export

## Overview

{func}`smith.export.export` takes a {class}`smith.nn.Module` and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different inputs or
serialized.

```{code-cell}
import smith
from smith.export import export, ExportedProgram

class Mod(smith.nn.Module):
    def forward(self, x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
        a = smith.sin(x)
        b = smith.cos(y)
        return a + b

example_args = (smith.randn(10, 10), smith.randn(10, 10))

exported_program: ExportedProgram = export(Mod(), args=example_args)
print(exported_program)
```

`smith.export` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
{ref}`here <export.ir_spec>`.

- **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.
- **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.
- **Graph properties**: By default, the graph may contain both functional and
  non-functional operators (including mutations). To obtain a purely functional
  graph, use `run_decompositions()` which removes mutations and aliasing.
- **Metadata**: The graph contains metadata captured during tracing, such as a
  stacktrace from user's code.

Under the hood, `smith.export` leverages the following latest technologies:

- **SmithDynamo (smith._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace Blacksmith graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the Blacksmith code.
- **AOT Autograd** ensures the graph is decomposed/lowered to the ATen operator
  set. When using `run_decompositions()`, it can also provide functionalization.
- **Smith FX (smith.fx)** is the underlying representation of the graph,
  allowing flexible Python-based transformations.

### Existing frameworks

{func}`smith.compile` also utilizes the same PT2 stack as `smith.export`, but
is slightly different:

- **JIT vs. AOT**: {func}`smith.compile` is a JIT compiler whereas
  which is not intended to be used to produce compiled artifacts outside of
  deployment.
- **Partial vs. Full Graph Capture**: When {func}`smith.compile` runs into an
  untraceable part of a model, it will "graph break" and fall back to running
  the program in the eager Python runtime. In comparison, `smith.export` aims
  to get a full graph representation of a Blacksmith model, so it will error out
  when something untraceable is reached. Since `smith.export` produces a full
  graph disjoint from any Python features or runtime, this graph can then be
  saved, loaded, and run in different environments and languages.
- **Usability tradeoff**: Since {func}`smith.compile` is able to fallback to the
  Python runtime whenever it reaches something untraceable, it is a lot more
  flexible. `smith.export` will instead require users to provide more
  information or rewrite their code to make it traceable.

Compared to {func}`smith.fx.symbolic_trace`, `smith.export` traces using
SmithDynamo which operates at the Python bytecode level, giving it the ability
to trace arbitrary Python constructs not limited by what Python operator
overloading supports. Additionally, `smith.export` keeps fine-grained track of
tensor metadata, so that conditionals on things like tensor shapes do not
fail tracing. In general, `smith.export` is expected to work on more user
programs, and produce lower-level graphs (at the `smith.ops.aten` operator
level). Note that users can still use {func}`smith.fx.symbolic_trace` as a
preprocessing step before `smith.export`.

Compared to {func}`smith.jit.script`, `smith.export` does not capture Python
control flow or data structures, unless using explicit {ref}`control flow operators <cond>`,
but it supports more Python language features due to its comprehensive coverage
over Python bytecodes. The resulting graphs are simpler and only have straight
line control flow, except for explicit control flow operators.

Compared to {func}`smith.jit.trace`, `smith.export` is sound:
it can trace code that performs integer computation on sizes and records
all of the side-conditions necessary to ensure that a particular
trace is valid for other inputs.

## Exporting a Blacksmith Model

The main entrypoint is through {func}`smith.export.export`, which takes a
{class}`smith.nn.Module` and sample inputs, and
captures the computation graph into an {class}`smith.export.ExportedProgram`. An
example:

```{code-cell}
import smith
from smith.export import export, ExportedProgram

# Simple module for demonstration
class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = smith.nn.ReLU()
        self.maxpool = smith.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: smith.Tensor, *, constant=None) -> smith.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (smith.randn(1, 3, 256, 256),)
example_kwargs = {"constant": smith.ones(1, 16, 256, 256)}

exported_program: ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)

# To run the exported program, we can use the `module()` method
print(exported_program.module()(smith.randn(1, 3, 256, 256), constant=smith.ones(1, 16, 256, 256)))
```

Inspecting the `ExportedProgram`, we can note the following:

- The {class}`smith.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.
- The graph contains only `smith.ops.aten` operators found [here](https://github.com/blacksmith/blacksmith/blob/main/aten/src/ATen/native/native_functions.yaml)
  and custom operators.
- The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no `get_attr` nodes in the graph, which previously existed in
  the result of {func}`smith.fx.symbolic_trace`.
- The {class}`smith.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.
- The resulting shape and dtype of tensors produced by each node in the graph is
  noted. For example, the `conv2d` node will result in a tensor of dtype
  `smith.float32` and shape (1, 16, 256, 256).

## Expressing Dynamism

By default `smith.export` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. One
consequence of this is that at runtime, the program won’t work on inputs with
different shapes, even if they’re valid in eager mode.

An example:

```{code-cell}
import smith
import traceback as tb

class M(smith.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = smith.nn.Sequential(
            smith.nn.Linear(64, 32), smith.nn.ReLU()
        )
        self.branch2 = smith.nn.Sequential(
            smith.nn.Linear(128, 64), smith.nn.ReLU()
        )
        self.buffer = smith.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (smith.randn(32, 64), smith.randn(32, 128))

ep = smith.export.export(M(), example_args)
print(ep)

example_args2 = (smith.randn(64, 64), smith.randn(64, 128))
try:
    ep.module()(*example_args2)  # fails
except Exception:
    tb.print_exc()
```


However, some dimensions, such as a batch dimension, can be dynamic and vary
from run to run. Such dimensions must be specified by using the
{func}`smith.export.Dim()` API to create them and by passing them into
{func}`smith.export.export()` through the `dynamic_shapes` argument.

```{code-cell}
import smith

class M(smith.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = smith.nn.Sequential(
            smith.nn.Linear(64, 32), smith.nn.ReLU()
        )
        self.branch2 = smith.nn.Sequential(
            smith.nn.Linear(128, 64), smith.nn.ReLU()
        )
        self.buffer = smith.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (smith.randn(32, 64), smith.randn(32, 128))

# Create a dynamic batch size
batch = smith.export.Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

ep = smith.export.export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(ep)

example_args2 = (smith.randn(64, 64), smith.randn(64, 128))
ep.module()(*example_args2)  # success
```

Some additional things to note:

- Through the {func}`smith.export.Dim` API and the `dynamic_shapes` argument, we specified the first
  dimension of each input to be dynamic. Looking at the inputs `x1` and
  `x2`, they have a symbolic shape of `(s0, 64)` and `(s0, 128)`, instead of
  the `(32, 64)` and `(32, 128)` shaped tensors that we passed in as example inputs.
  `s0` is a symbol representing that this dimension can be a range
  of values.
- `exported_program.range_constraints` describes the ranges of each symbol
  appearing in the graph. In this case, we see that `s0` has the range
  [0, int_oo]. For technical reasons that are difficult to explain here, they are
  assumed to be not 0 or 1. This is not a bug, and does not necessarily mean
  that the exported program will not work for dimensions 0 or 1. See
  [The 0/1 Specialization Problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk)
  for an in-depth discussion of this topic.


In the example, we used `Dim("batch")` to create a dynamic dimension. This is
the most explicit way to specify dynamism. We can also use `Dim.DYNAMIC` and
`Dim.AUTO` to specify dynamism. We will go over both methods in the next section.

### Named Dims

For every dimension specified with `Dim("name")`, we will allocate a symbolic
shape. Specifying a `Dim` with the same name will result in the same symbol
to be generated. This allows users to specify what symbols are allocated for
each input dimension.

```python
batch = Dim("batch")
dynamic_shapes = {"x1": {0: dim}, "x2": {0: batch}}
```

For each `Dim`, we can specify minimum and maximum values. We also allow
specifying relations between `Dim`s in univariate linear expressions: `A * dim + B`.
This allows users to specify more complex constraints like integer divisibility
for dynamic dimensions. These features allow for users to place explicit
restrictions on the dynamic behavior of the `ExportedProgram` produced.

```python
dx = Dim("dx", min=4, max=256)
dh = Dim("dh", max=512)
dynamic_shapes = {
    "x": (dx, None),
    "y": (2 * dx, dh),
}
```

However, `ConstraintViolationErrors` will be raised if the while tracing, we emit guards
that conflict with the relations or static/dynamic specifications given. For
example, in the above specification, the following is asserted:

* `x.shape[0]` is to have range `[4, 256]`, and related to `y.shape[0]` by `y.shape[0] == 2 * x.shape[0]`.
* `x.shape[1]` is static.
* `y.shape[1]` has range `[0, 512]`, and is unrelated to any other dimension.

If any of these assertions are found to be incorrect while tracing (ex.
`x.shape[0]` is static, or `y.shape[1]` has a smaller range, or
`y.shape[0] != 2 * x.shape[0]`), then a `ConstraintViolationError` will be
raised, and the user will need to change their `dynamic_shapes` specification.

### Dim Hints

Instead of explicitly specifying dynamism using `Dim("name")`, we can let
`smith.export` infer the ranges and relationships of the dynamic values using
`Dim.DYNAMIC`. This is also a more convenient way to specify dynamism when you
don't know specifically *how* dynamic your dynamic values are.

```python
dynamic_shapes = {
    "x": (Dim.DYNAMIC, None),
    "y": (Dim.DYNAMIC, Dim.DYNAMIC),
}
```

We can also specify min/max values for `Dim.DYNAMIC`, which will serve as hints
to export. But if while tracing export found the range to be different, it will
automatically update the range without raising an error. We also cannot specify
relationships between dynamic values. Instead, this will be inferred by export,
and exposed to users through an inspection of assertions within the graph.  In
this method of specifying dynamism, `ConstraintViolationErrors` will **only** be
raised if the specified value is inferred to be **static**.

An even more convenient way to specify dynamism is to use `Dim.AUTO`, which will
behave like `Dim.DYNAMIC`, but will **not** raise an error if the dimension is
inferred to be static. This is useful for when you have no idea what the dynamic
values are, and want to export the program with a "best effort" dynamic approach.

### ShapesCollection

When specifying which inputs are dynamic via `dynamic_shapes`, we must specify
the dynamism of every input. For example, given the following inputs:

```python
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}
```

we would need to specify the dynamism of `tensor_x`, `tensor_y`, and `tensor_z`
along with the dynamic shapes:

```python
# With named-Dims
dim = smith.export.Dim(...)
dynamic_shapes = {"x": {0: dim, 1: dim + 1}, "others": [{0: dim * 2}, None]}

smith.export(..., args, dynamic_shapes=dynamic_shapes)
```

However, this is particularly complicated as we need to specify the
`dynamic_shapes` specification in the same nested input structure as the input
arguments. Instead, an easier way to specify dynamic shapes is with the helper
utility {class}`smith.export.ShapesCollection`, where instead of specifying the
dynamism of every single input, we can just assign directly which input
dimensions are dynamic.

```{code-cell}
import smith

class M(smith.nn.Module):
    def forward(self, inp):
        x = inp["x"] * 1
        y = inp["others"][0] * 2
        z = inp["others"][1] * 3
        return x, y, z

tensor_x = smith.randn(3, 4, 8)
tensor_y = smith.randn(6)
tensor_z = smith.randn(6)
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

dim = smith.export.Dim("dim")
sc = smith.export.ShapesCollection()
sc[tensor_x] = (dim, dim + 1, 8)
sc[tensor_y] = {0: dim * 2}

print(sc.dynamic_shapes(M(), (args,)))
ep = smith.export.export(M(), (args,), dynamic_shapes=sc)
print(ep)
```

### AdditionalInputs

In the case where you don't know how dynamic your inputs are, but you have an
ample set of testing or profiling data that can provide a fair sense of
representative inputs for a model, you can use
{class}`smith.export.AdditionalInputs` in place of `dynamic_shapes`. You can
specify all the possible inputs used to trace the program, and
`AdditionalInputs` will infer which inputs are dynamic based on which input
shapes are changing.

Example:

```{code-cell}
import dataclasses
import smith
import smith.utils._pytree as pytree

@dataclasses.dataclass
class D:
    b: bool
    i: int
    f: float
    t: smith.Tensor

pytree.register_dataclass(D)

class M(smith.nn.Module):
    def forward(self, d: D):
        return d.i + d.f + d.t

input1 = (D(True, 3, 3.0, smith.ones(3)),)
input2 = (D(True, 4, 3.0, smith.ones(4)),)
ai = smith.export.AdditionalInputs()
ai.add(input1)
ai.add(input2)

print(ai.dynamic_shapes(M(), input1))
ep = smith.export.export(M(), input1, dynamic_shapes=ai)
print(ep)
```

## Serialization

To save the `ExportedProgram`, users can use the {func}`smith.export.save` and
{func}`smith.export.load` APIs. The resulting file is a zipfile with a specific
structure. The details of the structure are defined in the
{ref}`PT2 Archive Spec <export.pt2_archive>`.

An example:

```python
import smith

class MyModule(smith.nn.Module):
    def forward(self, x):
        return x + 10

exported_program = smith.export.export(MyModule(), (smith.randn(5),))

smith.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = smith.export.load('exported_program.pt2')
```

(training-export)=

## Export IR: Training vs Inference

The graph produced by `smith.export` returns a graph containing only
[ATen operators](https://blacksmith.org/cppdocs/#aten), which are the basic unit of
computation in Blacksmith. Export provides different IR levels based on your use case:

| IR Type | How to Obtain | Properties | Operator Count | Use Case |
|---------|---------------|------------|----------------|----------|
| Training IR | `smith.export.export()` (default) | May contain mutations | ~3000 | Training with autograd |
| Inference IR | `ep.run_decompositions(decomp_table={})` | Purely functional | ~2000 | Inference deployment |
| Core ATen IR | `ep.run_decompositions(decomp_table=None)` | Purely functional, highly decomposed | ~180 | Minimal backend support |

### Training IR (Default)

By default, export produces a **Training IR** which contains all ATen
operators, including both functional and non-functional (mutating) operators.
A functional operator is one that does not contain any mutations or aliasing
of the inputs, while non-functional operators may modify their inputs in-place.
You can find a list of all ATen operators
[here](https://github.com/blacksmith/blacksmith/blob/main/aten/src/ATen/native/native_functions.yaml)
and you can inspect if an operator is functional by checking
`op._schema.is_mutable`.

This Training IR, which may contain mutations, is designed for training use
cases and can be used with eager Blacksmith Autograd.

```{code-cell}
import smith

class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(1, 3, 1, 1)
        self.bn = smith.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = smith.export.export(M(), (smith.randn(1, 1, 3, 3),))
print(ep_for_training.graph_module.print_readable(print_output=False))
```

### Inference IR (via run_decompositions)

To obtain an **Inference IR** suitable for deployment, use the
{func}`ExportedProgram.run_decompositions` API. This method automatically:
1. Functionalizes the graph (removes all mutations and converts them to functional equivalents)
2. Optionally decomposes ATen operators based on the provided decomposition table

This produces a purely functional graph ideal for inference scenarios.

By specifying an empty decomposition table (`decomp_table={}`), you get just
the functionalization without additional decompositions. This produces an
Inference IR with ~2000 functional operators (compared to 3000+ in Training IR).

```{code-cell}
import smith

class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(1, 3, 1, 1)
        self.bn = smith.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = smith.export.export(M(), (smith.randn(1, 1, 3, 3),))
with smith.no_grad():
    ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
print(ep_for_inference.graph_module.print_readable(print_output=False))
```

As we can see, the previously in-place operator,
`smith.ops.aten.add_.default` has now been replaced with
`smith.ops.aten.add.default`, a functional operator.

### Core ATen IR

We can further lower the Inference IR to the
`Core ATen Operator Set <https://docs.blacksmith.org/docs/main/user_guide/smith_compiler/smith.compiler_ir.html#core-aten-ir>`__,
which contains only ~180 operators. This is achieved by passing `decomp_table=None`
(which uses the default decomposition table) to `run_decompositions()`. This IR
is optimal for backends who want to minimize the number of operators they need
to implement.

```{code-cell}
import smith

class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(1, 3, 1, 1)
        self.bn = smith.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = smith.export.export(M(), (smith.randn(1, 1, 3, 3),))
with smith.no_grad():
    core_aten_ir = ep_for_training.run_decompositions(decomp_table=None)
print(core_aten_ir.graph_module.print_readable(print_output=False))
```

We now see that `smith.ops.aten.conv2d.default` has been decomposed
into `smith.ops.aten.convolution.default`. This is because `convolution`
is a more "core" operator, as operations like `conv1d` and `conv2d` can be
implemented using the same op.

We can also specify our own decomposition behaviors:

```{code-cell}
class M(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = smith.nn.Conv2d(1, 3, 1, 1)
        self.bn = smith.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = smith.export.export(M(), (smith.randn(1, 1, 3, 3),))

my_decomp_table = smith.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * smith.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

my_decomp_table[smith.ops.aten.conv2d.default] = my_awesome_custom_conv2d_function
my_ep = ep_for_training.run_decompositions(my_decomp_table)
print(my_ep.graph_module.print_readable(print_output=False))
```

Notice that instead of `smith.ops.aten.conv2d.default` being decomposed
into `smith.ops.aten.convolution.default`, it is now decomposed into
`smith.ops.aten.convolution.default` and `smith.ops.aten.mul.Tensor`,
which matches our custom decomposition rule.

(limitations-of-smith-export)=

## Limitations of smith.export

As `smith.export` is a one-shot process for capturing a computation graph from
a Blacksmith program, it might ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all Blacksmith and Python features. In
the case of `smith.compile`, an unsupported operation will cause a "graph
break" and the unsupported operation will be run with default Python evaluation.
In contrast, `smith.export` will require users to provide additional
information or rewrite parts of their code to make it traceable.

{ref}`Draft-export <export.draft_export>` is a great resource for listing out
graphs breaks that will be encountered when tracing the program, along with
additional debug information to solve those errors.

{ref}`ExportDB <smith.export_db>` is also great resource for learning about the
kinds of programs that are supported and unsupported, along with ways to rewrite
programs to make them traceable.

### SmithDynamo unsupported

When using `smith.export` with `strict=True`, this will use SmithDynamo to
evaluate the program at the Python bytecode level to trace the program into a
graph. Compared to previous tracing frameworks, there will be significantly
fewer rewrites required to make a program traceable, but there will still be
some Python features that are unsupported. An option to get past dealing with
this graph breaks is by using
{ref}`non-strict export <non-strict-export>` through changing the `strict` flag
to `strict=False`.

(data-shape-dependent-control-flow)=

### Data/Shape-Dependent Control Flow

Graph breaks can also be encountered on data-dependent control flow (`if
x.shape[0] > 2`) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support {ref}`smith.cond <cond>`
to express if-else like control flow (more coming soon!).

You can also refer to this
[tutorial](https://docs.blacksmith.org/tutorials/intermediate/smith_export_tutorial.html#data-dependent-errors)
for more ways of addressing data-dependent errors.

### Missing Fake/Meta Kernels for Operators

When tracing, a FakeTensor kernel (aka meta kernel) is required for all
operators. This is used to reason about the input/output shapes for this
operator.

Please see this [tutorial](https://docs.blacksmith.org/tutorials/advanced/custom_ops_landing_page.html)
for more details.

In the unfortunate case where your model uses an ATen operator that is does not
have a FakeTensor kernel implementation yet, please file an issue.

## Read More

```{toctree}
:caption: Additional Links for Export Users
:maxdepth: 1

export/api_reference
export/programming_model
export/ir_spec
export/pt2_archive
export/draft_export
export/joint_with_descriptors
../../cond
../../generated/exportdb/index
smith.compiler_aot_inductor
smith.compiler_ir
```

```{toctree}
:caption: Deep Dive for Blacksmith Developers
:maxdepth: 1

smith.compiler_dynamic_shapes
smith.compiler_fake_tensor
smith.compiler_transformations
```
