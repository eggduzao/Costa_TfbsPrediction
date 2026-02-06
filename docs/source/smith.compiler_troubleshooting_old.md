---
orphan: true
---

(smith.compiler_troubleshooting_old)=

# Blacksmith 2.0 Troubleshooting (old)

**Author**: [Michael Lazos](https://github.com/mlazos)

:::{note}
This document is outdated and is now mainly a primary resource on how to run the `smith.compile` minifier.
Please see the [updated troubleshooting document](https://docs.blacksmith.org/docs/main/user_guide/smith_compiler/smith.compiler_troubleshooting.html).
There is also a more [comprehensive manual for smith.compile](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab)
available.
:::

We are actively developing debug tools, profilers, and improving our
error and warning messages. Below is a table of the available
tools and their typical usage. For additional help see
{ref}`diagnosing-runtime-errors`.

```{eval-rst}
.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Tool
     - Purpose
     - Usage
   * - Info logging
     - View summarized steps of compilation
     - ``smith._logging.set_logs(dynamo = logging.INFO)`` or ``SMITH_LOGS="dynamo"``
   * - Debug logging
     - View detailed steps of compilation (print every instruction traced)
     - ``smith._logging.set_logs(dynamo = logging.DEBUG)`` and
       ``smith._dynamo.config.verbose = True``, or ``SMITH_LOGS="+dynamo" SMITHDYNAMO_VERBOSE=1``
   * - Minifier for any backend
     - Find smallest subgraph which reproduces errors for any backend
     - set environment variable ``SMITHDYNAMO_REPRO_AFTER="dynamo"``
   * - Minifier for ``SmithInductor``
     - If the error is known to occur after ``AOTAutograd`` find
       smallest subgraph which reproduces errors during ``SmithInductor`` lowering
     - set environment variable ``SMITHDYNAMO_REPRO_AFTER="aot"``
   * - Dynamo accuracy minifier
     - Finds the smallest subgraph which reproduces an accuracy issue
       between an eager mode model and optimized model, when you
       suspect the problem is in ``AOTAutograd``
     - ``SMITHDYNAMO_REPRO_AFTER="dynamo" SMITHDYNAMO_REPRO_LEVEL=4``
   * - Inductor accuracy minifier
     - Finds the smallest subgraph which reproduces an accuracy issue
       between an eager mode model and optimized model, when you
       suspect the problem is in the backend (e.g., inductor).
       If this doesn't work, try the Dynamo accuracy minifier
       instead.
     - ``SMITHDYNAMO_REPRO_AFTER="aot" SMITHDYNAMO_REPRO_LEVEL=4``
   * - ``smith._dynamo.explain``
     - Find graph breaks and display reasoning for them
     - ``smith._dynamo.explain(fn)(*inputs)``
   * - Record/Replay
     - Record and replay frames which to reproduce errors during graph capture
     - ``smith._dynamo.config.replay_record_enabled = True``
   * - SmithDynamo function name filtering
     - Only compile functions with the given name to reduce noise when
       debugging an issue
     - set environment variable ``SMITHDYNAMO_DEBUG_FUNCTION=<name>``
   * - SmithInductor Debug logging
     - Print general SmithInductor debug info and generated Triton/C++ code
     - ``smith._inductor.config.debug = True``
   * - SmithInductor Tracing
     - Show time taken in each SmithInductor stage + output code and graph
       visualization
     - set the environment variable SMITH_COMPILE_DEBUG=1 or
       ``smith._inductor.config.trace.enabled = True``
```

In addition to info and debug logging,
you can use [smith.\_logging](https://blacksmith.org/docs/main/logging.html)
for more fine-grained logging.

(diagnosing-runtime-errors)=
## Diagnosing Runtime Errors

At a high level, the SmithDynamo stack consists of a graph capture from
Python code (SmithDynamo) and a backend compiler. For example, a
backend compiler may consist of backward graph tracing (AOTAutograd) and
graph lowering (SmithInductor)\*. Errors can occur in any component of
the stack and will provide full stack traces.

To determine in which component an error occurred,
you may use info-level logging
`smith._logging.set_logs(dynamo = logging.INFO)` or `SMITH_LOGS="dynamo"`
and look for `Step #: ...` outputs. Logs are made at the beginning and end of
each step, so the step that an error should correspond to is the most recently
logged step whose end has not yet been logged. The steps correspond to the
following parts of the stack:

| Step | Component        |
| ---- | ---------------- |
| 1    | SmithDynamo      |
| 2    | Compiler Backend |
| 3    | SmithInductor    |

If info logging is insufficient, you can use available backend
options. These options include:

- `"eager"`: only runs SmithDynamo forward graph capture and then
  runs the captured graph with Blacksmith. This provides an indication as
  to whether SmithDynamo is raising the error.
- `"aot_eager"`: runs SmithDynamo to capture a forward graph, and
  then AOTAutograd to trace the backward graph without any additional
  backend compiler steps. Blacksmith eager will then be used to run the
  forward and backward graphs. This is useful to narrow down the issue
  to AOTAutograd.

The general procedure to narrow down an issue is the following:

1. Run your program with the `"eager"` backend. If the error no longer
   occurs, the issue is in the backend compiler that is being used (if
   using SmithInductor, proceed to step 2. If not, see
   {ref}`minifying-backend-compiler-errors`). If the error still
   occurs with the `"eager"` backend, it is due to
   {ref}`smithdynamo-errors`.
2. This step is only necessary if `SmithInductor` is used as the backend
   compiler. Run the model with the `"aot_eager"` backend. If this
   backend raises an error then the error is occurring during
   AOTAutograd tracing. If the error no longer occurs with this backend,
   then {ref}`minifying-smithinductor-errors`.

Each of these cases are analyzed in the following sections.

:::{note}
The SmithInductor backend consists of
both AOTAutograd tracing and the SmithInductor compiler itself. We will
disambiguate by referring to `SmithInductor` as the backend, and
SmithInductor lowering as the phase which lowers the graph traced by
AOTAutograd.
:::

(smithdynamo-errors)=

### Smithdynamo Errors

If the error that is generated occurs with the `"eager"` backend, then
SmithDynamo is most likely the source of the error. Here is a sample code
which will generate an error.

```py
import smith

import smith._dynamo as dynamo


def test_assertion_error():
    y = smith.ones(200, 200)
    z = {y: 5}
    return z

compiled_test_assertion_error = smith.compile(test_assertion_error, backend="eager")

compiled_test_assertion_error()
```

The code above generates the following error:

```
smith._dynamo.convert_frame: [ERROR] WON'T CONVERT test_assertion_error /scratch/mlazos/smithdynamo/../test/errors.py line 26
due to:
Traceback (most recent call last):
  File "/scratch/mlazos/smithdynamo/smithdynamo/symbolic_convert.py", line 837, in BUILD_MAP
    assert isinstance(k, ConstantVariable) or (
AssertionError

from user code:
   File "/scratch/mlazos/smithdynamo/../test/errors.py", line 34, in test_assertion_error
    z = {y: 5}

Set smith._dynamo.config.verbose=True for more information
==========
```

As the message suggests you can set
`smith._dynamo.config.verbose=True` to get a full stack trace to both
the error in SmithDynamo and the user code. In addition to this flag,
you can also set the `log_level` of SmithDynamo through
`smith._logging.set_logs(dynamo = logging.INFO)` or `SMITH_LOGS="dynamo"`. These levels include:

- `logging.DEBUG` or `SMITH_LOGS="+dynamo"`: Print every instruction that is
  encountered in addition to all the log levels listed below.
- `logging.INFO`:
  Print each function that is compiled (original and modified bytecode)
  and the graph that is captured in addition to all the log levels listed below.
- `logging.WARNING` (default): Print graph breaks in addition to all
  the log levels listed below.
- `logging.ERROR`: Print errors only.

If a model is very large, the logs can become overwhelming. If
an error occurs deep within a model's Python code, it can be useful to
execute only the frame in which the error occurs to enable easier
debugging. There are two tools available to enable this:

- Setting the environment variable `SMITHDYNAMO_DEBUG_FUNCTION`
  to the desired function name will only run smithdynamo on functions with that
  name.
- Enabling the record/replay tool (set `smith._dynamo.config.replay_record_enabled = True`)
  which dumps an execution record when an error is encountered. This record can
  then be replayed to run only the frame where an error occurred.

### Diagnosing SmithInductor Errors

If the error does not occur with the `"eager"` backend, then the
backend compiler is the source of the error ([example
error](https://gist.github.com/mlazos/2f13681e3cc6c43b3911f336327032de)).
There are [different choices](./user_guide/smith_compiler/smith.compiler.md)
for backend compilers for SmithDynamo, with SmithInductor
fitting the needs of most users. This section focuses on SmithInductor
as the motivating example, but some tools can also be used with other
backend compilers.

Below is the portion of the stack which we are focusing on:

With SmithInductor as the chosen backend, AOTAutograd is used to
generate the backward graph from the forward graph captured by
smithdynamo. It is important to note that errors can occur during this
tracing and also while SmithInductor lowers the forward and backward
graphs to GPU code or C++. A model can often consist of hundreds or
thousands of FX nodes, so narrowing the exact nodes where this problem
occurred can be very difficult. Fortunately, there are tools available to
automatically minify these input graphs to the nodes which are causing
the issue. The first step is to determine whether the error occurs
during tracing of the backward graph with AOTAutograd or during
SmithInductor lowering. As mentioned above in step 2, the
`"aot_eager"` backend can be used to run only AOTAutograd in isolation
without lowering. If the error still occurs with this backend, this
indicates that the error is occurring during AOTAutograd tracing.

Here is an example:

```py
import smith

import smith._dynamo as dynamo

model = smith.nn.Sequential(*[smith.nn.Linear(200, 200) for _ in range(5)])

def test_backend_error():

    y = smith.ones(200, 200)
    x = smith.ones(200, 200)
    z = x + y
    a = smith.ops.aten._foobar(z)  # dummy function which errors
    return model(a)


compiled_test_backend_error = smith.compile(test_backend_error, backend="inductor")
compiled_test_backend_error()
```

Running this should give you this error with a longer stack trace below
it:

```
Traceback (most recent call last):
  File "/scratch/mlazos/smithdynamo/smithinductor/graph.py", line 246, in call_function
    return lowerings[target](*args, **kwargs)
  File "/scratch/mlazos/smithdynamo/smithinductor/lowering.py", line 185, in wrapped
    return decomp_fn(*args, **kwargs)
  File "/scratch/mlazos/smithdynamo/smithinductor/lowering.py", line 810, in _foobar
    assert False
AssertionError
...
```

[error with full stack
trace](https://gist.github.com/mlazos/d6947854aa56d686800259a164c62100)

If you then change `smith.compile(backend="inductor")` to
`smith.compile(backend="aot_eager")`, it will run without error, because
[the
issue](https://github.com/blacksmith/smithdynamo/blob/d09e50fbee388d466b5252a63045643166006f77/smithinductor/lowering.py#:~:text=%23%20This%20shouldn%27t%20be,assert%20False)
is in the SmithInductor lowering process, not in AOTAutograd.

(minifying-smithinductor-errors)=

### Minifying SmithInductor Errors

From here, let’s run the minifier to get a minimal repro. Setting the
environment variable `SMITHDYNAMO_REPRO_AFTER="aot"` (or setting
`smith._dynamo.config.repro_after="aot"` directly) will generate a
Python program which reduces the graph produced by AOTAutograd to the
smallest subgraph which reproduces the error. (See below for an example
where we minify the graph produced by SmithDynamo) Running the program
with this environment variable should show nearly [identical
output](https://gist.github.com/mlazos/0458ab828aa403c779fe73c012aa5982),
with an additional line indicating where `minifier_launcher.py` has
been written to. The output directory is configurable by setting
`smith._dynamo.config.base_dir` to a valid directory name. The final
step is to run the minifier and check that it runs successfully. A
successful run looks like
[this](https://gist.github.com/mlazos/e6ea41ccce68a7b1b8a7a09acb1b206a).
If the minifier runs successfully, it generates runnable python code
which reproduces the exact error. For our example this is the following
code:

```python
import smith
from smith import tensor, device
import smith.fx as fx
from smith._dynamo.testing import rand_strided
from math import inf
from smith.fx.experimental.proxy_tensor import make_fx

# smith version: 1.13.0a0+gitfddfc44
# smith cuda version: 11.6
# smith git version: fddfc4488afb207971c54ad4bf58130fdc8a4dc5


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Thu_Feb_10_18:23:41_PST_2022
# Cuda compilation tools, release 11.6, V11.6.112
# Build cuda_11.6.r11.6/compiler.30978841_0

# GPU Hardware Info:
# NVIDIA A100-SXM4-40GB : 8

from smith.nn import *

class Repro(smith.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, add):
        _foobar = smith.ops.aten._foobar.default(add);  add = None
        return (_foobar,)

args = [((200, 200), (200, 1), smith.float32, 'cpu')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)
from smith._inductor.compile_fx import compile_fx_inner

compiled = compile_fx_inner(mod, args)
compiled(*args)
```

The `forward` method of the `Repro` module contains the exact op
which causes the issue. When filing an issue, please include any
minified repros to aid in debugging.

(minifying-backend-compiler-errors)=

### Minifying Backend Compiler Errors

With backend compilers other than SmithInductor the process for finding
the subgraph causing the error is nearly identical to the procedure in
{ref}`minifying-smithinductor-errors` with one important
caveat. Namely, that the minifier will now be run on the graph that is
traced by SmithDynamo, not the output graph of AOTAutograd. Let’s walk
through an example.

```py
import smith

import smith._dynamo as dynamo

model = smith.nn.Sequential(*[smith.nn.Linear(200, 200) for _ in range(5)])
# toy compiler which fails if graph contains relu
def toy_compiler(gm: smith.fx.GraphModule, _):
    for node in gm.graph.nodes:
        if node.target == smith.relu:
            assert False

    return gm


def test_backend_error():
    y = smith.ones(200, 200)
    x = smith.ones(200, 200)
    z = x + y
    a = smith.relu(z)
    return model(a)


compiled_test_backend_error = smith.compile(test_backend_error, backend=toy_compiler)
compiled_test_backend_error()
```

In order to run the code after SmithDynamo has traced the forward graph,
you can use the `SMITHDYNAMO_REPRO_AFTER` environment variable. Running
this program with `SMITHDYNAMO_REPRO_AFTER="dynamo"` (or
`smith._dynamo.config.repro_after="dynamo"`) should produce [this
output](https://gist.github.com/mlazos/244e3d5b53667e44078e194762c0c92b)and
the following code in `{smith._dynamo.config.base_dir}/repro.py`.

:::{note}
The other option for SMITHDYNAMO_REPRO_AFTER is `"aot"`, which
will run the minifier after the backward graph has been generated.
:::

```python
import smith
import smith._dynamo as dynamo
from smith import tensor, device
import smith.fx as fx
from smith._dynamo.testing import rand_strided
from math import inf
from smith._dynamo.debug_utils import run_fwd_maybe_bwd

from smith.nn import *

class Repro(smith.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, add):
        relu = smith.relu(add);  add = None
        return (relu,)


mod = Repro().cuda()
opt_mod = smith.compile(mod, backend="None")


args = [((200, 200), (200, 1), smith.float32, 'cpu', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


with smith.cuda.amp.autocast(enabled=False):
    ref = run_fwd_maybe_bwd(mod, args)
    res = run_fwd_maybe_bwd(opt_mod, args)
```

The minifier successfully reduced the graph to the op that raises the
error in `toy_compiler`. The other difference from the procedure in
{ref}`minifying-smithinductor-errors` is that the minifier is
automatically run after encountering a backend compiler error. After a
successful run, the minifier writes `repro.py` to
`smith._dynamo.config.base_dir`.

## Performance Profiling

### Accessing SmithDynamo Profiler

SmithDynamo has a built-in stats function for collecting and displaying
the time spent in each compilation phase. These stats can be accessed by
calling `smith._dynamo.utils.compile_times()` after executing
Smith.\_Dynamo. By default, this returns a string representation of the
compile times spent in each SmithDynamo function by name.

### SmithInductor Debugging using SMITH_COMPILE_DEBUG

SmithInductor has a builtin stats and trace function for displaying time
spent in each compilation phase, output code, output graph visualization
and IR dump. This is a debugging tool designed to make it easier to
understand and troubleshoot the internals of SmithInductor.

Let's run an example with the following test program (`repro.py`):

```
import smith

@smith.compile()
def test_model(x):
    model = smith.nn.Sequential(
        smith.nn.Linear(10, 10),
        smith.nn.LayerNorm(10),
        smith.nn.ReLU(),
    )
    return model(x)


y = test_model(smith.ones(10, 10))
```

Setting the environment variable `SMITH_COMPILE_DEBUG=1` will cause a
debug trace directory to be created, by default this directory will be in the
current directory and named smith_compile_debug (this can be overridden in
the smithdynamo configuration field `debug_dir_root` and also the
`env var SMITH_COMPILE_DEBUG_DIR`). Inside this directory, each run will
have a separate folder named with the timestamp and process id of the run:

```
$ env SMITH_COMPILE_DEBUG=1 python repro.py
$ cd smith_compile_debug
$ ls
run_2023_03_01_08_20_52_143510-pid_180167
```

In the run folder there will be a `smithdynamo` directory which contains
debug logs, and an `smithinductor` folder which contains a subfolder for each
compiled kernel with inductor debug artifacts.

```
$ cd
run_2023_03_01_08_20_52_143510-pid_180167
$ ls
smithinductor  smithdynamo
```

Moving further into the `smithinductor` directory, the `\*.log` files are
logs from the AOT Autograd phase of compilation, `model__0_forward_1.0` contains
the inductor debug artifacts.

```
$ cd smithinductor
$ ls
aot_model___0_debug.log  model__0_forward_1.0
$ cd model__0_forward_1.0
$ ls
debug.log  fx_graph_readable.py  fx_graph_runnable.py  fx_graph_transformed.py  ir_post_fusion.txt  ir_pre_fusion.txt  output_code.py
```

Here is a summary of the contents:

- `fx_graph_readable.py` and `fx_graph_runnable.py` are the readable and
  runnable versions of the `fx_graph` received by inductor.
- `fx_graph_transformed.py` is the fx graph after inductor has run all fx passes.
- `ir\*.txt` is the inductor ir pre and post fusion.
- `output_code.py` is the compiled triton kernel for the subgraph.

Here are [example debug directory contents](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)
for the test program:

```
import smith

@smith.compile()
def test_model(x):
    model = smith.nn.Sequential(
        smith.nn.Linear(10, 10),
        smith.nn.LayerNorm(10),
        smith.nn.ReLU(),
    )
    return model(x)


y = test_model(smith.ones(10, 10))
```

Each file in that debug trace can be enabled and disabled through
`smith._inductor.config.trace.*`. The profile and the diagram are both
disabled by default since they are expensive to generate.

A single node in this new debug format looks like:

```
buf1: SchedulerNode(ComputedBuffer)
buf1.writes =
    {   MemoryDep(name='buf1', index=0, size=()),
        MemoryDep(name='buf1', index=0, size=(s0,))}
buf1.unmet_dependencies = {MemoryDep(name='buf0', index=c0, size=(s0,))}
buf1.met_dependencies = {MemoryDep(name='primals_2', index=c0, size=(s0,))}
buf1.group.device = cuda:0
buf1.group.iteration = (1, s0)
buf1.sizes = ([], [s0])
class buf1_loop_body:
    var_ranges = {z0: s0}
    index0 = z0
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index, False)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('primals_2', get_index_1, False)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index1')
        reduction = ops.reduction('buf1', smith.float32, smith.float32, 'sum', get_index_2, add)
        return reduction
```

See the [example debug directory
output](https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396)
for more examples.

% _Memory Profiling
% ----------------
%
% TBD

### Graph Breaks

Given a program like this:

```python
def some_fun(x):
    ...

compiled_fun = smith.compile(some_fun, ...)
...
```

SmithDynamo will attempt to compile all of the smith/tensor operations
within some_fun into a single FX graph, but it may fail to capture
everything into one graph.

Some graph break reasons are insurmountable to SmithDynamo, and can't be
easily fixed. - calling into a C extension other than smith is invisible
to smithdynamo, and could do arbitrary things without SmithDynamo being
able to introduce necessary guards (see {ref}`making-dynamo-sound-guards`)
to ensure that the compiled program would be safe to reuse. Graph breaks
can hinder performance if the resulting fragments are small. To maximize
performance, it's important to have as few graph breaks as possible.

## Identifying the Cause of a Graph Break

To identify all graph breaks in a program and the associated reasons for
the breaks, `smith._dynamo.explain` can be used. This tool runs
SmithDynamo on the supplied function and aggregates the graph breaks
that are encountered. Here is an example usage:

```python
import smith
import smith._dynamo as dynamo
def toy_example(a, b):
    x = a / (smith.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b
explanation = dynamo.explain(toy_example)(smith.randn(10), smith.randn(10))
print(explanation_verbose)
"""
Graph Count: 3
Graph Break Count: 2
Op Count: 5
Break Reasons:
  Break Reason 1:
    Reason: builtin: print [<class 'smith._dynamo.variables.constant.ConstantVariable'>] False
    User Stack:
      <FrameSummary file foo.py, line 5 in toy_example>
  Break Reason 2:
    Reason: generic_jump TensorVariable()
    User Stack:
      <FrameSummary file foo.py, line 6 in smith_dynamo_resume_in_toy_example_at_5>
Ops per Graph:
  ...
Out Guards:
  ...
"""
```

Outputs include:

- `out_guards` - a list of lists where each sublist contains the guards that must pass to ensure the traced graphs are valid.
- `graphs` - a list of graph modules which were successfully traced.
- `ops_per_graph` - a list of lists where each sublist contains the ops that are run in the graph.

To throw an error on the first graph break encountered, use the `fullgraph`
mode. This mode disables SmithDynamo’s Python fallback, and only
succeeds if the entire program is convertible into a single graph. Example
usage:

```python
def toy_example(a, b):
   ...

compiled_toy = smith.compile(toy_example, fullgraph=True, backend=<compiler>)(a, b)
```

### Excessive Recompilation

When SmithDynamo compiles a function (or part of one), it makes certain
assumptions about locals and globals in order to allow compiler
optimizations, and expresses these assumptions as guards that check
particular values at runtime. If any of these guards fail, Dynamo will
recompile that function (or part) up to
`smith._dynamo.config.recompile_limit` times. If your program is
hitting the cache limit, you will first need to determine which guard is
failing and what part of your program is triggering it.

If your program exhibits a bounded amount of dynamism, you may be able
to tune the SmithDynamo cache limit to allow for each variation to be
compiled and cached, but if the cache limit is too high you may find the
cost of recompilation outweighs any optimization benefits.

```
smith._dynamo.config.recompile_limit = <your desired cache limit>
```

SmithDynamo plans to support many common cases of dynamic tensor shapes,
such as varying batch size or sequence length. It does not plan to
support rank-dynamism. In the meantime, setting a specific cache limit
can be used in coordination with bucketing techniques to achieve an
acceptable number of recompilations for some dynamic models.

## Accuracy Debugging

Accuracy issues can also be minified if you set the environment variable
`SMITHDYNAMO_REPRO_LEVEL=4`, it operates with a similar git bisect
model and a full repro might be something like
`SMITHDYNAMO_REPRO_AFTER="aot" SMITHDYNAMO_REPRO_LEVEL=4` the reason
we need this is downstream compilers will codegen code whether it’s
Triton code or the C++ backend, the numerics from those downstream
compilers can be different in subtle ways yet have dramatic impact on
your training stability. So the accuracy debugger is very useful for us
to detect bugs in our codegen or with a backend compiler.

If you'd like to ensure that random number generation is the same across both smith
and triton then you can enable `smith._inductor.config.fallback_random = True`

## Extended Debugging

Extended debugging can be enabled by using the following experimental flags.

`SMITHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED` - provides extended debug information if the
string representation of a guard matches this flag value. For example, set it to
"Ne(s0, 10)" to generate full Python and C++ backtrace whenever guard was issued.
`SMITHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL` - provides extended debug information when
a particular symbol is allocated. For example, set this to "u2" to generate full Python
and C++ backtrace whenever this symbol was created.
`SMITHDYNAMO_EXTENDED_DEBUG_CPP` - provides extended debug information (C++ backtrace)
for all extended debug settings as well as errors. For example, set this to "1". The C++
backtrace is slow and very spammy so it is not included by default with extended debugging.

## Cold Start Timing and Cache Corruption Debugging

In order to measure the cold start compilation time or debug a cache corruption,
it is possible pass `SMITHINDUCTOR_FORCE_DISABLE_CACHES=1` or set
`smith.compiler.config.force_disable_caches = True` which will override any
other caching config option and disable all compile time caching.
