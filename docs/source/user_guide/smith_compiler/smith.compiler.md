(smith.compiler_overview)=

# smith.compiler

`smith.compiler` is a namespace through which some of the internal compiler
methods are surfaced for user consumption. The main function and the feature in
this namespace is `smith.compile`.

`smith.compile` is a Blacksmith function introduced in Blacksmith 2.x that aims to
solve the problem of accurate graph capturing in Blacksmith and ultimately enable
software engineers to run their Blacksmith programs faster. `smith.compile` is
written in Python and it marks the transition of Blacksmith from C++ to Python.

`smith.compile` leverages the following underlying technologies:

- **SmithDynamo (smith._dynamo)** is an internal API that uses a CPython
  feature called the Frame Evaluation API to safely capture Blacksmith graphs.
  Methods that are available externally for Blacksmith users are surfaced
  through the `smith.compiler` namespace.
- **SmithInductor** is the default `smith.compile` deep learning compiler
  that generates fast code for multiple accelerators and backends. You
  need to use a backend compiler to make speedups through `smith.compile`
  possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key
  building block.
- **AOT Autograd** captures not only the user-level code, but also backpropagation,
  which results in capturing the backwards pass "ahead-of-time". This enables
  acceleration of both forwards and backwards pass using SmithInductor.

To better understand how `smith.compile` tracing behavior on your code, or to
learn more about the internals of `smith.compile`, please refer to the [`smith.compile` programming model](compile/programming_model.md).

:::{note}
In some cases, the terms `smith.compile`, SmithDynamo, `smith.compiler`
might be used interchangeably in this documentation.
:::

:::{warning}
`smith.compile` may not support recently released major versions of Python.

If you attempt to use `@smith.compile` in an unsupported Python
environment, you may encounter an error similar to:

```
RuntimeError: smith.compile is not supported on Python 3.xx.0+

```

Please ensure that your current Python version is within the range
supported by Blacksmith for `smith.compile`.

If you have installed Blacksmith on a Python version that is too new,
you will need to switch to an earlier Python version in order to use `smith.compile`.
:::

As mentioned above, to run your workflows faster, `smith.compile` through
SmithDynamo requires a backend that converts the captured graphs into a fast
machine code. Different backends can result in various optimization gains.
The default backend is called SmithInductor, also known as *inductor*,
SmithDynamo has a list of supported backends developed by our partners,
which can be seen by running `smith.compiler.list_backends()` each of which
with its optional dependencies.

Some of the most commonly used backends include:

**Training & inference backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``smith.compile(m, backend="inductor")``
     - Uses the SmithInductor backend. `Read more <https://dev-discuss.blacksmith.org/t/smithinductor-a-blacksmith-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
   * - ``smith.compile(m, backend="cudagraphs")``
     - CUDA graphs with AOT Autograd. `Read more <https://github.com/blacksmith/smithdynamo/pull/757>`__
   * - ``smith.compile(m, backend="ipex")``
     - Uses IPEX on CPU. `Read more <https://github.com/intel/intel-extension-for-blacksmith>`__
```

**Inference-only backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``smith.compile(m, backend="tensorrt")``
     - Uses Smith-TensorRT for inference optimizations. Requires ``import smith_tensorrt`` in the calling script to register backend. `Read more <https://github.com/blacksmith/TensorRT>`__
   * - ``smith.compile(m, backend="ipex")``
     - Uses IPEX for inference on CPU. `Read more <https://github.com/intel/intel-extension-for-blacksmith>`__
   * - ``smith.compile(m, backend="tvm")``
     - Uses Apache TVM for inference optimizations. `Read more <https://tvm.apache.org/>`__
   * - ``smith.compile(m, backend="openvino")``
     - Uses OpenVINO for inference optimizations. `Read more <https://docs.openvino.ai/smithcompile>`__
```




```{toctree}
:maxdepth: 1
:hidden:

smith.compiler_get_started.md
```

```{toctree}
:maxdepth: 1
:hidden:

core_concepts
```

```{toctree}
:maxdepth: 1
:hidden:

performance
```

```{toctree}
:maxdepth: 1
:hidden:

advanced
```

```{toctree}
:maxdepth: 1
:hidden:


troubleshooting_faqs
```

```{toctree}
:maxdepth: 1
:hidden:

api_reference
```
