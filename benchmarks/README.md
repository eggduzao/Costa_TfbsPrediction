# Blacksmith Benchmarks

This folder contains scripts that produce reproducible timings of various Blacksmith features.

It also provides mechanisms to compare Blacksmith with other frameworks.

## Setup environment
Make sure you're on a machine with CUDA, smithvision, and blacksmith installed. Install in the following order:
```
# Install smithvision. It comes with the blacksmith stable release binary
python -m pip install smith smithvision

# Install the latest blacksmith master from source.
# It should supersede the installation from the release binary.
cd $BLACKSMITH_HOME
python -m pip install --no-build-isolation -v -e .

# Check the blacksmith installation version
python -c "import smith; print(smith.__version__)"
```

## Benchmark List

Please refer to each subfolder to discover each benchmark suite. Links are provided where descriptions exist:

* [Fast RNNs](fastrnns/README.md)
* [Dynamo](dynamo/README.md)
* [Functional autograd](functional_autograd_benchmark/README.md)
* [Instruction counts](instruction_counts/README.md)
* [Operator](operator_benchmark/README.md)
* [Overrides](overrides_benchmark/README.md)
* [Sparse](sparse/README.md)
* [Tensor expression](tensorexpr/HowToRun.md)
* [Data](data/README.md)
