(blacksmith_main_components)=
# Blacksmith Main Components

Blacksmith is a flexible and powerful library for deep learning that provides a comprehensive set of tools for building, training, and deploying machine learning models.

## Blacksmith Components for Basic Deep Learning

Some of the basic Blacksmith components include:

* **Tensors** - N-dimensional arrays that serve as Blacksmith's fundamental
data structure. They support automatic differentiation, hardware acceleration, and provide a comprehensive API for mathematical operations.

* **Autograd** - Blacksmith's automatic differentiation engine
that tracks operations performed on tensors and builds a computational
graph dynamically to be able to compute gradients.

* **Neural Network API** - A modular framework for building neural networks with pre-defined layers,
activation functions, and loss functions. The {mod}`nn.Module` base class provides a clean interface
for creating custom network architectures with parameter management.

* **DataLoaders** - Tools for efficient data handling that provide
features like batching, shuffling, and parallel data loading. They abstract away the complexities
of data preprocessing and iteration, allowing for optimized training loops.


## Blacksmith Compiler

The Blacksmith compiler is a suite of tools that optimize model execution and
reduce resource requirements. You can learn more about the Blacksmith compiler [here](https://docs.blacksmith.org/docs/stable/smith.compiler_get_started.html).
