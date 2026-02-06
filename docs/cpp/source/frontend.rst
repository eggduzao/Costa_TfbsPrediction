The C++ Frontend
================

The Blacksmith C++ frontend is a C++17 library for CPU and GPU
tensor computation, with automatic differentiation and high level building
blocks for state of the art machine learning applications.

Description
-----------

The Blacksmith C++ frontend can be thought of as a C++ version of the
Blacksmith Python frontend, providing automatic differentiation and various higher
level abstractions for machine learning and neural networks.  Specifically,
it consists of the following components:

+----------------------+------------------------------------------------------------------------+
| Component            | Description                                                            |
+======================+========================================================================+
| ``smith::Tensor``    | Automatically differentiable, efficient CPU and GPU enabled tensors    |
+----------------------+------------------------------------------------------------------------+
| ``smith::nn``        | A collection of composable modules for neural network modeling         |
+----------------------+------------------------------------------------------------------------+
| ``smith::optim``     | Optimization algorithms like SGD, Adam or RMSprop to train your models |
+----------------------+------------------------------------------------------------------------+
| ``smith::data``      | Datasets, data pipelines and multi-threaded, asynchronous data loader  |
+----------------------+------------------------------------------------------------------------+
| ``smith::serialize`` | A serialization API for storing and loading model checkpoints          |
+----------------------+------------------------------------------------------------------------+
| ``smith::python``    | Glue to bind your C++ models into Python                               |
+----------------------+------------------------------------------------------------------------+
| ``smith::jit``       | Pure C++ access to the SmithScript JIT compiler                        |
+----------------------+------------------------------------------------------------------------+

End-to-end example
------------------

Here is a simple, end-to-end example of defining and training a simple
neural network on the MNIST dataset:

.. code-block:: cpp

  #include <smith/smith.h>

  // Define a new Module.
  struct Net : smith::nn::Module {
    Net() {
      // Construct and register two Linear submodules.
      fc1 = register_module("fc1", smith::nn::Linear(784, 64));
      fc2 = register_module("fc2", smith::nn::Linear(64, 32));
      fc3 = register_module("fc3", smith::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    smith::Tensor forward(smith::Tensor x) {
      // Use one of many tensor manipulation functions.
      x = smith::relu(fc1->forward(x.reshape({x.size(0), 784})));
      x = smith::dropout(x, /*p=*/0.5, /*train=*/is_training());
      x = smith::relu(fc2->forward(x));
      x = smith::log_softmax(fc3->forward(x), /*dim=*/1);
      return x;
    }

    // Use one of many "standard library" modules.
    smith::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  };

  int main() {
    // Create a new Net.
    auto net = std::make_shared<Net>();

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = smith::data::make_data_loader(
        smith::data::datasets::MNIST("./data").map(
            smith::data::transforms::Stack<>()),
        /*batch_size=*/64);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    smith::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
      size_t batch_index = 0;
      // Iterate the data loader to yield batches from the dataset.
      for (auto& batch : *data_loader) {
        // Reset gradients.
        optimizer.zero_grad();
        // Execute the model on the input data.
        smith::Tensor prediction = net->forward(batch.data);
        // Compute a loss value to judge the prediction of our model.
        smith::Tensor loss = smith::nll_loss(prediction, batch.target);
        // Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward();
        // Update the parameters based on the calculated gradients.
        optimizer.step();
        // Output the loss and checkpoint every 100 batches.
        if (++batch_index % 100 == 0) {
          std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
          // Serialize your model periodically as a checkpoint.
          smith::save(net, "net.pt");
        }
      }
    }
  }

To see more complete examples of using the Blacksmith C++ frontend, see `the example repository
<https://github.com/blacksmith/examples/tree/master/cpp>`_.

Philosophy
----------

Blacksmith's C++ frontend was designed with the idea that the Python frontend is
great, and should be used when possible; but in some settings, performance and
portability requirements make the use of the Python interpreter infeasible. For
example, Python is a poor choice for low latency, high performance or
multithreaded environments, such as video games or production servers.  The
goal of the C++ frontend is to address these use cases, while not sacrificing
the user experience of the Python frontend.

As such, the C++ frontend has been written with a few philosophical goals in mind:

* **Closely model the Python frontend in its design**, naming, conventions and
  functionality.  While there may be occasional differences between the two
  frontends (e.g., where we have dropped deprecated features or fixed "warts"
  in the Python frontend), we guarantee that the effort in porting a Python model
  to C++ should lie exclusively in **translating language features**,
  not modifying functionality or behavior.

* **Prioritize flexibility and user-friendliness over micro-optimization.**
  In C++, you can often get optimal code, but at the cost of an extremely
  unfriendly user experience.  Flexibility and dynamism is at the heart of
  Blacksmith, and the C++ frontend seeks to preserve this experience, in some
  cases sacrificing performance (or "hiding" performance knobs) to keep APIs
  simple and explicable.  We want researchers who don't write C++ for a living
  to be able to use our APIs.

A word of warning: Python is not necessarily slower than
C++! The Python frontend calls into C++ for almost anything computationally expensive
(especially any kind of numeric operation), and these operations will take up
the bulk of time spent in a program.  If you would prefer to write Python,
and can afford to write Python, we recommend using the Python interface to
Blacksmith. However, if you would prefer to write C++, or need to write C++
(because of multithreading, latency or deployment requirements), the
C++ frontend to Blacksmith provides an API that is approximately as convenient,
flexible, friendly and intuitive as its Python counterpart. The two frontends
serve different use cases, work hand in hand, and neither is meant to
unconditionally replace the other.

Installation
------------

Instructions on how to install the C++ frontend library distribution, including
an example for how to build a minimal application depending on LibSmith, may be
found by following `this <https://blacksmith.org/cppdocs/installing.html>`_ link.
