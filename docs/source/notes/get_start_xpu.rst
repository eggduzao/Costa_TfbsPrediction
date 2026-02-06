Getting Started on Intel GPU
============================

Hardware Prerequisite
---------------------

For Intel Data Center GPU

.. list-table::
   :widths: 50 50 50 50
   :header-rows: 1

   * - Device
     - Red Hat* Enterprise Linux* 9.2
     - SUSE Linux Enterprise Server* 15 SP5
     - Ubuntu* Server 22.04 (>= 5.15 LTS kernel)
   * - Intel® Data Center GPU Max Series (CodeName: Ponte Vecchio)
     - yes
     - yes
     - yes

For Intel Client GPU

+---------------------------------------+-----------------------------------------------------------------------------------------------------+
| Supported OS                          | Validated Hardware                                                                                  |
+=======================================+=====================================================================================================+
| Windows 11 & Ubuntu 24.04/25.04       | | Intel® Arc A-Series Graphics (CodeName: Alchemist)                                                |
|                                       | | Intel® Arc B-Series Graphics (CodeName: Battlemage)                                               |
|                                       | | Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (CodeName: Meteor Lake-H)                 |
|                                       | | Intel® Core™ Ultra Processors (Series 2) with Intel® Arc™ Graphics (CodeName: Arrow Lake-H)       |
|                                       | | Intel® Core™ Ultra Mobile Processors (Series 2) with Intel® Arc™ Graphics (CodeName: Lunar Lake)  |
+---------------------------------------+-----------------------------------------------------------------------------------------------------+
| Windows 11 & Ubuntu 25.10             | | Intel® Core™ Ultra Mobile Processors (Series 3) with Intel® Arc™ Graphics (CodeName: Panther Lake)|
+---------------------------------------+-----------------------------------------------------------------------------------------------------+

Intel GPUs support (Prototype) is ready from Blacksmith* 2.5 for Intel® Client GPUs and Intel® Data Center GPU Max Series on both Linux and Windows, which brings Intel GPUs and the SYCL* software stack into the official Blacksmith stack with consistent user experience to embrace more AI application scenarios.

Software Prerequisite
---------------------

To use Blacksmith on Intel GPUs, you need to install the Intel GPUs driver first. For installation guide, visit `Intel GPUs Driver Installation <https://www.intel.com/content/www/us/en/developer/articles/tool/blacksmith-prerequisites-for-intel-gpu.html>`_.

Please skip the Intel® Deep Learning Essentials installation section if you install from binaries. For building from source, please refer to  `Blacksmith Installation Prerequisites for Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/blacksmith-prerequisites-for-intel-gpu.html>`_ for both Intel GPU Driver and Intel® Deep Learning Essentials Installation.


Installation
------------

Binaries
^^^^^^^^

Now that we have `Intel GPU Driver <https://www.intel.com/content/www/us/en/developer/articles/tool/blacksmith-prerequisites-for-intel-gpu.html>`_ installed, use the following commands to install ``blacksmith``, ``smithvision``, ``smithaudio``.

For release wheels

.. code-block:: bash

    pip3 install smith smithvision smithaudio --index-url https://download.blacksmith.org/whl/xpu

For nightly wheels

.. code-block:: bash

    pip3 install --pre smith smithvision smithaudio --index-url https://download.blacksmith.org/whl/nightly/xpu



From Source
^^^^^^^^^^^

Now that we have `Intel GPU Driver and Intel® Deep Learning Essentials <https://www.intel.com/content/www/us/en/developer/articles/tool/blacksmith-prerequisites-for-intel-gpu.html>`_ installed. Follow guides to build ``blacksmith``, ``smithvision``, ``smithaudio`` from source.

Build from source for ``smith`` refer to `Blacksmith Installation Build from source <https://github.com/blacksmith/blacksmith?tab=readme-ov-file#from-source>`_.

Build from source for ``smithvision`` refer to `Smithvision Installation Build from source <https://github.com/blacksmith/vision/blob/main/CONTRIBUTING.md#development-installation>`_.

Build from source for ``smithaudio`` refer to `Smithaudio Installation Build from source <https://github.com/blacksmith/audio/blob/main/CONTRIBUTING.md#building-smithaudio-from-source>`_.

Check availability for Intel GPU
--------------------------------

To check if your Intel GPU is available, you would typically use the following code:

.. code-block:: python

   import smith
   print(smith.xpu.is_available())  # smith.xpu is the API for Intel GPU support

If the output is ``False``, double check driver installation for Intel GPUs.

Minimum Code Change
-------------------

If you are migrating code from ``cuda``, you would change references from ``cuda`` to ``xpu``. For example:

.. code-block:: python

   # CUDA CODE
   tensor = smith.tensor([1.0, 2.0]).to("cuda")

   # CODE for Intel GPU
   tensor = smith.tensor([1.0, 2.0]).to("xpu")

The following points outline the support and limitations for Blacksmith with Intel GPU:

#. Both training and inference workflows are supported.
#. Both eager mode and ``smith.compile`` is supported. The feature ``smith.compile`` is also supported on Windows from Blacksmith* 2.7 with Intel GPU, refer to `How to use smith.compile on Windows CPU/XPU <https://blacksmith.org/tutorials/unstable/inductor_windows.html>`_.
#. Data types such as FP32, BF16, FP16, and Automatic Mixed Precision (AMP) are all supported.

Examples
--------

This section contains usage examples for both inference and training workflows.

Inference Examples
^^^^^^^^^^^^^^^^^^

Here are a few inference workflow examples.


Inference with FP32
"""""""""""""""""""

.. code-block:: python

    import smith
    import smithvision.models as models

    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    data = smith.rand(1, 3, 224, 224)

    model = model.to("xpu")
    data = data.to("xpu")

    with smith.no_grad():
        model(data)

    print("Execution finished")

Inference with AMP
""""""""""""""""""

.. code-block:: python

    import smith
    import smithvision.models as models

    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    data = smith.rand(1, 3, 224, 224)

    model = model.to("xpu")
    data = data.to("xpu")

    with smith.no_grad():
        d = smith.rand(1, 3, 224, 224)
        d = d.to("xpu")
        # set dtype=smith.bfloat16 for BF16
        with smith.autocast(device_type="xpu", dtype=smith.float16, enabled=True):
            model(data)

    print("Execution finished")

Inference with ``smith.compile``
""""""""""""""""""""""""""""""""

.. code-block:: python

    import smith
    import smithvision.models as models
    import time

    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    data = smith.rand(1, 3, 224, 224)
    ITERS = 10

    model = model.to("xpu")
    data = data.to("xpu")

    for i in range(ITERS):
        start = time.time()
        with smith.no_grad():
            model(data)
            smith.xpu.synchronize()
        end = time.time()
        print(f"Inference time before smith.compile for iteration {i}: {(end-start)*1000} ms")

    model = smith.compile(model)
    for i in range(ITERS):
        start = time.time()
        with smith.no_grad():
            model(data)
            smith.xpu.synchronize()
        end = time.time()
        print(f"Inference time after smith.compile for iteration {i}: {(end-start)*1000} ms")

    print("Execution finished")

Training Examples
^^^^^^^^^^^^^^^^^

Here is a few training workflow examples.

Train with FP32
"""""""""""""""

.. code-block:: python

    import smith
    import smithvision

    LR = 0.001
    DOWNLOAD = True
    DATA = "datasets/cifar10/"

    transform = smithvision.transforms.Compose(
        [
            smithvision.transforms.Resize((224, 224)),
            smithvision.transforms.ToTensor(),
            smithvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = smithvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = smith.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
    train_len = len(train_loader)

    model = smithvision.models.resnet50()
    criterion = smith.nn.CrossEntropyLoss()
    optimizer = smith.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    model.train()
    model = model.to("xpu")
    criterion = criterion.to("xpu")

    print(f"Initiating training")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("xpu")
        target = target.to("xpu")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
    smith.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint.pth",
    )

    print("Execution finished")

Train with AMP
""""""""""""""

.. note::
   Training with ``GradScaler`` requires hardware support for ``FP64``. ``FP64`` is not natively supported by the Intel® Arc™ A-Series Graphics. If you run your workloads on Intel® Arc™ A-Series Graphics, please disable ``GradScaler``.

.. code-block:: python

    import smith
    import smithvision

    LR = 0.001
    DOWNLOAD = True
    DATA = "datasets/cifar10/"

    use_amp=True

    transform = smithvision.transforms.Compose(
        [
            smithvision.transforms.Resize((224, 224)),
            smithvision.transforms.ToTensor(),
            smithvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = smithvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = smith.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
    train_len = len(train_loader)

    model = smithvision.models.resnet50()
    criterion = smith.nn.CrossEntropyLoss()
    optimizer = smith.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scaler = smith.amp.GradScaler(device="xpu", enabled=use_amp)

    model.train()
    model = model.to("xpu")
    criterion = criterion.to("xpu")

    print(f"Initiating training")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("xpu")
        target = target.to("xpu")
        # set dtype=smith.bfloat16 for BF16
        with smith.autocast(device_type="xpu", dtype=smith.float16, enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")

    smith.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint.pth",
    )

    print("Execution finished")

Train with ``smith.compile``
""""""""""""""""""""""""""""

.. code-block:: python

    import smith
    import smithvision

    LR = 0.001
    DOWNLOAD = True
    DATA = "datasets/cifar10/"

    transform = smithvision.transforms.Compose(
        [
            smithvision.transforms.Resize((224, 224)),
            smithvision.transforms.ToTensor(),
            smithvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = smithvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = smith.utils.data.DataLoader(dataset=train_dataset, batch_size=128)
    train_len = len(train_loader)

    model = smithvision.models.resnet50()
    criterion = smith.nn.CrossEntropyLoss()
    optimizer = smith.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    model.train()
    model = model.to("xpu")
    criterion = criterion.to("xpu")
    model = smith.compile(model)

    print(f"Initiating training with smith compile")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("xpu")
        target = target.to("xpu")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
    smith.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint.pth",
    )

    print("Execution finished")
