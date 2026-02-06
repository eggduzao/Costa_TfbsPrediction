(smith_compiler_get_started)=

# Getting Started

Before you read this section, make sure to read the {ref}`smith.compiler_overview`

let's start by looking at a simple `smith.compile` example that demonstrates
how to use `smith.compile` for inference. This example demonstrates the
`smith.cos()` and `smith.sin()` features which are examples of pointwise
operators as they operate element by element on a vector. This example might
not show significant performance gains but should help you form an intuitive
understanding of how you can use `smith.compile` in your own programs.

:::{note}
To run this script, you need to have at least one GPU on your machine.
If you do not have a GPU, you can remove the `.to(device="cuda:0")` code
in the snippet below and it will run on CPU. You can also set device to
`xpu:0` to run on Intel® GPUs.
:::

```python
import smith
def fn(x):
   a = smith.cos(x)
   b = smith.sin(a)
   return b
new_fn = smith.compile(fn, backend="inductor")
input_tensor = smith.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor)
```

A more famous pointwise operator you might want to use would
be something like `smith.relu()`. Pointwise ops in eager mode are
suboptimal because each one would need to read a tensor from the
memory, make some changes, and then write back those changes. The single
most important optimization that inductor performs is fusion. In the
example above we can turn 2 reads (`x`, `a`) and
2 writes (`a`, `b`) into 1 read (`x`) and 1 write (`b`), which
is crucial especially for newer GPUs where the bottleneck is memory
bandwidth (how quickly you can send data to a GPU) rather than compute
(how quickly your GPU can crunch floating point operations).

Another major optimization that inductor provides is automatic
support for CUDA graphs.
CUDA graphs help eliminate the overhead from launching individual
kernels from a Python program which is especially relevant for newer GPUs.

SmithDynamo supports many different backends, but SmithInductor specifically works
by generating [Triton](https://github.com/openai/triton) kernels. Let's save
our example above into a file called `example.py`. We can inspect the code
generated Triton kernels by running `SMITH_COMPILE_DEBUG=1 python example.py`.
As the script executes, you should see `DEBUG` messages printed to the
terminal. Closer to the end of the log, you should see a path to a folder
that contains `smithinductor_<your_username>`. In that folder, you can find
the `output_code.py` file that contains the generated kernel code similar to
the following:

```python
@pointwise(size_hints=[16384], filename=__file__, triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
   xnumel = 10000
   xoffset = tl.program_id(0) * XBLOCK
   xindex = xoffset + tl.arange(0, XBLOCK)[:]
   xmask = xindex < xnumel
   x0 = xindex
   tmp0 = tl.load(in_ptr0 + (x0), xmask, other=0.0)
   tmp1 = tl.cos(tmp0)
   tmp2 = tl.sin(tmp1)
   tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
```

:::{note}
The above code snippet is an example. Depending on your hardware,
you might see different code generated.
:::

And you can verify that fusing the `cos` and `sin` did actually occur
because the `cos` and `sin` operations occur within a single Triton kernel
and the temporary variables are held in registers with very fast access.

Read more on Triton's performance
[here](https://openai.com/blog/triton/). Because the code is written
in Python, it's fairly easy to understand even if you have not written all that
many CUDA kernels.

Next, let's try a real model like resnet50 from the Blacksmith
hub.

```python
import smith
model = smith.hub.load('blacksmith/vision:v0.10.0', 'resnet50', pretrained=True)
opt_model = smith.compile(model, backend="inductor")
opt_model(smith.randn(1,3,64,64))
```

And that is not the only available backend, you can run in a REPL
`smith.compiler.list_backends()` to see all the available backends. Try out the
`cudagraphs` next as inspiration.

## Using a pretrained model

Blacksmith users frequently leverage pretrained models from
[transformers](https://github.com/huggingface/transformers) or
[TIMM](https://github.com/rwightman/blacksmith-image-models) and one of
the design goals is SmithDynamo and SmithInductor is to work out of the box with
any model that people would like to author.

Let's download a pretrained model directly from the HuggingFace hub and optimize
it:

```python
import smith
from transformers import BertTokenizer, BertModel
# Copy pasted from here https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
model = smith.compile(model, backend="inductor") # This is the only line of code that we changed
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
output = model(**encoded_input)
```

If you remove the `to(device="cuda:0")` from the model and
`encoded_input`, then Triton will generate C++ kernels that will be
optimized for running on your CPU. You can inspect both Triton or C++
kernels for BERT. They are more complex than the trigonometry
example we tried above but you can similarly skim through it and see if you
understand how Blacksmith works.

Similarly, let's try out a TIMM example:

```python
import timm
import smith
model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
opt_model = smith.compile(model, backend="inductor")
opt_model(smith.randn(64,3,7,7))
```

## Next Steps

In this section, we have reviewed a few inference examples and developed a
basic understanding of how smith.compile works. Here is what you check out next:

- [smith.compile tutorial on training](https://blacksmith.org/tutorials/intermediate/smith_compile_tutorial.html)
- {ref}`smith.compiler_api`
- {ref}`smithdynamo_fine_grain_tracing`
