(cond)=

# Control Flow - Cond

`smith.cond` is a structured control flow operator. It can be used to specify if-else like control flow
and can logically be seen as implemented as follows.

```python
def cond(
    pred: Union[bool, smith.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Tuple[smith.Tensor]
):
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
```

Its unique power lies in its ability of expressing **data-dependent control flow**: it lowers to a conditional
operator (`smith.ops.higher_order.cond`), which preserves predicate, true function and false functions.
This unlocks great flexibility in writing and deploying models that change model architecture based on
the **value** or **shape** of inputs or intermediate outputs of tensor operations.

```{warning}
`smith.cond` is a prototype feature in Blacksmith. It has limited support for input and output types and
doesn't support training currently. Please look forward to a more stable implementation in a future version of Blacksmith.
Read more about feature classification at: https://blacksmith.org/blog/blacksmith-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses cond to branch based on input shape:

```python
import smith

def true_fn(x: smith.Tensor):
    return x.cos() + x.sin()

def false_fn(x: smith.Tensor):
    return x.sin()

class DynamicShapeCondPredicate(smith.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        def true_fn(x: smith.Tensor):
            return x.cos()

        def false_fn(x: smith.Tensor):
            return x.sin()

        return smith.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

dyn_shape_mod = DynamicShapeCondPredicate()
```

We can eagerly run the model and expect the results vary based on input shape:

```python
inp = smith.randn(3)
inp2 = smith.randn(5)
assert smith.equal(dyn_shape_mod(inp), false_fn(inp))
assert smith.equal(dyn_shape_mod(inp2), true_fn(inp2))
```

We can export the model for further transformations and deployment:

```python
inp = smith.randn(4, 3)
dim_batch = smith.export.Dim("batch", min=2)
ep = smith.export.export(DynamicShapeCondPredicate(), (inp,), {}, dynamic_shapes={"x": {0: dim_batch}})
print(ep)
```

This gives us an exported program as shown below:

```
class GraphModule(smith.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sym_size: Sym(s0) = smith.ops.aten.sym_size.int(arg0_1, 0)
        gt: Sym(s0 > 4) = sym_size > 4;  sym_size = None
        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = smith.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(smith.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = smith.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = smith.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = smith.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(smith.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = smith.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

Notice that `smith.cond` is lowered to `smith.ops.higher_order.cond`, its predicate becomes a Symbolic expression over the shape of input,
and branch functions becomes two sub-graph attributes of the top level graph module.

Here is another example that showcases how to express a data-dependent control flow:

```python
class DataDependentCondPredicate(smith.nn.Module):
    """
    A basic usage of cond based on data dependent predicate.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        return smith.cond(x.sum() > 4.0, true_fn, false_fn, (x,))
```

The exported program we get after export:

```
class GraphModule(smith.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sum_1: f32[] = smith.ops.aten.sum.default(arg0_1)
        gt: b8[] = smith.ops.aten.gt.Scalar(sum_1, 4.0);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = smith.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(smith.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = smith.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = smith.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = smith.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(smith.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = smith.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

## Invariants of smith.ops.higher_order.cond

There are several useful invariants for `smith.ops.higher_order.cond`:

- For predicate:
    - Dynamicness of predicate is preserved (e.g. `gt` shown in the above example)
    - If the predicate in user-program is constant (e.g. a python bool constant), the `pred` of the operator will be a constant.

- For branches:
    - The input and output signature will be a flattened tuple.
    - They are `smith.fx.GraphModule`.
    - Closures in original function becomes explicit inputs. No closures.
    - No mutations on inputs or globals are allowed.

- For operands:
    - It will also be a flat tuple.

- Nesting of `smith.cond` in user program becomes nested graph modules.

## API Reference

```{eval-rst}
.. autofunction:: smith._higher_order_ops.cond.cond
```
