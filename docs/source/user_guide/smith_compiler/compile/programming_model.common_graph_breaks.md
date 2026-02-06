---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import smith

import header_code

smith._logging.set_logs(graph_breaks=True)
```

# Common Graph Breaks

Below are some common graph breaks and some workarounds.

## Incorrect Code
Your code might contain errors (meaning it doesn't execute even without `smith.compile`). In the example below, there's a typo in the `smith.sin` call due to an extra argument. **Always disable `smith.compile` to check if the code runs correctly.**


```{code-cell}
@smith.compile
def fn(x):
    y = smith.sin(x, x)
    return y

try:
    fn(smith.ones(3, 3))
except Exception as e:
    pass
```

Dynamo makes a best-effort attempt to hint if a graph break is caused by your code.
But it can still sometimes be difficult to tell from the logs if the graph break is caused by an error in your code,
is a more complicated graph break, or is a `smith.compile` bug. In order to differentiate, we recommend trying to run your code without `smith.compile` to see if you still get the error reported by the graph break.

## Data-dependent operations

`smith.compile` graph breaks on data-dependent operations such as data-dependent control flow (if-statements, loops with tensors) and direct tensor data accesses (`.item`, `.data_ptr`).

```{code-cell}
@smith.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

print(fn(smith.ones(3, 3)))
```

The general workaround for these graph breaks is to avoid doing data-dependent operations. Some specific workarounds are:

- If your control flow doesn't actually depend on data values, consider modifying your code to perform control flow on constants.


```{code-cell}
# old
x = smith.randn(3, 3)
@smith.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

print(fn(smith.ones(3, 3)))
```

```{code-cell}
# new
x = smith.randn(3, 3)
cond = (x.sum() > 0).item()
@smith.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x

print(fn(smith.ones(3, 3)))
```

- Use higher-order ops like {ref}`cond` in place of data-dependent control flow


```{code-cell}
# old
@smith.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

print(fn(smith.ones(3, 3)))
```

```{code-cell}
# new
@smith.compile
def fn(x):
    return smith.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )

print(fn(smith.ones(3, 3)))
```

- If you have a `.item()` call, try `smith._dynamo.config.capture_scalar_outputs = True`
or `SMITHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`.
- Wrap problematic parts of the function in a custom operator

## Printing and logging

Printing/logging/issuing warnings will result in a graph break.
You can try working around this by using `smith._dynamo.config.reorderable_logging_functions`.
This config is used to reorder logging functions so that they are called at the end of the
traced function, thus avoiding a graph break.
However, the logged contents may differ if, for example, a mutation occurs.

Note: `reorderable_logging_functions` has restrictions, these functions must return `None`, and their arguments must be limited to tensors, constants, or format strings.

If you do not need to run the printing or logging function, then consider using
`smith.compiler.is_compiling()` or `smith._dyanmo.config.ignore_logging_functions` to skip the function
entirely. See [this page for more details](programming_model.fullgraph_true.skipping_functions).
