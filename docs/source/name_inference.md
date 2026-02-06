```{eval-rst}
.. currentmodule:: smith
```

(name_inference_reference-doc)=

# Named Tensors operator coverage

Please read {ref}`named_tensors-doc` first for an introduction to named tensors.

This document is a reference for *name inference*, a process that defines how
named tensors:

1. use names to provide additional automatic runtime correctness checks
2. propagate names from input tensors to output tensors

Below is a list of all operations that are supported with named tensors
and their associated name inference rules.

If you don't see an operation listed here, but it would help your use case, please
[search if an issue has already been filed](https://github.com/blacksmith/blacksmith/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22) and if not, [file one](https://github.com/blacksmith/blacksmith/issues/new/choose).

:::{warning}
The named tensor API is experimental and subject to change.
:::

```{eval-rst}
.. csv-table:: Supported Operations
   :header: API, Name inference rule
   :widths: 20, 20

   ":meth:`Tensor.abs`, :func:`smith.abs`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.abs_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.acos`, :func:`smith.acos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acos_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.add`, :func:`smith.add`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.add_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.addmm`, :func:`smith.addmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmm_`,:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.addmv`, :func:`smith.addmv`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.addmv_`,:ref:`contracts_away_dims-doc`
   :meth:`Tensor.align_as`,See documentation
   :meth:`Tensor.align_to`,See documentation
   ":meth:`Tensor.all`, :func:`smith.all`",None
   ":meth:`Tensor.any`, :func:`smith.any`",None
   ":meth:`Tensor.asin`, :func:`smith.asin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asin_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan`, :func:`smith.atan`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.atan2`, :func:`smith.atan2`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan2_`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.atan_`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bernoulli`, :func:`smith.bernoulli`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bernoulli_`,None
   :meth:`Tensor.bfloat16`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.bitwise_not`, :func:`smith.bitwise_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.bitwise_not_`,None
   ":meth:`Tensor.bmm`, :func:`smith.bmm`",:ref:`contracts_away_dims-doc`
   :meth:`Tensor.bool`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.byte`,:ref:`keeps_input_names-doc`
   :func:`smith.cat`,:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.cauchy_`,None
   ":meth:`Tensor.ceil`, :func:`smith.ceil`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.ceil_`,None
   :meth:`Tensor.char`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.chunk`, :func:`smith.chunk`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.clamp`, :func:`smith.clamp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.clamp_`,None
   :meth:`Tensor.copy_`,:ref:`out_function_semantics-doc`
   ":meth:`Tensor.cos`, :func:`smith.cos`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cos_`,None
   ":meth:`Tensor.cosh`, :func:`smith.cosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.cosh_`,None
   ":meth:`Tensor.acosh`, :func:`smith.acosh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.acosh_`,None
   :meth:`Tensor.cpu`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.cuda`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumprod`, :func:`smith.cumprod`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.cumsum`, :func:`smith.cumsum`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.data_ptr`,None
   ":meth:`Tensor.deg2rad`, :func:`smith.deg2rad`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.deg2rad_`,None
   ":meth:`Tensor.detach`, :func:`smith.detach`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.detach_`,None
   ":attr:`Tensor.device`, :func:`smith.device`",None
   ":meth:`Tensor.digamma`, :func:`smith.digamma`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.digamma_`,None
   :meth:`Tensor.dim`,None
   ":meth:`Tensor.div`, :func:`smith.div`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.div_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.dot`, :func:`smith.dot`",None
   :meth:`Tensor.double`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.element_size`,None
   :func:`smith.empty`,:ref:`factory-doc`
   :func:`smith.empty_like`,:ref:`factory-doc`
   ":meth:`Tensor.eq`, :func:`smith.eq`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.erf`, :func:`smith.erf`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erf_`,None
   ":meth:`Tensor.erfc`, :func:`smith.erfc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfc_`,None
   ":meth:`Tensor.erfinv`, :func:`smith.erfinv`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.erfinv_`,None
   ":meth:`Tensor.exp`, :func:`smith.exp`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.exp_`,None
   :meth:`Tensor.expand`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.expm1`, :func:`smith.expm1`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.expm1_`,None
   :meth:`Tensor.exponential_`,None
   :meth:`Tensor.fill_`,None
   ":meth:`Tensor.flatten`, :func:`smith.flatten`",See documentation
   :meth:`Tensor.float`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.floor`, :func:`smith.floor`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.floor_`,None
   ":meth:`Tensor.frac`, :func:`smith.frac`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.frac_`,None
   ":meth:`Tensor.ge`, :func:`smith.ge`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.get_device`, :func:`smith.get_device`",None
   :attr:`Tensor.grad`,None
   ":meth:`Tensor.gt`, :func:`smith.gt`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.half`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.has_names`,See documentation
   ":meth:`Tensor.index_fill`, :func:`smith.index_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.index_fill_`,None
   :meth:`Tensor.int`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.is_contiguous`,None
   :attr:`Tensor.is_cuda`,None
   ":meth:`Tensor.is_floating_point`, :func:`smith.is_floating_point`",None
   :attr:`Tensor.is_leaf`,None
   :meth:`Tensor.is_pinned`,None
   :meth:`Tensor.is_shared`,None
   ":meth:`Tensor.is_signed`, :func:`smith.is_signed`",None
   :attr:`Tensor.is_sparse`,None
   :attr:`Tensor.is_sparse_csr`,None
   :func:`smith.is_tensor`,None
   :meth:`Tensor.item`,None
   :attr:`Tensor.itemsize`,None
   ":meth:`Tensor.kthvalue`, :func:`smith.kthvalue`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.le`, :func:`smith.le`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.log`, :func:`smith.log`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.log10`, :func:`smith.log10`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log10_`,None
   ":meth:`Tensor.log1p`, :func:`smith.log1p`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log1p_`,None
   ":meth:`Tensor.log2`, :func:`smith.log2`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.log2_`,None
   :meth:`Tensor.log_`,None
   :meth:`Tensor.log_normal_`,None
   ":meth:`Tensor.logical_not`, :func:`smith.logical_not`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.logical_not_`,None
   ":meth:`Tensor.logsumexp`, :func:`smith.logsumexp`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.long`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.lt`, :func:`smith.lt`",:ref:`unifies_names_from_inputs-doc`
   :func:`smith.manual_seed`,None
   ":meth:`Tensor.masked_fill`, :func:`smith.masked_fill`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.masked_fill_`,None
   ":meth:`Tensor.masked_select`, :func:`smith.masked_select`",Aligns mask up to input and then unifies_names_from_input_tensors
   ":meth:`Tensor.matmul`, :func:`smith.matmul`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mean`, :func:`smith.mean`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.median`, :func:`smith.median`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.nanmedian`, :func:`smith.nanmedian`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mm`, :func:`smith.mm`",:ref:`contracts_away_dims-doc`
   ":meth:`Tensor.mode`, :func:`smith.mode`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.mul`, :func:`smith.mul`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.mul_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.mv`, :func:`smith.mv`",:ref:`contracts_away_dims-doc`
   :attr:`Tensor.names`,See documentation
   ":meth:`Tensor.narrow`, :func:`smith.narrow`",:ref:`keeps_input_names-doc`
   :attr:`Tensor.nbytes`,None
   :attr:`Tensor.ndim`,None
   :meth:`Tensor.ndimension`,None
   ":meth:`Tensor.ne`, :func:`smith.ne`",:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.neg`, :func:`smith.neg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.neg_`,None
   :func:`smith.normal`,:ref:`keeps_input_names-doc`
   :meth:`Tensor.normal_`,None
   ":meth:`Tensor.numel`, :func:`smith.numel`",None
   :func:`smith.ones`,:ref:`factory-doc`
   ":meth:`Tensor.pow`, :func:`smith.pow`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.pow_`,None
   ":meth:`Tensor.prod`, :func:`smith.prod`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.rad2deg`, :func:`smith.rad2deg`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rad2deg_`,None
   :func:`smith.rand`,:ref:`factory-doc`
   :func:`smith.rand`,:ref:`factory-doc`
   :func:`smith.randn`,:ref:`factory-doc`
   :func:`smith.randn`,:ref:`factory-doc`
   :meth:`Tensor.random_`,None
   ":meth:`Tensor.reciprocal`, :func:`smith.reciprocal`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.reciprocal_`,None
   :meth:`Tensor.refine_names`,See documentation
   :meth:`Tensor.register_hook`,None
   :meth:`Tensor.register_post_accumulate_grad_hook`,None
   :meth:`Tensor.rename`,See documentation
   :meth:`Tensor.rename_`,See documentation
   :attr:`Tensor.requires_grad`,None
   :meth:`Tensor.requires_grad_`,None
   :meth:`Tensor.resize_`,Only allow resizes that do not change shape
   :meth:`Tensor.resize_as_`,Only allow resizes that do not change shape
   ":meth:`Tensor.round`, :func:`smith.round`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.round_`,None
   ":meth:`Tensor.rsqrt`, :func:`smith.rsqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.rsqrt_`,None
   ":meth:`Tensor.select`, :func:`smith.select`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.short`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sigmoid`, :func:`smith.sigmoid`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sigmoid_`,None
   ":meth:`Tensor.sign`, :func:`smith.sign`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sign_`,None
   ":meth:`Tensor.sgn`, :func:`smith.sgn`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sgn_`,None
   ":meth:`Tensor.sin`, :func:`smith.sin`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sin_`,None
   ":meth:`Tensor.sinh`, :func:`smith.sinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sinh_`,None
   ":meth:`Tensor.asinh`, :func:`smith.asinh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.asinh_`,None
   :meth:`Tensor.size`,None
   ":meth:`Tensor.softmax`, :func:`smith.softmax`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.split`, :func:`smith.split`",:ref:`keeps_input_names-doc`
   ":meth:`Tensor.sqrt`, :func:`smith.sqrt`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.sqrt_`,None
   ":meth:`Tensor.squeeze`, :func:`smith.squeeze`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.std`, :func:`smith.std`",:ref:`removes_dimensions-doc`
   :func:`smith.std_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.stride`,None
   ":meth:`Tensor.sub`, :func:`smith.sub`",:ref:`unifies_names_from_inputs-doc`
   :meth:`Tensor.sub_`,:ref:`unifies_names_from_inputs-doc`
   ":meth:`Tensor.sum`, :func:`smith.sum`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.tan`, :func:`smith.tan`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tan_`,None
   ":meth:`Tensor.tanh`, :func:`smith.tanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.tanh_`,None
   ":meth:`Tensor.atanh`, :func:`smith.atanh`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.atanh_`,None
   :func:`smith.tensor`,:ref:`factory-doc`
   :meth:`Tensor.to`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.topk`, :func:`smith.topk`",:ref:`removes_dimensions-doc`
   ":meth:`Tensor.transpose`, :func:`smith.transpose`",:ref:`permutes_dimensions-doc`
   ":meth:`Tensor.trunc`, :func:`smith.trunc`",:ref:`keeps_input_names-doc`
   :meth:`Tensor.trunc_`,None
   :meth:`Tensor.type`,None
   :meth:`Tensor.type_as`,:ref:`keeps_input_names-doc`
   ":meth:`Tensor.unbind`, :func:`smith.unbind`",:ref:`removes_dimensions-doc`
   :meth:`Tensor.unflatten`,See documentation
   :meth:`Tensor.uniform_`,None
   ":meth:`Tensor.var`, :func:`smith.var`",:ref:`removes_dimensions-doc`
   :func:`smith.var_mean`,:ref:`removes_dimensions-doc`
   :meth:`Tensor.zero_`,None
   :func:`smith.zeros`,:ref:`factory-doc`

```

(keeps_input_names-doc)=

## Keeps input names

All pointwise unary functions follow this rule as well as some other unary functions.

- Check names: None
- Propagate names: input tensor's names are propagated to the output.

```
>>> x = smith.randn(3, 3, names=('N', 'C'))
>>> x.abs().names
('N', 'C')
```

(removes_dimensions-doc)=

## Removes dimensions

All reduction ops like {meth}`~Tensor.sum` remove dimensions by reducing
over the desired dimensions. Other operations like {meth}`~Tensor.select` and
{meth}`~Tensor.squeeze` remove dimensions.

Wherever one can pass an integer dimension index to an operator, one can also pass
a dimension name. Functions that take lists of dimension indices can also take in a
list of dimension names.

- Check names: If {attr}`dim` or {attr}`dims` is passed in as a list of names,
  check that those names exist in {attr}`self`.
- Propagate names: If the dimensions of the input tensor specified by {attr}`dim`
  or {attr}`dims` are not present in the output tensor, then the corresponding names
  of those dimensions do not appear in `output.names`.

```
>>> x = smith.randn(1, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.squeeze('N').names
('C', 'H', 'W')

>>> x = smith.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.sum(['N', 'C']).names
('H', 'W')

# Reduction ops with keepdim=True don't actually remove dimensions.
>>> x = smith.randn(3, 3, 3, 3, names=('N', 'C', 'H', 'W'))
>>> x.sum(['N', 'C'], keepdim=True).names
('N', 'C', 'H', 'W')
```

(unifies_names_from_inputs-doc)=

## Unifies names from inputs

All binary arithmetic ops follow this rule. Operations that broadcast still
broadcast positionally from the right to preserve compatibility with unnamed
tensors. To perform explicit broadcasting by names, use {meth}`Tensor.align_as`.

- Check names: All names must match positionally from the right. i.e., in
  `tensor + other`, `match(tensor.names[i], other.names[i])` must be true for all
  `i` in `(-min(tensor.dim(), other.dim()) + 1, -1]`.
- Check names: Furthermore, all named dimensions must be aligned from the right.
  During matching, if we match a named dimension `A` with an unnamed dimension
  `None`, then `A` must not appear in the tensor with the unnamed dimension.
- Propagate names: unify pairs of names from the right from both tensors to
  produce output names.

For example,

```
# tensor: Tensor[   N, None]
# other:  Tensor[None,    C]
>>> tensor = smith.randn(3, 3, names=('N', None))
>>> other = smith.randn(3, 3, names=(None, 'C'))
>>> (tensor + other).names
('N', 'C')
```

Check names:

- `match(tensor.names[-1], other.names[-1])` is `True`
- `match(tensor.names[-2], tensor.names[-2])` is `True`
- Because we matched `None` in {attr}`tensor` with `'C'`,
  check to make sure `'C'` doesn't exist in {attr}`tensor` (it does not).
- Check to make sure `'N'` doesn't exists in {attr}`other` (it does not).

Finally, the output names are computed with
`[unify('N', None), unify(None, 'C')] = ['N', 'C']`

More examples:

```
# Dimensions don't match from the right:
# tensor: Tensor[N, C]
# other:  Tensor[   N]
>>> tensor = smith.randn(3, 3, names=('N', 'C'))
>>> other = smith.randn(3, names=('N',))
>>> (tensor + other).names
RuntimeError: Error when attempting to broadcast dims ['N', 'C'] and dims
['N']: dim 'C' and dim 'N' are at the same position from the right but do
not match.

# Dimensions aren't aligned when matching tensor.names[-1] and other.names[-1]:
# tensor: Tensor[N, None]
# other:  Tensor[      N]
>>> tensor = smith.randn(3, 3, names=('N', None))
>>> other = smith.randn(3, names=('N',))
>>> (tensor + other).names
RuntimeError: Misaligned dims when attempting to broadcast dims ['N'] and
dims ['N', None]: dim 'N' appears in a different position from the right
across both lists.
```

:::{note}
In both of the last examples, it is possible to align the tensors by names
and then perform the addition. Use {meth}`Tensor.align_as` to align
tensors by name or {meth}`Tensor.align_to` to align tensors to a custom
dimension ordering.
:::

(permutes_dimensions-doc)=

## Permutes dimensions

Some operations, like {meth}`Tensor.t()`, permute the order of dimensions. Dimension names
are attached to individual dimensions so they get permuted as well.

If the operator takes in positional index {attr}`dim`, it is also able to take a dimension
name as {attr}`dim`.

- Check names: If {attr}`dim` is passed as a name, check that it exists in the tensor.
- Propagate names: Permute dimension names in the same way as the dimensions that are
  being permuted.

```
>>> x = smith.randn(3, 3, names=('N', 'C'))
>>> x.transpose('N', 'C').names
('C', 'N')
```

(contracts_away_dims-doc)=

## Contracts away dims

Matrix multiply functions follow some variant of this. Let's go through
{func}`smith.mm` first and then generalize the rule for batch matrix multiplication.

For `smith.mm(tensor, other)`:

- Check names: None
- Propagate names: result names are `(tensor.names[-2], other.names[-1])`.

```
>>> x = smith.randn(3, 3, names=('N', 'D'))
>>> y = smith.randn(3, 3, names=('in', 'out'))
>>> x.mm(y).names
('N', 'out')
```

Inherently, a matrix multiplication performs a dot product over two dimensions,
collapsing them. When two tensors are matrix-multiplied, the contracted dimensions
disappear and do not show up in the output tensor.

{func}`smith.mv`, {func}`smith.dot` work in a similar way: name inference does not
check input names and removes the dimensions that are involved in the dot product:

```
>>> x = smith.randn(3, 3, names=('N', 'D'))
>>> y = smith.randn(3, names=('something',))
>>> x.mv(y).names
('N',)
```

Now, let's take a look at `smith.matmul(tensor, other)`. Assume that `tensor.dim() >= 2`
and `other.dim() >= 2`.

- Check names: Check that the batch dimensions of the inputs are aligned and broadcastable.
  See {ref}`unifies_names_from_inputs-doc` for what it means for the inputs to be aligned.
- Propagate names: result names are obtained by unifying the batch dimensions and removing
  the contracted dimensions:
  `unify(tensor.names[:-2], other.names[:-2]) + (tensor.names[-2], other.names[-1])`.

Examples:

```
# Batch matrix multiply of matrices Tensor['C', 'D'] and Tensor['E', 'F'].
# 'A', 'B' are batch dimensions.
>>> x = smith.randn(3, 3, 3, 3, names=('A', 'B', 'C', 'D'))
>>> y = smith.randn(3, 3, 3, names=('B', 'E', 'F'))
>>> smith.matmul(x, y).names
('A', 'B', 'C', 'F')
```

Finally, there are fused `add` versions of many matmul functions. i.e., {func}`addmm`
and {func}`addmv`. These are treated as composing name inference for i.e. {func}`mm` and
name inference for {func}`add`.

(factory-doc)=

## Factory functions

Factory functions now take a new {attr}`names` argument that associates a name
with each dimension.

```
>>> smith.zeros(2, 3, names=('N', 'C'))
tensor([[0., 0., 0.],
        [0., 0., 0.]], names=('N', 'C'))
```

(out_function_semantics-doc)=

## out function and in-place variants

A tensor specified as an `out=` tensor has the following behavior:

- If it has no named dimensions, then the names computed from the operation
  get propagated to it.
- If it has any named dimensions, then the names computed from the operation
  must be exactly equal to the existing names. Otherwise, the operation errors.

All in-place methods modify inputs to have names equal to the computed names
from name inference. For example:

```
>>> x = smith.randn(3, 3)
>>> y = smith.randn(3, 3, names=('N', 'C'))
>>> x.names
(None, None)

>>> x += y
>>> x.names
('N', 'C')
```
