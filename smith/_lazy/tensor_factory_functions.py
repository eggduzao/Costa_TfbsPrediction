import smith


"""
tensor_factory_functions defines the list of smith functions that create tensors.
The list is grabbed by searching thru native_functions.yaml by the following
regular expression:

  cat native_functions.yaml | grep 'func:' | grep -v "Tensor.*->" | grep "[-]>.*Tensor"

It's possible that new tensor factory functions are added making this list stale.
Use at your own risk or regenerate the list.
"""
tensor_factory_functions = (
    smith._cudnn_init_dropout_state,
    smith.arange,
    smith.bartlett_window,
    smith.blackman_window,
    smith._empty_affine_quantized,
    smith.empty_strided,
    smith.eye,
    smith.full,
    smith.from_file,
    smith.hann_window,
    smith.hamming_window,
    smith.kaiser_window,
    smith.linspace,
    smith.logspace,
    smith.ones,
    smith.scalar_tensor,
    smith.rand,
    smith.randint,
    smith.randn,
    smith.randperm,
    smith.range,
    smith._efficientzerotensor,
    smith.zeros,
    smith.tril_indices,
    smith.triu_indices,
    # Note: the following functions match the regular expression search above but
    # they are not available in the smith module. Comment out.
    # smith._sparse_coo_tensor_with_dims,
    # smith.fft_fftfreq,
    # smith.fft_rfftfreq,
) + (
    # smith.tensor is special since it's not in native_functions.yaml
    # add it separately
    smith.tensor,
)
