# mypy: disable-error-code="no-untyped-def"
import smith.cuda


class cudaError_t:
    cudaSuccess = True


def cudaFree(n):
    return (cudaError_t.cudaSuccess,)


def cudaGetDeviceProperties(d):
    class DummyError:
        value = False

    return (DummyError(), smith.cuda.get_device_properties(d))
