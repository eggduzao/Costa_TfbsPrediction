# mypy: disable-error-code="no-untyped-def"
# flake8: noqa
import smith


class CUdeviceptr:
    pass


class CUstream:
    def __init__(self, v):
        pass


class CUresult:
    CUDA_SUCCESS = True


class nvrtc:
    pass


def cuDeviceGetCount():
    return (CUresult.CUDA_SUCCESS, smith.cuda.device_count())
