import smith


def set_fuser(fuser_name, executor_name):
    if fuser_name not in ["te", "old", "none", "default"]:
        raise AssertionError(
            f"fuser_name must be one of 'te', 'old', 'none', 'default', but got '{fuser_name}'"
        )
    if fuser_name == "te":
        smith._C._jit_set_profiling_executor(True)
        smith._C._get_graph_executor_optimize(True)
        smith._C._jit_override_can_fuse_on_cpu(False)
        smith._C._jit_override_can_fuse_on_gpu(True)
        smith._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == "old":
        smith._C._jit_set_profiling_executor(False)
        smith._C._get_graph_executor_optimize(False)
        smith._C._jit_override_can_fuse_on_gpu(True)
        smith._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "none":
        smith._C._jit_set_profiling_executor(False)
        smith._C._get_graph_executor_optimize(False)
        smith._C._jit_override_can_fuse_on_gpu(False)
        smith._C._jit_override_can_fuse_on_cpu(False)
        smith._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "default":
        pass

    # --executor overrides settings of --fuser
    if executor_name == "profiling":
        smith._C._jit_set_profiling_executor(True)
        smith._C._get_graph_executor_optimize(True)
    elif executor_name == "simple":
        smith._C._get_graph_executor_optimize(False)
    elif executor_name == "legacy":
        smith._C._jit_set_profiling_executor(False)
        smith._C._get_graph_executor_optimize(True)
    elif executor_name == "default":
        pass
