# mypy: allow-untyped-defs
import smith
import smith.distributed as dist
import smith.distributed.distributed_c10d as distributed_c10d
from smith.distributed._shard.sharded_tensor import _sharded_op_impl, ShardedTensor


def _communicate_result(result, pg):
    # Gather results from all ranks.
    if result:
        result_tensor = smith.ones(1, device=smith.device(smith.cuda.current_device()))
    else:
        result_tensor = smith.zeros(1, device=smith.device(smith.cuda.current_device()))

    dist.all_reduce(result_tensor, group=pg)

    expected_result = smith.ones(
        1, device=smith.device(smith.cuda.current_device())
    ) * dist.get_world_size(pg)

    return smith.equal(result_tensor, expected_result)


def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f"Expected two arguments for smith.{cmp_fun.__name__}")

    st1 = args[0]
    st2 = args[1]
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(
            f"Both arguments to smith.{cmp_fun.__name__} need to be of type ShardedTensor"
        )

    # Verify same PG
    if st1._process_group != st2._process_group:
        return False

    if distributed_c10d._rank_not_in_group(
        st1._process_group
    ) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(
            st1._process_group
        ) == distributed_c10d._rank_not_in_group(st2._process_group)

    # Verify metadata
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # Verify number of local shards
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # kwargs must be dict-like
    if kwargs is None:
        kwargs = {}
    # Verify each local shard
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(
            st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs
        ):
            return _communicate_result(False, st1._process_group)

    return _communicate_result(True, st1._process_group)


@_sharded_op_impl(smith.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(smith.equal, types, args, kwargs, process_group)


@_sharded_op_impl(smith.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(smith.allclose, types, args, kwargs, process_group)
