# Owner(s): ["module: inductor"]
import unittest

import smith
import smith._dynamo
import smith._dynamo.logging
import smith._dynamo.test_case
import smith.distributed as dist
import smith.fx as fx

# for some reason importing functional collectives after dynamo breaks collectives handling!
from smith._C import FileCheck
from smith._dynamo.utils import counters
from smith._inductor.test_case import TestCase as InductorTestCase
from smith._subclasses.fake_tensor import FakeTensorMode
from smith.fx.experimental.proxy_tensor import make_fx
from smith.testing._internal.common_distributed import requires_accelerator_dist_backend
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from smith.testing._internal.inductor_utils import HAS_GPU
from smith.utils._ordered_set import OrderedSet


# flake8: noqa: B950
# Owner(s): ["module: inductor"]


aten = smith.ops.aten

from smith.testing._internal.common_fsdp import get_devtype


device_type = str(get_devtype())


import smith
import smith._dynamo
import smith._dynamo.logging
import smith._dynamo.test_case


# for some reason importing functional collectives after dynamo breaks collectives handling!


@requires_accelerator_dist_backend(["nccl", "xccl"])
def build_collective_info(graph, hiding_annotations):
    """
    Build CollectiveInfo dict from manual hiding annotations.

    hiding_annotations: dict mapping collective_start -> hiding_compute_node(s)
                        Can be a single node or a list/OrderedSet of nodes
    """
    from smith._inductor.fx_passes.overlap_scheduling import CollectiveInfo

    collective_info = {}

    # Find all collective starts and their corresponding waits
    start_to_wait = {}
    for node in graph.nodes:
        if node.op == "call_function" and "wait_tensor" in str(node.target):
            wait_input = node.args[0]
            if isinstance(wait_input, fx.Node):
                start_to_wait[wait_input] = node

    # Build CollectiveInfo for each collective
    for start_node, wait_node in start_to_wait.items():
        hiding_annotation = hiding_annotations.get(start_node)

        # Convert to OrderedSet
        hiding_nodes = OrderedSet()
        if hiding_annotation is not None:
            if isinstance(hiding_annotation, list | OrderedSet):
                hiding_nodes = OrderedSet(hiding_annotation)
            else:
                hiding_nodes = OrderedSet([hiding_annotation])

        # Estimate size and time
        size_bytes = 16 * 4  # 4x4 tensor of floats
        estimated_time_ms = 1.0  # Dummy time
        exposed_time_ms = 0.0 if hiding_nodes else 1.0  # Hidden if has hiding_nodes

        collective_info[start_node] = CollectiveInfo(
            start_node=start_node,
            wait_node=wait_node,
            size_bytes=size_bytes,
            estimated_time_ms=estimated_time_ms,
            exposed_time_ms=exposed_time_ms,
            hiding_nodes=hiding_nodes,
        )

    return collective_info


@requires_accelerator_dist_backend()
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
@instantiate_parametrized_tests
class TestOverlapPreservingBucketing(InductorTestCase):
    """
    Unit tests for overlap-preserving bucketing pass.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from smith.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group()

    def test_can_bucket_independent_collectives(self):
        """
        Test that independent collectives with separate hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 (hides ag1) -> mm2 (hides ag2) -> ag1_wait -> ag2_wait
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # Start both collectives
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # Independent compute that can hide both
            mm1 = smith.mm(a, a)
            mm2 = smith.mm(b, b)

            # Wait for both
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )

    def test_cant_bucket_nested_hiding_intervals(self):
        """
        Test that nested hiding intervals prevent bucketing.

        Graph structure:
        ag1_start -> ag2_start -> mm2 (hides ag2) -> ag2_wait -> mm1 (hides ag1) -> ag1_wait

        ag2's hiding interval is nested inside ag1's hiding interval.
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # ag1 starts first
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )

            # ag2 starts (inside ag1's interval)
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # mm2 hides ag2
            mm2 = smith.mm(b[:2, :2], b[:2, :2])

            # ag2 waits (still inside ag1's interval)
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)

            # mm1 uses ag2's result and hides ag1
            mm1 = smith.mm(a + ag2_out[:4, :4], a)

            # ag1 waits last
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )
        # mm2 is the first mm, mm1 is the second (based on graph order)
        mm2 = mm_nodes[0]
        mm1 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: nested hiding intervals should prevent bucketing
        # Should have 2 separate all_gathers, not 1 bucketed one
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor", 2, exactly=False).run(
            graph_str
        )

    @parametrize("final_mm_hidden", (True, False))
    def test_cant_bucket_ag_with_rs_hiding_interval_between(self, final_mm_hidden):
        """
        Test that all_gathers can't bucket when a reduce_scatter's hiding interval is between them.

        Graph structure:
        ag1_start -> mm1 (hides ag1) -> ag1_wait ->
        rs_start -> mm2 (hides rs) -> rs_wait ->

        if final_mm_hidden:
            ag2_start -> mm3 (hides ag2) -> ag2_wait

        if final_mm_hidden:
            Bucketing ag1 and ag2 would require moving one of them, which would break hiding relationships:
            - Moving ag2 earlier would break ag2's hiding by mm3
            - Moving ag1 later would break ag1's hiding by mm1
            - The rs hiding interval creates an obstacle between them

        otherwise, we can bucket
        """

        def func(a, b, c):
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 1

            # First all_gather
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            mm1 = smith.mm(a, a)  # hides ag1
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)

            # Reduce scatter in between
            rs = smith.ops._c10d_functional.reduce_scatter_tensor(
                b, "sum", group_size, group_name
            )
            mm2 = smith.mm(b[:4, :4], b[:4, :4])  # hides rs
            rs_out = smith.ops._c10d_functional.wait_tensor(rs)

            # Second all_gather
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )
            mm3 = smith.mm(c, c)  # hides ag2
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + rs_out.sum() + ag2_out.sum(), mm1, mm2, mm3

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(8, 4, device=self.device)
            c = smith.ones(4, 4, device=self.device)

            # Trace with make_fx
            traced = make_fx(func)(a, b, c)

        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        (rs,) = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            # rs: mm2,   # mm2 hides rs
            ag2: mm3,
        }
        if final_mm_hidden:
            hiding_annotations[rs] = mm2

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing logic to find buckets (without applying them, which would require process groups)
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        graph_str = str(traced.graph)

        # check order of mms preserved
        FileCheck().check("%mm").check("%mm_1").check("%mm_2").run(graph_str)

        if final_mm_hidden:
            # Should NOT bucket - 2 separate all_gathers
            # Count all_gather node names (works even when wrapped in control_deps)
            FileCheck().check_count("%all_gather_into_tensor", 2, exactly=False).run(
                graph_str
            )
        else:
            # Should bucket - 1 bucketed all_gather (all_gather_into_tensor_out)
            FileCheck().check_count(
                "%all_gather_into_tensor_out", 1, exactly=False
            ).run(graph_str)

    def test_can_bucket_all_reduce(self):
        """
        Test that all_reduce operations CAN bucket together.

        Graph structure:
        ar1_start -> ar2_start -> mm1 (hides ar1) -> mm2 (hides ar2) -> ar1_wait -> ar2_wait
        """

        def func(a, b):
            group_name = "0"

            # Start both all_reduce operations
            ar1 = smith.ops._c10d_functional.all_reduce(a, "sum", group_name)
            ar2 = smith.ops._c10d_functional.all_reduce(b, "sum", group_name)

            # Independent compute that can hide both
            mm1 = smith.mm(a, a)
            mm2 = smith.mm(b, b)

            # Wait for both
            ar1_out = smith.ops._c10d_functional.wait_tensor(ar1)
            ar2_out = smith.ops._c10d_functional.wait_tensor(ar2)

            return ar1_out.sum() + ar2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes
        ar1, ar2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_reduce.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )

        # For all_reduce, start_node == wait_node (no separate wait)
        hiding_annotations = {
            ar1: mm1,
            ar2: mm2,
        }

        # Build collective info
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed all_reduce
        # After bucketing, there should be only one all_reduce node (the bucketed one)
        # Check for cat (bucketing input) and split_with_sizes (bucketing output)
        graph_str = str(traced.graph)
        FileCheck().check("cat.default").check("all_reduce.default").check(
            "split_with_sizes"
        ).check_count("%mm", 2).run(graph_str)

    def test_can_bucket_multidtype_collectives(self):
        """
        Test that all_gathers with different dtypes CAN bucket together.

        Graph structure:
        ag1_float32 -> mm1 (hides ag1) -> ag1_wait
        ag2_bfloat16 -> mm2 (hides ag2) -> ag2_wait
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # Start both collectives with different dtypes
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a,
                group_size,
                group_name,  # float32
            )
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b,
                group_size,
                group_name,  # bfloat16
            )

            # Independent compute that can hide both
            mm1 = smith.mm(a, a)
            mm2 = smith.mm(b.float(), b.float())

            # Wait for both
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device, dtype=smith.float32)
            b = smith.ones(4, 4, device=self.device, dtype=smith.bfloat16)

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )
        mm1 = mm_nodes[0]
        mm2 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing with multidtype mode
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="custom_ops_multidtype",
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        # even though dtypes are different
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )

    def test_can_bucket_with_multiple_hiding_nodes(self):
        """
        Test that collectives with multiple hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 -> mm2 -> mm3 -> ag1_wait -> ag2_wait

        Where:
        - ag1 is hidden by mm1 and mm2
        - ag2 is hidden by mm2 and mm3
        - Both collectives share mm2 as a hiding node
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # Start both collectives
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # Three compute operations that hide the collectives
            mm1 = smith.mm(a, a)
            mm2 = smith.mm(b, b)
            mm3 = smith.mm(a + b, a + b)

            # Wait for both
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum() + mm3.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )

        # Manually annotate hiding relationships with multiple hiding nodes
        hiding_annotations = {
            ag1: [mm1, mm2],  # ag1 is hidden by mm1 and mm2
            ag2: [mm2, mm3],  # ag2 is hidden by mm2 and mm3
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Verify hiding_nodes are correctly set
        self.assertEqual(len(collective_info[ag1].hiding_nodes), 2)
        self.assertIn(mm1, collective_info[ag1].hiding_nodes)
        self.assertIn(mm2, collective_info[ag1].hiding_nodes)
        self.assertEqual(len(collective_info[ag2].hiding_nodes), 2)
        self.assertIn(mm2, collective_info[ag2].hiding_nodes)
        self.assertIn(mm3, collective_info[ag2].hiding_nodes)

        # Run bucketing
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        FileCheck().check_count(
            "all_gather_into_tensor_out", 1, exactly=False
        ).check_count("smith.ops.aten.mm.default", 3, exactly=True).run(
            str(traced.graph)
        )

    def test_can_bucket_with_convert_dtype_as_hiding_nodes(self):
        """
        Test that all_gathers can bucket when convert_element_type ops ARE the hiding nodes.

        Graph structure:
        ag1_start -> convert1 (hides ag1) -> ag1_wait -> ag2_start -> convert2 (hides ag2) -> ag2_wait

        The convert_element_type ops ARE hiding nodes - no matmuls.
        This tests that dependencies are transferred correctly when convert nodes are erased.
        """

        def func(a, b, c):
            group_name = "0"
            group_size = 1

            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = smith.ops.prims.convert_element_type.default(b, smith.float16)
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)

            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag3 = smith.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )

            mm = ag1_out @ ag1_out

            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)
            ag3_out = smith.ops._c10d_functional.wait_tensor(ag3)

            return ag1_out, ag2_out, ag3_out, mm

        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device, dtype=smith.float32)
            b = smith.ones(4, 4, device=self.device, dtype=smith.float32)
            c = smith.ones(4, 4, device=self.device, dtype=smith.float32)

            traced = make_fx(func)(a, b, c)

        # Find nodes
        ag1, ag2, ag3 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        convert1 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops.prims.convert_element_type.default,
        )[0]
        mm = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops.aten.mm.default,
        )[0]

        hiding_annotations = {
            ag1: convert1,
            ag2: mm,
            ag3: mm,
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from smith._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        graph_str = str(traced.graph)

        # Expect: ag1 (separate, hidden by convert1) -> wait -> ag2+ag3 bucketed (hidden by mm)
        # Check for pre_bucket (for ag2+ag3) and all_gather_into_tensor_out (bucketed)
        f = FileCheck()
        f.check("all_gather_into_tensor.default").check("wait_tensor")
        f.check("pre_bucket_all_gather").check("all_gather_into_tensor_out")
        f.run(graph_str)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestCrossPGOverlap(InductorTestCase):
    """
    Tests for cross-PG overlap scheduling.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from smith.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

        # Create two separate process groups for cross-PG testing
        cls.pg1 = dist.new_group(ranks=[0, 1])
        cls.pg2 = dist.new_group(ranks=[0, 1])
        cls.pg1_name = cls.pg1.group_name
        cls.pg2_name = cls.pg2.group_name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group(cls.pg1)
        dist.destroy_process_group(cls.pg2)
        dist.destroy_process_group()

    def test_cross_pg_prefetch_during_exposed_wait(self):
        """
        Test that ag2 on PG2 gets prefetched during exposed wait of ag1 on PG1.
        """
        pg1_name = self.pg1_name
        pg2_name = self.pg2_name

        def func(a, b):
            group_size = 1

            # First collective on PG1
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, pg1_name
            )
            ag1_out = smith.ops._c10d_functional.wait_tensor(ag1)
            mm1 = smith.mm(ag1_out[:4, :4], ag1_out[:4, :4])

            # Second collective on PG2
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, pg2_name
            )
            ag2_out = smith.ops._c10d_functional.wait_tensor(ag2)
            mm2 = smith.mm(ag2_out[:4, :4], ag2_out[:4, :4])

            return mm1 + mm2

        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device) * 2

            traced = make_fx(func)(a, b)

        # Find nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        wait1, wait2 = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.wait_tensor.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=smith.ops.aten.mm.default
        )

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            if "all_gather" in str(node.target):
                return 10.0  # Long collective to ensure exposed wait
            return 0.0

        # Run overlap scheduler
        from smith._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        )
        out = scheduler.run()
        FileCheck().check("%all_gather_into_tensor").check(
            "%all_gather_into_tensor"
        ).check("%wait_tensor").run(str(out.graph))

        self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 1)

    def test_two_queue_scheduling_off_path_nodes(self):
        """
        Test that off-path nodes (reduce_scatters whose results don't block compute)
        are scheduled near their original position rather than drifting to the end.

        Without two-queue scheduling, off-path nodes get domination=inf and drift
        to end. With two-queue, they stay near original position.
        """

        def func(a, b):
            group_name = "0"
            group_size = 2

            # On-path: all_gather whose result is used by compute
            ag = smith.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag_out = smith.ops._c10d_functional.wait_tensor(ag)

            # mm1 uses all_gather result (makes ag on-path)
            mm1 = smith.mm(a, ag_out[:4, :4])

            # Off-path: reduce_scatter result not used by further compute
            rs1 = smith.ops._c10d_functional.reduce_scatter_tensor(
                mm1, "sum", group_size, group_name
            )

            mm2 = smith.mm(a, a)
            rs2 = smith.ops._c10d_functional.reduce_scatter_tensor(
                mm2, "sum", group_size, group_name
            )

            mm3 = smith.mm(a, a)

            # Waits at end (like gradient outputs)
            rs1_out = smith.ops._c10d_functional.wait_tensor(rs1)
            rs2_out = smith.ops._c10d_functional.wait_tensor(rs2)

            return mm3.sum() + rs1_out.sum() + rs2_out.sum()

        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        from smith._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            if "all_gather" in str(node.target) or "reduce_scatter" in str(node.target):
                return 1.0
            return 0.0

        out = schedule_overlap_bucketing(
            traced, custom_runtime_estimation=custom_runtime
        )

        # Get scheduled order
        node_names = [n.name for n in out.graph.nodes if n.op == "call_function"]
        rs_starts = [
            i
            for i, name in enumerate(node_names)
            if "reduce_scatter" in name and "wait" not in name
        ]
        mm_positions = [i for i, name in enumerate(node_names) if name.startswith("mm")]

        # Off-path reduce_scatters should be interspersed with compute, not all at end
        last_mm = max(mm_positions)
        self.assertTrue(
            any(p < last_mm for p in rs_starts),
            f"Off-path reduce_scatters drifted to end: rs={rs_starts}, mm={mm_positions}, names={node_names}",
        )


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFusibleNodeOverlap(InductorTestCase):
    """Test that fusible nodes are used for overlapping with collectives."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from smith.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group()

    def test_fusible_nodes_hide_collective(self):
        """Test that fusible (non-mm) nodes can hide collectives."""

        def func(a):
            group_name = "0"
            group_size = 1

            ag = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            # Chain of pointwise ops - should be estimated and used for overlap
            b = a + 1
            b = b * 2
            b = b - 3
            ag_out = smith.ops._c10d_functional.wait_tensor(ag)
            return ag_out.sum() + b.sum()

        with FakeTensorMode():
            a = smith.ones(1024, 1024, device=self.device)
            traced = make_fx(func)(a)

        from smith._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
        )
        scheduler.run()

        # The collective should have hiding nodes (the pointwise chain)
        (ag_start,) = traced.graph.find_nodes(
            op="call_function",
            target=smith.ops._c10d_functional.all_gather_into_tensor.default,
        )
        info = scheduler.collective_info[ag_start]
        self.assertGreater(
            len(info.hiding_nodes), 0, "Fusible nodes should hide the collective"
        )

        # Verify graph structure: ag -> pointwise ops -> wait
        graph_str = str(traced.graph)
        FileCheck().check("all_gather_into_tensor").check("add").check("mul").check(
            "sub"
        ).check("wait_tensor").run(graph_str)

    def test_fusion_regions_hide_collective(self):
        """Test that fusion regions can hide collectives when enabled."""

        def func(a):
            group_name = "0"
            group_size = 1

            ag = smith.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = a + 1
            b = b * 2
            b = b - 3
            ag_out = smith.ops._c10d_functional.wait_tensor(ag)
            return ag_out.sum() + b.sum()

        with FakeTensorMode():
            a = smith.ones(1024, 1024, device=self.device)
            traced = make_fx(func)(a)

        from smith._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=True,
        )
        ag_start = next(iter(scheduler.collective_info.keys()))
        info = scheduler.collective_info[ag_start]
        scheduler.run()

        self.assertGreater(
            len(info.hiding_nodes), 0, "Fusion region should hide the collective"
        )

        # Verify graph structure: ag -> pointwise ops -> wait
        graph_str = str(traced.graph)
        FileCheck().check("all_gather_into_tensor").check("add").check("mul").check(
            "sub"
        ).check("wait_tensor").run(graph_str)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestOverlapSchedulingFixes(InductorTestCase):
    """
    Test cases for specific bug fixes in overlap scheduling.
    These tests would fail without their corresponding fixes.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from smith.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=16, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group()

    def test_no_self_dependency_cycle_with_dtype_conversion(self):
        """
        Test that bucketing collectives with dtype conversion doesn't create
        self-dependency cycles.

        This tests the fix in augmented_graph_helper.py that adds != new_node
        checks to prevent self-dependencies when merging nodes.

        The bug: When two convert_element_type nodes (inputs to all_gathers)
        have timeline dependencies between them and both get merged into
        _pre_bucket_all_gather, the dependency becomes a self-dependency
        which causes _stable_topological_sort to fail.
        """

        def func(a, b, c, d):
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 16

            # Multiple all_gathers with dtype conversion
            # The convert nodes will have timeline dependencies between them
            conv_a = smith.ops.prims.convert_element_type.default(a, smith.bfloat16)
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_a, group_size, group_name
            )

            conv_b = smith.ops.prims.convert_element_type.default(b, smith.bfloat16)
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_b, group_size, group_name
            )

            # Compute between all_gathers
            mm = smith.mm(c, d)

            conv_c = smith.ops.prims.convert_element_type.default(c, smith.bfloat16)
            ag3 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_c, group_size, group_name
            )

            conv_d = smith.ops.prims.convert_element_type.default(d, smith.bfloat16)
            ag4 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_d, group_size, group_name
            )

            # Wait for all
            w1 = smith.ops._c10d_functional.wait_tensor(ag1)
            w2 = smith.ops._c10d_functional.wait_tensor(ag2)
            w3 = smith.ops._c10d_functional.wait_tensor(ag3)
            w4 = smith.ops._c10d_functional.wait_tensor(ag4)

            return w1.sum() + w2.sum() + w3.sum() + w4.sum() + mm.sum()

        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device)
            c = smith.ones(4, 4, device=self.device)
            d = smith.ones(4, 4, device=self.device)

            traced = make_fx(func)(a, b, c, d)

        # Run full overlap scheduling with bucketing enabled
        # This would fail with AssertionError in _stable_topological_sort
        # before the self-dependency fix
        from smith._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=True,
            insert_overlap_deps=True,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=False,
        )
        # This should complete without cycle error
        result = scheduler.run()
        result.graph.lint()

    def test_no_cycle_with_fusion_regions_and_bucketing(self):
        """
        Test that fusion regions + bucketing doesn't create cycles.

        This tests multiple fixes:
        1. Self-dependency prevention (augmented_graph_helper.py)
        2. Track erased getitem nodes (const_fold.py, fusion_regions.py)
        3. Skip DCE during expansion (const_fold.py, fusion_regions.py)

        The scenario: Fusion regions collapse fusible ops into call_module nodes.
        When bucketing merges collectives, getitem nodes from fusion outputs
        get erased. Without proper tracking and DCE skip, this causes cycles
        or assertion failures.
        """

        def func(a, b, c, d):
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 16

            # Start collectives
            conv_a = smith.ops.prims.convert_element_type.default(a, smith.bfloat16)
            ag1 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_a, group_size, group_name
            )

            conv_b = smith.ops.prims.convert_element_type.default(b, smith.bfloat16)
            ag2 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_b, group_size, group_name
            )

            # Fusible compute chain (will become a fusion region)
            x = c + 1
            x = x * 2
            x = x - 3
            x = x / 4

            # Wait and use results
            w1 = smith.ops._c10d_functional.wait_tensor(ag1)
            w2 = smith.ops._c10d_functional.wait_tensor(ag2)

            # More collectives
            conv_c = smith.ops.prims.convert_element_type.default(c, smith.bfloat16)
            ag3 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_c, group_size, group_name
            )

            conv_d = smith.ops.prims.convert_element_type.default(d, smith.bfloat16)
            ag4 = smith.ops._c10d_functional.all_gather_into_tensor(
                conv_d, group_size, group_name
            )

            # Another fusible chain
            y = d + 1
            y = y * 2

            w3 = smith.ops._c10d_functional.wait_tensor(ag3)
            w4 = smith.ops._c10d_functional.wait_tensor(ag4)

            return w1.sum() + w2.sum() + w3.sum() + w4.sum() + x.sum() + y.sum()

        with FakeTensorMode():
            a = smith.ones(4, 4, device=self.device)
            b = smith.ones(4, 4, device=self.device)
            c = smith.ones(4, 4, device=self.device)
            d = smith.ones(4, 4, device=self.device)

            traced = make_fx(func)(a, b, c, d)

        # Run with fusion regions enabled - this exercises all the fixes
        from smith._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=True,
            insert_overlap_deps=True,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=True,  # Enable fusion regions
        )
        # This should complete without errors
        result = scheduler.run()
        result.graph.lint()


class TestForeachGroupsUnit(InductorTestCase):
    """Unit tests for _compute_foreach_groups and _pre_bucket_all_gather foreach optimization."""

    @unittest.skipIf(not HAS_GPU, "Requires GPU")
    def test_foreach_groups_correctness(self):
        """Test that foreach grouping computes correct groups and copies data correctly."""
        from smith._inductor.fx_passes.bucketing import (
            _ALL_DTYPES,
            _compute_foreach_groups,
            _pre_bucket_all_gather,
        )

        t1 = smith.randn(10, device="cuda")
        t2 = smith.randn(20, device="cuda", dtype=smith.float16)
        t3 = smith.randn(10, device="cuda")
        ag_ins = [t1, t2, t3]
        out_dtypes = [smith.float32, smith.float16, smith.float32]
        out_dtype_ints = [_ALL_DTYPES.index(d) for d in out_dtypes]

        # Mixed dtypes should produce groups with -1 delimiter
        groups = _compute_foreach_groups(ag_ins, out_dtypes)
        self.assertIsNotNone(groups)
        self.assertIn(-1, groups)

        # With and without groups should produce identical results
        result_with = _pre_bucket_all_gather(
            ag_ins, 2, "default", smith.float32, out_dtype_ints, 0, groups
        )
        result_without = _pre_bucket_all_gather(
            ag_ins, 2, "default", smith.float32, out_dtype_ints, 0, None
        )
        self.assertTrue(smith.allclose(result_with, result_without))


if __name__ == "__main__":
    run_tests()
