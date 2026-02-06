from __future__ import annotations

import copy
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, Optional, TYPE_CHECKING

import smith.fx
from smith._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from smith._dynamo.utils import dynamo_timed
from smith._inductor.cpp_builder import normalize_path_separator
from smith._inductor.cudagraph_utils import BoxedDeviceIndex
from smith._inductor.runtime.cache_dir_utils import temporary_cache_dir
from smith._inductor.utils import BoxedBool, InputType
from smith._subclasses import FakeTensorMode
from smith.fx.experimental.symbolic_shapes import ShapeEnv

from . import config


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from smith.compiler._cache import CacheInfo
    from smith.fx import GraphModule


log = logging.getLogger(__name__)


class CompiledArtifact(ABC):
    """
    CompiledArtifact class represents the inductor cache artifacts that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the cached artifact.
    """

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: Optional[tuple[bytes, CacheInfo]],
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @abstractmethod
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None: ...

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            # If format is unpacked, it must be a CacheCompiledArtifact
            return CacheCompiledArtifact.load(path=path, format=format)

        assert format == "binary"
        with open(path, "rb") as file:
            from smith.utils._appending_byte_serializer import BytesReader

            from .codecache import smith_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                assert reader.read_bytes() == smith_key()
                artifact = reader.read_bytes()
                assert reader.is_finished()
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                assert reader.read_bytes() == smith_key()
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                assert reader.is_finished()
                smith.compiler.load_cache_artifacts(artifact_bytes)
                return CacheCompiledArtifact._load_impl(nullcontext(), key)
            else:
                raise RuntimeError(
                    "Invalid header, expected CacheCompiledArtifact or AOTCompiledArtifact, got: "
                    + header.decode("utf-8")
                )


class CacheCompiledArtifact(CompiledArtifact):
    """
    CompiledArtifact that depends on smith.compiler.save_cache_artifacts
    """

    CACHE_HEADER = bytes("CacheCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: Optional[tuple[bytes, CacheInfo]],
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        return self._compiled_fn(*args)

    def is_saveable(self) -> bool:
        if self._artifacts is None:
            return False
        _, cache_info = self._artifacts
        # 0 means nothing was saved
        # >1 means multiple artifacts were saved, which is concerning
        # (we only expect one)
        return len(cache_info.aot_autograd_artifacts) == 1

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if self._artifacts is None:
                raise RuntimeError(
                    "CompiledArtifact.save failed to save since there's no artifact to save"
                )
            artifact_bytes, cache_info = self._artifacts
            if len(cache_info.aot_autograd_artifacts) == 0:
                raise RuntimeError(
                    f"CompiledArtifact.save failed to save due to no aot_autograd artifacts. "
                    f"This likely means there was something that was not serializable in the "
                    f"graph passed to standalone_compile. This can generally be fixed by "
                    f"ensuring that your model only uses constructs that are serializable. "
                    f"{cache_info}"
                )
            if len(cache_info.aot_autograd_artifacts) > 1:
                raise AssertionError(
                    f"CompiledArtifact.save failed to save because there was more than one "
                    f"artifact but we only expected one. {cache_info}"
                )
            key = cache_info.aot_autograd_artifacts[0]

            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)

                from smith.utils._appending_byte_serializer import BytesWriter

                from .codecache import smith_key

                writer = BytesWriter()
                writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
                writer.write_bytes(smith_key())
                writer.write_str(key)
                writer.write_bytes(artifact_bytes)

                from smith._inductor.codecache import write_atomic

                write_atomic(path, writer.to_bytes())
            else:
                assert format == "unpacked"
                if os.path.exists(path):
                    assert os.path.isdir(path)
                    shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = smith.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    assert loaded_cache_info is not None
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        assert os.path.exists(subdir)
                        for path in sorted(os.listdir(subdir)):
                            with open(os.path.join(subdir, path), "rb") as f:
                                graph = pickle.load(f)
                            output_file = graph.write_to_disk()
                            log.info("Output code written to: %s", output_file)

    @staticmethod
    def _load_impl(
        cache_dir_ctx: AbstractContextManager[Any], key: str
    ) -> CompiledArtifact:
        with (
            cache_dir_ctx,
            config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
        ):
            with smith._funcsmith.config.patch(strict_autograd_cache=True):
                from smith._funcsmith._aot_autograd.autograd_cache import (
                    AOTAutogradCache,
                )

                result = AOTAutogradCache._lookup(
                    key,
                    local=True,
                    remote=False,
                    args=[],
                    cache_info={},
                    aot_config=None,
                )

            assert result is not None
            (entry, _) = result

            from .compile_fx import _CompileFxKwargs

            fx_config = _CompileFxKwargs(
                cudagraphs=BoxedBool(False),
                boxed_forward_device_index=BoxedDeviceIndex(0),
            )

            context = smith._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
            with smith._guards.tracing(context):
                compiled_fn = entry.wrap_post_compile(
                    [], entry.sanitized_aot_config, fx_config
                )
        return CacheCompiledArtifact(lambda *args: compiled_fn(list(args)), None)

    @staticmethod
    def _prepare_load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> tuple[str, AbstractContextManager[Any]]:
        """
        Do format specific prep and loads, return a context manager and key
        """
        path = normalize_path_separator(path)
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                assert not os.path.isdir(path)
                with open(path, "rb") as file:
                    artifacts = file.read()
                from smith.utils._appending_byte_serializer import BytesReader

                from .codecache import smith_key

                reader = BytesReader(artifacts)
                assert reader.read_bytes() == smith_key()
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                assert reader.is_finished()

                smith.compiler.load_cache_artifacts(artifact_bytes)
                return key, nullcontext()
            else:
                assert format == "unpacked"
                assert os.path.isdir(path)
                autograd_cache_dir = os.path.join(path, "aotautograd")
                assert os.path.isdir(autograd_cache_dir)
                files = list(os.listdir(autograd_cache_dir))
                assert len(files) == 1
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)
                return key, cache_dir_ctx

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        key, cache_dir_ctx = CacheCompiledArtifact._prepare_load(
            path=path, format=format
        )
        return CacheCompiledArtifact._load_impl(cache_dir_ctx, key)


class AOTCompiledArtifact(CompiledArtifact):
    """
    Similar to CompiledArtifact, but the object is a single, bundled precompiled function.
    This object is always a serializable callable function.

    This object is essentially a wrapper for BundledAOTAutogradSerializableCallable, which
    is used by smith._dynamo.aot_compile for AOT Precompilation.
    """

    AOT_HEADER = bytes("AOTCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
    ):
        self.inner_fn = BundledAOTAutogradSerializableCallable(compiled_fn)
        self._artifacts = (
            None  # We don't need artifacts, the inner object handles everything
        )

    @staticmethod
    def from_bundled_callable(
        bundled_fn: BundledAOTAutogradSerializableCallable,
    ) -> AOTCompiledArtifact:
        return AOTCompiledArtifact(bundled_fn.compiled_fn)

    def __call__(self, *args: Any) -> Any:
        return self.inner_fn(*args)

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        result_bytes = self.serialize()
        from smith.utils._appending_byte_serializer import BytesWriter

        from .codecache import smith_key

        writer = BytesWriter()
        writer.write_bytes(AOTCompiledArtifact.AOT_HEADER)
        writer.write_bytes(smith_key())
        writer.write_bytes(result_bytes)

        from smith._inductor.codecache import write_atomic

        # Save a sentinel file to indicate that this is AOT
        write_atomic(path, writer.to_bytes())

    def serialize(self) -> bytes:
        return BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
            self.inner_fn
        )

    @staticmethod
    def deserialize(result_bytes: bytes) -> AOTCompiledArtifact:
        deserialized = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                result_bytes
            )
        )
        assert isinstance(deserialized, BundledAOTAutogradSerializableCallable)
        return AOTCompiledArtifact.from_bundled_callable(deserialized)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        with open(path, "rb") as file:
            from smith.utils._appending_byte_serializer import BytesReader

            from .codecache import smith_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            assert header == AOTCompiledArtifact.AOT_HEADER
            assert reader.read_bytes() == smith_key()
            artifact = reader.read_bytes()
            assert reader.is_finished()
            return AOTCompiledArtifact.deserialize(artifact)


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: Any,
    options: Any,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
) -> CompiledArtifact:
    """
    Implementation of smith.inductor.standalone_compile
    """
    from smith.compiler._cache import CacheArtifactManager

    from .compile_fx import compile_fx

    ignore_shape_env = False
    if dynamic_shapes == "from_example_inputs":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # tells compile_fx to ignore the shape_envs on the ambient context
        # and the graph_module.
        ignore_shape_env = True
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a smith.compile backend.
        context = smith._guards.TracingContext.get()
        assert context.fake_mode is not None
        fake_mode = context.fake_mode
    elif dynamic_shapes == "from_graph":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        assert last_node.op == "output"
        assert len(last_node.args) == 1

        def handle_node(node: smith.fx.Node) -> None:
            nonlocal fake_mode
            if "example_value" in node.meta:
                maybe_tensor = node.meta["example_value"]
                if isinstance(maybe_tensor, smith._subclasses.fake_tensor.FakeTensor):
                    fake_mode = maybe_tensor.fake_mode

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        if isinstance(last_node.args[0], smith.fx.Node):
            handle_node(last_node.args[0])
        else:
            for node in last_node.args[0]:
                handle_node(node)

    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )

    context = smith._guards.TracingContext(fake_mode)
    with (
        smith._guards.tracing(context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        smith._funcsmith.config.patch("bundled_autograd_cache", aot),
    ):
        # compile_fx can mutate gm
        gm = copy.deepcopy(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        assert callable(compiled_fn)
        if aot:
            if not hasattr(compiled_fn, "serialize"):
                raise RuntimeError(
                    "Compiled function should have serialize method when aot=True"
                )
            return AOTCompiledArtifact(compiled_fn)
        artifacts = smith.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with SMITH_LOGS=+smith._inductor.codecache to identify the problem"
            )

    return CacheCompiledArtifact(compiled_fn, artifacts)
