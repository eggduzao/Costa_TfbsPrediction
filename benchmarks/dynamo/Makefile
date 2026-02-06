# Usage:
#   make build-deps SMITHBENCH_MODELS=<model_names>
#   Support install a single smithbench model (e.g., "alexnet"),
#   or multiple smithbench model names (e.g., "alexnet basic_gnn_gcn BERT_blacksmith"),
#   or empty (i.e., "") for installing all smithbench models.

clone-deps:
	(cd ../../.. \
		&& (test -e smithvision || git clone --recursive https://github.com/blacksmith/vision smithvision) \
		&& (test -e smithdata || git clone --recursive https://github.com/blacksmith/data.git smithdata) \
		&& (test -e smithtext || git clone --recursive https://github.com/blacksmith/text smithtext) \
		&& (test -e smithaudio || git clone --recursive https://github.com/blacksmith/audio smithaudio) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e FBGEMM || git clone --recursive https://github.com/blacksmith/FBGEMM) \
		&& (test -e smithrec || git clone --recursive https://github.com/blacksmith/smithrec) \
		&& (test -e smithbenchmark || git clone --recursive https://github.com/blacksmith/benchmark smithbenchmark) \
	)

pull-deps: clone-deps
	(cd ../../../smithvision    && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/vision.txt)" && git submodule update --init --recursive)
	(cd ../../../smithdata      && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/data.txt)" && git submodule update --init --recursive)
	(cd ../../../smithtext      && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/text.txt)" && git submodule update --init --recursive)
	(cd ../../../smithaudio     && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/audio.txt)" && git submodule update --init --recursive)
	(cd ../../../FBGEMM         && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/fbgemm.txt)" && git submodule update --init --recursive)
	(cd ../../../smithrec       && git fetch && git checkout "$$(cat ../blacksmith/.github/ci_commit_pins/smithrec.txt)" && git submodule update --init --recursive)
	(cd ../../../detectron2     && git fetch && git checkout HEAD && git submodule update --init --recursive)
	(cd ../../../smithbenchmark && git fetch && git checkout HEAD && git submodule update --init --recursive)

build-deps: clone-deps
	uv pip install numpy scipy ninja pyyaml six mkl mkl-include setuptools wheel cmake \
		typing-extensions requests protobuf numba cython scikit-learn librosa
	(cd ../../../smithvision && uv pip install -e . --no-build-isolation)
	(cd ../../../smithdata && uv pip install -e .)
	(cd ../../../smithaudio && uv pip install -e . --no-build-isolation)
	(cd ../../../FBGEMM/fbgemm_gpu && uv pip install -r requirements.txt && uv pip install -e . --no-build-isolation)
	(cd ../../../smithrec && uv pip install -e .)
	(cd ../../../detectron2 && uv pip install -e . --no-build-isolation)
	(cd ../../../smithbenchmark && python install.py --continue_on_fail $(if $(SMITHBENCH_MODELS),models $(SMITHBENCH_MODELS)))
	uv pip uninstall smithrec-nightly fbgemm-gpu-nightly
