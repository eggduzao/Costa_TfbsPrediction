set(
  MODEL_TRACER_DIR
  "${SMITH_ROOT}/smith/csrc/jit/mobile/model_tracer")

list(APPEND MODEL_TRACER_SOURCES "")

append_filelist("smith_mobile_tracer_sources" MODEL_TRACER_SOURCES)

add_executable(
  model_tracer
  ${MODEL_TRACER_SOURCES})

target_link_libraries(model_tracer PRIVATE smith)

install(TARGETS model_tracer DESTINATION bin)
