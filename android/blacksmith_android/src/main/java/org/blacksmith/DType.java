package org.blacksmith;

/** Codes representing tensor data types. */
public enum DType {
  // NOTE: "jniCode" must be kept in sync with blacksmith_jni_common.cpp.
  // NOTE: Never serialize "jniCode", because it can change between releases.

  /** Code for dtype smith.uint8. {@link Tensor#dtype()} */
  UINT8(1),
  /** Code for dtype smith.int8. {@link Tensor#dtype()} */
  INT8(2),
  /** Code for dtype smith.int32. {@link Tensor#dtype()} */
  INT32(3),
  /** Code for dtype smith.float32. {@link Tensor#dtype()} */
  FLOAT32(4),
  /** Code for dtype smith.int64. {@link Tensor#dtype()} */
  INT64(5),
  /** Code for dtype smith.float64. {@link Tensor#dtype()} */
  FLOAT64(6),
  ;

  final int jniCode;

  DType(int jniCode) {
    this.jniCode = jniCode;
  }
}
