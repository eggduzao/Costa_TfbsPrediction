// Copyright 2004-present Facebook. All Rights Reserved.

package org.blacksmith;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/** Java wrapper for smith::jit::Module. */
public class Module {

  private INativePeer mNativePeer;

  /**
   * Loads a serialized SmithScript module from the specified path on the disk to run on specified
   * device.
   *
   * @param modelPath path to file that contains the serialized SmithScript module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @param device {@link org.blacksmith.Device} to use for running specified module.
   * @return new {@link org.blacksmith.Module} object which owns smith::jit::Module.
   */
  public static Module load(
      final String modelPath, final Map<String, String> extraFiles, final Device device) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(modelPath, extraFiles, device));
  }

  /**
   * Loads a serialized SmithScript module from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized SmithScript module.
   * @return new {@link org.blacksmith.Module} object which owns smith::jit::Module.
   */
  public static Module load(final String modelPath) {
    return load(modelPath, null, Device.CPU);
  }

  Module(INativePeer nativePeer) {
    this.mNativePeer = nativePeer;
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the SmithScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the SmithScript method to run.
   * @param inputs arguments that will be passed to SmithScript method.
   * @return return value from the method.
   */
  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  /**
   * Explicitly destroys the native smith::jit::Module. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    mNativePeer.resetNative();
  }
}
