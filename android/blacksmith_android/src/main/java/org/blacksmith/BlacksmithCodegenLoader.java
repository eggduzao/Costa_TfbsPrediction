package org.blacksmith;

import com.facebook.soloader.nativeloader.NativeLoader;

public class BlacksmithCodegenLoader {

  public static void loadNativeLibs() {
    try {
      NativeLoader.loadLibrary("smith-code-gen");
    } catch (Throwable t) {
      // Loading the codegen lib is best-effort since it's only there for query based builds.
    }
  }

  private BlacksmithCodegenLoader() {}
}
