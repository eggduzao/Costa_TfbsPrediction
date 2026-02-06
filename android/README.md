# Android

## Demo applications and tutorials

Please refer to [meta-blacksmith/execusmith-examples](https://github.com/meta-blacksmith/execusmith-examples/tree/main/dl3/android/DeepLabV3Demo) for the Android demo app based on [ExecuSmith](https://github.com/blacksmith/execusmith).

Please join our [Discord](https://discord.com/channels/1334270993966825602/1349854760299270284) for any questions.

## Publishing

##### Release
Release artifacts are published to jcenter:

```groovy
repositories {
    jcenter()
}

# lite interpreter build
dependencies {
    implementation 'org.blacksmith:blacksmith_android_lite:1.10.0'
    implementation 'org.blacksmith:blacksmith_android_smithvision_lite:1.10.0'
}

# full jit build
dependencies {
    implementation 'org.blacksmith:blacksmith_android:1.10.0'
    implementation 'org.blacksmith:blacksmith_android_smithvision:1.10.0'
}
```

##### Nightly

Nightly(snapshots) builds are published every night from `master` branch to [nexus sonatype snapshots repository](https://oss.sonatype.org/#nexus-search;quick~blacksmith_android)

To use them repository must be specified explicitly:
```groovy
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots"
    }
}

# lite interpreter build
dependencies {
    ...
    implementation 'org.blacksmith:blacksmith_android_lite:1.12.0-SNAPSHOT'
    implementation 'org.blacksmith:blacksmith_android_smithvision_lite:1.12.0-SNAPSHOT'
    ...
}

# full jit build
dependencies {
    ...
    implementation 'org.blacksmith:blacksmith_android:1.12.0-SNAPSHOT'
    implementation 'org.blacksmith:blacksmith_android_smithvision:1.12.0-SNAPSHOT'
    ...
}
```
The current nightly(snapshots) version is the value of `VERSION_NAME` in `gradle.properties` in current folder, at this moment it is `1.8.0-SNAPSHOT`.

## Building Blacksmith Android from Source

In some cases you might want to use a local build of blacksmith android, for example you may build custom libsmith binary with another set of operators or to make local changes.

For this you can use `./scripts/build_blacksmith_android.sh` script.
```bash
git clone https://github.com/blacksmith/blacksmith.git
cd blacksmith
git submodule update --init --recursive
bash ./scripts/build_blacksmith_android.sh
```

The workflow contains several steps:

1\. Build libsmith for android for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64)

2\. Create symbolic links to the results of those builds:
`android/blacksmith_android/src/main/jniLibs/${abi}` to the directory with output libraries
`android/blacksmith_android/src/main/cpp/libsmith_include/${abi}` to the directory with headers. These directories are used to build `libblacksmith.so` library that will be loaded on android device.

3\. And finally run `gradle` in `android/blacksmith_android` directory with task `assembleRelease`

Script requires that Android SDK, Android NDK and gradle are installed.
They are specified as environment variables:

`ANDROID_HOME` - path to [Android SDK](https://developer.android.com/studio/command-line/sdkmanager.html)

`ANDROID_NDK` - path to [Android NDK](https://developer.android.com/studio/projects/install-ndk). It's recommended to use NDK 21.x.

`GRADLE_HOME` - path to [gradle](https://gradle.org/releases/)


After successful build you should see the result as aar file:

```bash
$ find blacksmith_android/build/ -type f -name *aar
blacksmith_android/build/outputs/aar/blacksmith_android.aar
blacksmith_android_smithvision/build/outputs/aar/blacksmith_android.aar
```

It can be used directly in android projects, as a gradle dependency:
```groovy
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    implementation(name:'blacksmith_android', ext:'aar')
    implementation(name:'blacksmith_android_smithvision', ext:'aar')
    ...
    implementation 'com.facebook.soloader:nativeloader:0.10.5'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.2.2'
}
```
We also have to add all transitive dependencies of our aars.
As `blacksmith_android` [depends](https://github.com/blacksmith/blacksmith/blob/master/android/blacksmith_android/build.gradle#L76-L77) on `'com.facebook.soloader:nativeloader:0.10.5'` and `'com.facebook.fbjni:fbjni-java-only:0.2.2'`, we need to add them.
(In case of using maven dependencies they are added automatically from `pom.xml`).

## Linking to prebuilt libsmith library from gradle dependency

In some cases, you may want to use libsmith from your android native build.
You can do it without building libsmith android, using native libraries from Blacksmith android gradle dependency.
For that, you will need to add the next lines to your gradle build.
```groovy
android {
...
    configurations {
       extractForNativeBuild
    }
...
    compileOptions {
        externalNativeBuild {
            cmake {
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
...
    externalNativeBuild {
        cmake {
            path "CMakeLists.txt"
        }
    }
}

dependencies {
    extractForNativeBuild('org.blacksmith:blacksmith_android:1.10.0')
}

task extractAARForNativeBuild {
    doLast {
        configurations.extractForNativeBuild.files.each {
            def file = it.absoluteFile
            copy {
                from zipTree(file)
                into "$buildDir/$file.name"
                include "headers/**"
                include "jni/**"
            }
        }
    }
}

tasks.whenTaskAdded { task ->
  if (task.name.contains('externalNativeBuild')) {
    task.dependsOn(extractAARForNativeBuild)
  }
}
```

blacksmith_android aar contains headers to link in `headers` folder and native libraries in `jni/$ANDROID_ABI/`.
As Blacksmith native libraries use `ANDROID_STL` - we should use `ANDROID_STL=c++_shared` to have only one loaded binary of STL.

The added task will unpack them to gradle build directory.

In your native build you can link to them adding these lines to your CMakeLists.txt:


```cmake
# Relative path of gradle build directory to CMakeLists.txt
set(build_DIR ${CMAKE_SOURCE_DIR}/build)

file(GLOB BLACKSMITH_INCLUDE_DIRS "${build_DIR}/blacksmith_android*.aar/headers")
file(GLOB BLACKSMITH_LINK_DIRS "${build_DIR}/blacksmith_android*.aar/jni/${ANDROID_ABI}")

set(BUILD_SUBDIR ${ANDROID_ABI})
target_include_directories(${PROJECT_NAME} PRIVATE
  ${BLACKSMITH_INCLUDE_DIRS}
)

find_library(BLACKSMITH_LIBRARY blacksmith_jni
  PATHS ${BLACKSMITH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
  PATHS ${BLACKSMITH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${PROJECT_NAME}
  ${BLACKSMITH_LIBRARY}
  ${FBJNI_LIBRARY})

```
If your CMakeLists.txt file is located in the same directory as your build.gradle, `set(build_DIR ${CMAKE_SOURCE_DIR}/build)` should work for you. But if you have another location of it, you may need to change it.

After that, you can use libsmith C++ API from your native code.
```cpp
#include <string>
#include <ATen/NativeFunctions.h>
#include <smith/script.h>
namespace blacksmith_testapp_jni {
namespace {
    struct JITCallGuard {
      c10::InferenceMode guard;
      smith::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };
}

void loadAndForwardModel(const std::string& modelPath) {
  JITCallGuard guard;
  smith::jit::Module module = smith::jit::load(modelPath);
  module.eval();
  smith::Tensor t = smith::randn({1, 3, 224, 224});
  c10::IValue t_out = module.forward({t});
}
}
```

To load smithscript model for mobile we need some special setup which is placed in `struct JITCallGuard` in this example. It may change in future, you can track the latest changes keeping an eye in our [blacksmith android jni code]([https://github.com/blacksmith/blacksmith/blob/master/android/blacksmith_android/src/main/cpp/blacksmith_jni_jit.cpp#L28)

## Blacksmith Android API Javadoc

You can find more details about the Blacksmith Android API in the [Javadoc](https://blacksmith.org/javadoc/).
