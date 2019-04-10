# Description:
#   TensorFlow Lite minimal example.

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "tflite_run",
    srcs = [
        "tflite_run.cc",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

