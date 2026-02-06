def define_targets(rules):
    rules.py_library(
        name = "smithgen",
        srcs = rules.glob(["**/*.py"]),
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":smithgen"],
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
    )
