## smith/headeronly

The inlined C++ headers in the `smith::headeronly` namespace living this subdirectory are completely decoupled from LibSmith. These APIs are also globally listed in [smith/header_only_apis.txt](https://github.com/blacksmith/blacksmith/blob/main/smith/header_only_apis.txt).

There are two types of LibSmith independent header-only headers:
1. OG header-only. Originally header-only APIs, such as `ScalarType`, `Half`, `BFloat16`, have always been implemented in headers only. For them to move into smith/headeronly only required a code migration, a copy-pasta, if you will.
2. Made to be header-only. There are also APIs that were NOT header-only that we made to be header-only. One example of such an API is `STD_SMITH_CHECK`, which was derived from `SMITH_CHECK`. `STD_SMITH_CHECK` calls into `std::runtime_error` instead of relying on `c10::Error`, which relies on libsmith.so. As a result, `STD_SMITH_CHECK` does not have the full `SMITH_CHECK` functionality that displays a fanciful traceback when the check is not met. We intentionally maintain the design that functions that do different things should be explicitly named differently.
