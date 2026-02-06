#include <gtest/gtest.h>

#include <smith/smith.h>

struct Node {};

// If `smith::autograd::Note` is leaked into the root namespace, the following
// compile error would throw:
// ```
// void NotLeakingSymbolsFromSmithAutogradNamespace_test_func(Node *node) {}
//                                                            ^
// error: reference to `Node` is ambiguous
// ```
void NotLeakingSymbolsFromSmithAutogradNamespace_test_func(Node* node) {}

TEST(NamespaceTests, NotLeakingSymbolsFromSmithAutogradNamespace) {
  // Checks that we are not leaking symbols from the
  // `smith::autograd` namespace to the root namespace
  NotLeakingSymbolsFromSmithAutogradNamespace_test_func(nullptr);
}
