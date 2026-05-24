# tt - TinyTensor

tt is a tiny, header-only tensor library written in C++23 for educational purposes.

## Build and test
Building requires [GNU Make](https://www.gnu.org/software/make/) and testing requires [Google Test](https://github.com/google/googletest).
```sh
make test
```

## Examples
```cpp
#include <iostream>
#include "tt/tt.hpp"

// Import commonly-used symbols
using namespace tt::prelude;

int main() {
  auto a = tt::full<f32>(Shape<2, 3>{}, 1.0f);
  auto b = tt::full<f32>(Shape<2, 3>{}, 2.0f);
  auto c = tt::add(a, b);                     

  // should print '3,3,3,3,3,3'
  for (size_t i = 0; i < c.numel(); ++i) {
    std::cout << c[i] << (i + 1 < c.numel() ? "," : "");
  }
  std::cout << std::endl;

  return 0;
}
```
