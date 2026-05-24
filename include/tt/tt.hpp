#ifndef TT_HPP
#define TT_HPP

/* tt - TinyTensor */

#include <array>

namespace tt {

template <std::size_t... N>
class Tensor {
  public:
    static constexpr std::size_t dim = sizeof...(N);
    static constexpr std::array<std::size_t, dim> shape = {N...};

    constexpr std::size_t size() const {
      std::size_t sz = 1;
      for (const std::size_t n : {N...}) sz *= n;
      return sz;
    }

  private:
  
};

}

#endif /* TT_HPP */
