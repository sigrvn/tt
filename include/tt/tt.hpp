#ifndef TT_HPP
#define TT_HPP

// tt - TinyTensor

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdfloat>

namespace tt {

enum class dtype {
  u8, u16, u32, u64,
  i8, i16, i32, i64,
  f16, f32, f64
};

template<dtype D>
struct dtype_traits;

// Specializations
template<> struct dtype_traits<dtype::u8>  { using type = std::uint8_t;  };
template<> struct dtype_traits<dtype::u16> { using type = std::uint16_t; };
template<> struct dtype_traits<dtype::u32> { using type = std::uint32_t; };
template<> struct dtype_traits<dtype::u64> { using type = std::uint64_t; };

template<> struct dtype_traits<dtype::i8>  { using type = std::int8_t;  };
template<> struct dtype_traits<dtype::i16> { using type = std::int16_t; };
template<> struct dtype_traits<dtype::i32> { using type = std::int32_t; };
template<> struct dtype_traits<dtype::i64> { using type = std::int64_t; };

template<> struct dtype_traits<dtype::f16> { using type = std::float16_t; };
template<> struct dtype_traits<dtype::f32> { using type = std::float32_t; };
template<> struct dtype_traits<dtype::f64> { using type = std::float64_t; };

// Convenience alias
template<dtype D>
using dtype_t = typename dtype_traits<D>::type;

template<const dtype D, const std::size_t... N>
class Tensor {
  public:
    // Properties
    static constexpr std::size_t dim = sizeof...(N);
    static constexpr std::array<std::size_t, dim> shape = {N...};

    static constexpr std::size_t size() {
      std::size_t sz = 1;
      for (const std::size_t n : {N...}) { sz *= n; }
      return sz;
    }

    using scalar_t = dtype_t<D>;
    using databuf_t = std::array<scalar_t, size()>;

    // Constructors
    Tensor() = default;
      
    Tensor(scalar_t fill) {
      std::fill(_data.begin(), _data.end(), fill);
    }
        
    // Standard iterator (linear element-wise)
    using iterator       = typename databuf_t::iterator;
    using const_iterator = typename databuf_t::const_iterator;

    iterator       begin()        { return _data.begin(); }
    iterator       end()          { return _data.end();   }
    const_iterator begin() const  { return _data.begin(); }
    const_iterator end()   const  { return _data.end();   }
    const_iterator cbegin() const { return _data.cbegin(); }
    const_iterator cend()   const { return _data.cend();   }
   
  private:
    databuf_t _data;
};

// Factory functions
template<dtype D, size_t... N>
Tensor<D, N...> zeros() {
    return Tensor<D, N...>(0.0);
}

template<dtype D, size_t... N>
Tensor<D, N...> ones() {
    return Tensor<D, N...>(1.0);
}

template<dtype D, size_t... N>
Tensor<D, N...> full(dtype_t<D> value) {
    return Tensor<D, N...>(value);
}

template<dtype D, size_t... N>
Tensor<D, N...> from(std::initializer_list<dtype_t<D>> values = {}) {
    Tensor<D, N...> t;
    if (values.size() > 0) {
      std::copy(values.begin(), values.end(), t.begin());
    }
    return t;
}

}

#endif /* TT_HPP */
