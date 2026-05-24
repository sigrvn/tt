#ifndef TT_HPP
#define TT_HPP

// tt - TinyTensor

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <stdfloat>
#include <type_traits>
#include <vector>

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

// Compile-time tensor (immutable)
template<const dtype D, const std::size_t... N>
class StaticTensor {
  public:
    // Properties
    static constexpr std::size_t dim() {
      return sizeof...(N);
    }

    static constexpr std::size_t size() {
      std::size_t sz = 1;
      for (const std::size_t n : {N...}) { sz *= n; }
      return sz;
    }

    static constexpr std::array<std::size_t, dim()> shape = {N...};

    using scalar_t  = dtype_t<D>;
    using databuf_t = std::array<scalar_t, size()>;

    // Constructors
    StaticTensor() = default;
      
    StaticTensor(scalar_t fill) {
      std::fill(_data.begin(), _data.end(), fill);
    }
        
    // Standard iterator (linear element-wise)
    using iterator       = typename databuf_t::iterator;
    using const_iterator = typename databuf_t::const_iterator;

    iterator       begin()        { return _data.begin();  }
    iterator       end()          { return _data.end();    }
    const_iterator begin() const  { return _data.begin();  }
    const_iterator end()   const  { return _data.end();    }
    const_iterator cbegin() const { return _data.cbegin(); }
    const_iterator cend()   const { return _data.cend();   }
   
  private:
    databuf_t _data;
};

template<std::size_t... N>
struct Shape {
  std::array<std::size_t, sizeof...(N)> dims = {N...};
};

// Runtime tensor (mutable)
template<const dtype D>
class DynamicTensor {
  public:
    // Properties
    std::size_t dim() {
      return shape.size();
    }
    
    std::size_t size() {
      return _data.size();
    }

    std::vector<std::size_t> shape;

    using scalar_t  = dtype_t<D>;
    using databuf_t = std::vector<scalar_t>;

    // Constructors
    DynamicTensor() {
      shape = {1, _data.capacity()};
    }

    template<std::size_t... N>
    DynamicTensor(Shape<N...>, std::initializer_list<scalar_t> values = {}) {
      shape.emplace_back({N...});
      _data.reserve(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>()));
      std::copy(values.begin(), values.end(), _data.begin());
    }
    
    DynamicTensor(scalar_t fill) {
      shape = {1, _data.capacity()};
      std::fill(_data.begin(), _data.end(), fill);
    }
        
    // Standard iterator (linear element-wise)
    using iterator       = typename databuf_t::iterator;
    using const_iterator = typename databuf_t::const_iterator;

    iterator       begin()        { return _data.begin();  }
    iterator       end()          { return _data.end();    }
    const_iterator begin() const  { return _data.begin();  }
    const_iterator end()   const  { return _data.end();    }
    const_iterator cbegin() const { return _data.cbegin(); }
    const_iterator cend()   const { return _data.cend();   }
   
  private:
    databuf_t _data;
};

// Conditional tensor alias based on template parameters
template<dtype D, std::size_t... N>
using Tensor = std::conditional_t<
  sizeof...(N) == 0,
  DynamicTensor<D>,
  StaticTensor<D, N...>
>;

// Factory functions
template<dtype D, std::size_t... N>
Tensor<D, N...> zeros() {
    return Tensor<D, N...>(0.0);
}

template<dtype D, std::size_t... N>
Tensor<D, N...> ones() {
    return Tensor<D, N...>(1.0);
}

template<dtype D, std::size_t... N>
Tensor<D, N...> full(dtype_t<D> value) {
    return Tensor<D, N...>(value);
}

template<dtype D, std::size_t... N>
Tensor<D, N...> from(std::initializer_list<dtype_t<D>> values = {}) {
    Tensor<D, N...> t;
    if (values.size() > 0) {
      std::copy(values.begin(), values.end(), t.begin());
    }
    return t;
}

} // namespace tt

#endif // TT_HPP
