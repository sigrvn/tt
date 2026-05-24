#ifndef TT_HPP
#define TT_HPP

// tt - TinyTensor

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdfloat>
#include <stdexcept>

namespace tt {

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
struct Range {
  T start = 0;
  T end   = 0;
  T step  = 1;
};

template<std::size_t... N>
struct Shape {
  std::array<std::size_t, sizeof...(N)> dims = {N...};
};

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

template<typename Derived, dtype D>
class TensorBase {
  public:
    using scalar_t = dtype_t<D>;

    template<typename It>
    void assign(It first, It last) {
      std::copy(first, last, derived().begin());
    }

    template<typename F>
    requires std::invocable<F&, std::size_t> &&
             std::convertible_to<std::invoke_result_t<F&, std::size_t>, scalar_t>
    void assign_fn(F&& fill_fn) {
        auto& d = derived();
        for (size_t i = 0; i < d.size(); ++i) {
          d[i] = static_cast<scalar_t>(std::invoke(fill_fn, i));
        }
    }

  protected:
    Derived&       derived()       { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// Compile-time tensor (immutable)
template<dtype D, std::size_t... N>
class StaticTensor : public TensorBase<StaticTensor<D, N...>, D> {
  public:
    // Properties
    static constexpr std::size_t dim() { return sizeof...(N); }
    static constexpr std::size_t size() {
      auto pack = {N...};
      return std::accumulate(pack.begin(), pack.end(),
                             1, std::multiplies<std::size_t>());
    }
    static constexpr std::array<std::size_t, dim()> shape = {N...};

    using scalar_t  = dtype_t<D>;
    using databuf_t = std::array<scalar_t, size()>;

    scalar_t& operator[](size_t i)       { return _data[i]; }
    scalar_t  operator[](size_t i) const { return _data[i]; }
    
    const databuf_t& data() const { return _data; }

    // Constructors
    explicit StaticTensor(scalar_t fill = 0) { _data.fill(fill); }

    template<typename It>
    StaticTensor(It first, It last) { this->assign(first, last); }

    template<typename F>
    requires std::invocable<F&, std::size_t> &&
             std::convertible_to<std::invoke_result_t<F&, std::size_t>, scalar_t>
    explicit StaticTensor(F&& fill_fn) { this->assign_fn(std::forward<F>(fill_fn)); }
    
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

// Runtime tensor (mutable)
template<dtype D>
class DynamicTensor : public TensorBase<DynamicTensor<D>, D> {
  public:
    // Properties
    std::size_t dim()  { return shape.size(); }
    std::size_t size() { return _data.size(); }
    std::vector<std::size_t> shape;

    using scalar_t  = dtype_t<D>;
    using databuf_t = std::vector<scalar_t>;

    scalar_t& operator[](size_t i)       { return _data[i]; }
    scalar_t  operator[](size_t i) const { return _data[i]; }

    const databuf_t& data() const { return _data; }

    // Constructors
    explicit DynamicTensor(std::initializer_list<std::size_t> shape, scalar_t fill = 0)
      : shape(shape),
        _data(std::accumulate(shape.begin(), shape.end(),
                              1, std::multiplies<std::size_t>()), fill) {}

    template<typename It>
    DynamicTensor(std::initializer_list<std::size_t> shape, It first, It last)
      : shape(shape),
        _data(first, last) {}

    template<typename F>
    requires std::invocable<F&, std::size_t> &&
             std::convertible_to<std::invoke_result_t<F&, std::size_t>, scalar_t>
    DynamicTensor(std::initializer_list<std::size_t> shape, F&& fill_fn)
      : shape(shape),
        _data(std::accumulate(shape.begin(), shape.end(),
                              1, std::multiplies<std::size_t>()))
        {
          this->assign_fn(std::forward<F>(fill_fn));
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
  Tensor<D, N...> zeros() { return Tensor<D, N...>(0); }

template<dtype D>
  Tensor<D> zeros(std::initializer_list<std::size_t> shape)
  { return Tensor<D>(shape, 0); }

template<dtype D, std::size_t... N>
  Tensor<D, N...> ones() { return Tensor<D, N...>(1); }

template<dtype D>
  Tensor<D> ones(std::initializer_list<std::size_t> shape)
  { return Tensor<D>(shape, 1); }

template<dtype D, std::size_t... N>
  Tensor<D, N...> full(dtype_t<D> value) { return Tensor<D, N...>(value); }

template<dtype D>
  Tensor<D> full(std::initializer_list<std::size_t> shape, dtype_t<D> value)
  { return Tensor<D>(shape, value); }

template<dtype D, size_t... N>
  Tensor<D, N...> from(std::initializer_list<dtype_t<D>> values) {
    if (values.size() != (N * ... * 1))
        throw std::invalid_argument("initializer_list size does not match tensor shape");
    return Tensor<D, N...>(values.begin(), values.end());
  }
  
template<dtype D>
  Tensor<D> from(std::initializer_list<std::size_t> shape,
                 std::initializer_list<dtype_t<D>> values)
  { return Tensor<D>(shape, values.begin(), values.end()); }

template<dtype D, size_t... N, Numeric T>
Tensor<D, N...> arange(const Range<T> r) {
  using scalar_t = dtype_t<D>;
  scalar_t start = static_cast<scalar_t>(r.start);
  scalar_t end   = static_cast<scalar_t>(r.end);
  scalar_t step  = static_cast<scalar_t>(r.step);

  if constexpr (sizeof...(N) > 0) {
      // Static: use index function, no heap allocation
      return Tensor<D, N...>([&](size_t i) {
          return start + step * static_cast<scalar_t>(i);
      });
  } else {
      // Dynamic: build vector, infer shape from range
      std::vector<scalar_t> values;
      for (scalar_t val = start; val < end; val += step)
          values.push_back(val);
      return Tensor<D>({values.size()}, values.begin(), values.end());
  }
}

} // namespace tt

#endif // TT_HPP
