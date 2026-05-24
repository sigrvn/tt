#ifndef TT_HPP
#define TT_HPP

// tt - TinyTensor

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <stdfloat>
#include <stdexcept>
#include <vector>

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
    static constexpr std::size_t  rank  = sizeof...(N);
    static constexpr std::size_t  numel = (N * ... * 1);
    std::array<std::size_t, rank> dims  = {N...};

    template<size_t... M>
    static constexpr bool equal(Shape<M...>) {
        return std::is_same_v<Shape<N...>, Shape<M...>>;
    }

    constexpr size_t operator[](size_t i) const { return dims[i]; }
};

enum class dtype { u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64 };

template<dtype D> struct dtype_traits;

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
        for (size_t i = 0; i < d.numel(); ++i) {
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
    using scalar_t  = dtype_t<D>;
    using shape_t   = Shape<N...>;

    static constexpr std::size_t _numel = shape_t::numel;
    using databuf_t = std::array<scalar_t, _numel>;

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

    scalar_t& operator[](size_t i)       { return _data[i]; }
    scalar_t  operator[](size_t i) const { return _data[i]; }

    std::size_t      numel() const { return _numel; }
    const databuf_t& data()  const { return _data;  }
    static constexpr shape_t shape = {};
   
  private:
    databuf_t _data;
};

// Runtime tensor (mutable)
template<dtype D>
class DynamicTensor : public TensorBase<DynamicTensor<D>, D> {
  public:
    // Properties
    using scalar_t  = dtype_t<D>;
    using databuf_t = std::vector<scalar_t>;

    // Constructors
    explicit DynamicTensor(std::vector<std::size_t> shape, scalar_t fill = 0)
      : shape(shape),
        _data(std::accumulate(shape.begin(), shape.end(),
                              1, std::multiplies<std::size_t>()), fill) {}

    template<typename It>
    DynamicTensor(std::vector<std::size_t> shape, It first, It last)
      : shape(shape),
        _data(first, last) {}

    template<typename F>
    requires std::invocable<F&, std::size_t> &&
             std::convertible_to<std::invoke_result_t<F&, std::size_t>, scalar_t>
    DynamicTensor(std::vector<std::size_t> shape, F&& fill_fn)
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

    scalar_t& operator[](size_t i)       { return _data[i]; }
    scalar_t  operator[](size_t i) const { return _data[i]; }

    std::size_t      numel() const { return _data.size(); }
    const databuf_t& data()  const { return _data; }
    std::vector<std::size_t> shape;
   
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
Tensor<D, N...> zeros(Shape<N...>) {
  return Tensor<D, N...>(0);
}

template<dtype D>
Tensor<D> zeros(std::vector<std::size_t> shape) {
  return Tensor<D>(shape, 0);
}

template<dtype D, std::size_t... N>
Tensor<D, N...> ones(Shape<N...>) {
  return Tensor<D, N...>(1);
}

template<dtype D>
Tensor<D> ones(std::vector<std::size_t> shape) {
  return Tensor<D>(shape, 1);
}

template<dtype D, std::size_t... N>
Tensor<D, N...> full(Shape<N...>, dtype_t<D> value) {
  return Tensor<D, N...>(value);
}

template<dtype D>
Tensor<D> full(std::vector<std::size_t> shape, dtype_t<D> value) {
  return Tensor<D>(shape, value);
}

template<dtype D, size_t... N>
Tensor<D, N...> from(Shape<N...>, std::initializer_list<dtype_t<D>> values) {
  if (values.size() != Shape<N...>::numel)
      throw std::invalid_argument("values size does not match shape");
  return Tensor<D, N...>(values.begin(), values.end());
}

template<dtype D>
Tensor<D> from(std::initializer_list<size_t> shape,
               std::initializer_list<dtype_t<D>> values) {
  std::vector<size_t> s(shape.begin(), shape.end());
  size_t n = std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>());
  if (values.size() != n)
      throw std::invalid_argument("values size does not match shape");
  return Tensor<D>(s, values.begin(), values.end());
}

template<dtype D, size_t... N, typename T>
Tensor<D, N...> arange(Shape<N...>, const Range<T> r) {
  using scalar_t = dtype_t<D>;
  scalar_t start = static_cast<scalar_t>(r.start);
  scalar_t step  = static_cast<scalar_t>(r.step);
  return Tensor<D, N...>([&](size_t i) {
    return start + step * static_cast<scalar_t>(i);
  });
}

template<dtype D, typename T>
Tensor<D> arange(const Range<T> r) {
  using scalar_t = dtype_t<D>;
  scalar_t start = static_cast<scalar_t>(r.start);
  scalar_t end   = static_cast<scalar_t>(r.end);
  scalar_t step  = static_cast<scalar_t>(r.step);
  std::vector<scalar_t> values;
  for (scalar_t val = start; val < end; val += step)
    values.push_back(val);
  return Tensor<D>({values.size()}, values.begin(), values.end());
}

template<dtype D, size_t... N>
Tensor<D, N...> add(const StaticTensor<D, N...>& a,const StaticTensor<D, N...>& b) {
  Tensor<D, N...> out;
  for (size_t i = 0; i < Shape<N...>::numel; ++i)
      out[i] = a[i] + b[i];
  return out;
}

template<dtype D>
Tensor<D> add(const DynamicTensor<D>& a, const DynamicTensor<D>& b) {
  if (a.shape != b.shape)
      throw std::invalid_argument("shape mismatch in add()");
  Tensor<D> out(a.shape);
  for (size_t i = 0; i < a.numel(); ++i)
      out[i] = a[i] + b[i];
  return out;
}

} // namespace tt

namespace tt::prelude {
  using enum tt::dtype;
  using tt::Shape;
  using tt::Range;
} // namespace tt::prelude

#endif // TT_HPP
