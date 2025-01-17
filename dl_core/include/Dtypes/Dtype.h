#ifndef DTYPE_H
#define DTYPE_H
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <map>

namespace cortex {

    using ui8_t = unsigned char;
    using ui16_t = unsigned short;
    using ui32_t = unsigned int;
    using ui64_t = unsigned long int;
    using i8_t = char;
    using i16_t = short int;
    using i32_t = int;
    using i64_t = long int;
    using f32_t = float;
    using f64_t = double;

    enum class dtype {
        f32,
        None,
    };

    template <dtype>
    struct dtype_trait;

    template <>
    struct dtype_trait<dtype::f32> {
        using type = f32_t;
    };

    template <dtype dtype>
    using dtype_trait_t = typename dtype_trait<dtype>::type;

    inline std::unordered_map<dtype, ui16_t> dtype_size_map = {
        {dtype::f32, 4}
    };

    inline ui16_t get_dtype_size(const dtype type) {
        return dtype_size_map[type];
    }

    template<typename T>
    dtype get_dtype() {
        if (std::is_same_v<T, f32_t>) {
            return dtype::f32;
        }
        throw std::invalid_argument("Unknown template data type!");
    }

    template <typename T>
    void CHECK_DATA_TYPE(const dtype type) {
        switch (type) {
            case dtype::f32:
                static_assert(std::is_same_v<T, f32_t>, "data type error!");
                break;
            default:
                throw std::invalid_argument("Unknown data type!");
        }
    }

}

#endif
