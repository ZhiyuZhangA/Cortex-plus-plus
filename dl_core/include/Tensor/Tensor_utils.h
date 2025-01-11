//
// Created by zzy on 2024/12/27.
//

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H
#include <vector>
#include <algorithm>
#include <cstring>

namespace dl_core {
    /**
     * Copies elements from a vector into a given array 'ptr' one by one
     * using row-major order
     * @tparam T the type of element in the array
     * @param ptr a pointer to the target array
     * @param values a std::vector containing the values to be copied into the array
     */
    template<typename T>
    void fill_with_arr_r_major(T* ptr, const std::vector<T>& values) {
        std::copy(values.begin(), values.end(), ptr);
    }

    /**
     * Fill the array with a given value
     * @tparam T The type of the element in the array
     * @param ptr the pointer points to the array
     * @param size the size of the array
     * @param value the value to be filled in the array
     * @param byte_size the bytesize of the data to be filled into the tensor
     */
    template<typename T>
    void fill_with_value(void* ptr, const uint32_t& size, const T& value, const uint8_t& byte_size) {
        for (uint32_t i = 0; i < size; i++) {
            void* target = static_cast<char*>(ptr) + i * byte_size;
            std::memcpy(target, &value, byte_size);
        }
    }

    /**
     * Fill the array with zeros
     * @param ptr the pointer points to the array (fp32 in default)
     * @param size the size of the array
     * @param dtype
     */
    inline void fill_with_zero(void* ptr, const uint32_t& size, const dtype& dtype) {
        fill_with_value<f32_t>(ptr, size, 0, get_dtype_size(dtype));
    }

    /**
     * Fill the array with ones
     * @param ptr the pointer points to the array (fp32 in default)
     * @param size the size of the array
     * @param dtype the data type of the element in the tensor
     */
    inline void fill_with_ones(void* ptr, const uint32_t size, const dtype& dtype) {
        fill_with_value(ptr, size, 1, get_dtype_size(dtype));
    }

    /**
     * This function count the number of elements of the tensor using the dimension
     * @param shape the dimension of the tensor
     * @return the total number of elements in the tensor
     */
    inline uint32_t num_elements(const std::vector<uint32_t>& shape) {
        uint32_t num_elements = 1;
        for (const unsigned int i : shape)
            num_elements *= i;
        return num_elements;
    }

    /**
     * Get the stride of the tensor in row-major form.
     * @note The stride is the array of integers that describes the step size
     * between adjacent element along each dimension of a tensor.
     * @param shape the shape of the tensor
     * @return a vector containing the stride for each dimension of the tensor
     */
    inline std::vector<uint32_t> get_stride_r_major(const std::vector<uint32_t>& shape) {
        std::vector<uint32_t> stride;
        uint32_t stride_size = 1;
        stride.push_back(stride_size);
        for (int i = shape.size() - 1; i > 0; i--) {
            stride_size *= shape[i];
            stride.push_back(stride_size);
        }

        std::ranges::reverse(stride);

        return stride;
    }

    inline uint32_t flatten_index_r_major(const std::vector<uint32_t> &indices,
                                          const std::vector<uint32_t> &stride) {
        if (indices.size() != stride.size())
            throw std::runtime_error("indices and stride sizes must match!");

        uint32_t flatten_index = 0;

        for (int i = 0; i < indices.size(); i++)
            flatten_index += indices[i] * stride[i];

        return flatten_index;
    }

    inline void shuffle(const std::vector<uint32_t>& shape) {

    }

    /**
     * Convert the vector to the string
     * @param vec the vector
     * @return the string representing the data of vector
     */
    inline std::string vec_to_string(const std::vector<uint32_t>& vec) {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < vec.size(); i++) {
            ss << vec[i];
            if (i != vec.size() - 1)
                ss << ", ";
        }
        ss << "]";

        return ss.str();
    }

    /**
     * Returns the shape that the tensor would broadcast to
     * @param shape1
     * @param shape2
     * @return
     */
    inline std::vector<uint32_t> broadcast_shape(const std::vector<uint32_t>& shape1, const std::vector<uint32_t>& shape2) {
        std::vector<uint32_t> result;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();

        while (it1 != shape1.rend() || it2 != shape2.rend()) {
            int dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            int dim2 = (it2 != shape2.rend()) ? *it2++ : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                throw std::invalid_argument("Incompatible shape for broadcasting!");

            result.push_back(std::max(dim1, dim2));
        }

        std::ranges::reverse(result);
        return result;
    }

    /**
     *
     * @param shape1
     * @param shape2
     * @return
     */
    inline std::vector<uint32_t> extend_shape(const std::vector<uint32_t>& shape1, const std::vector<uint32_t>& shape2) {
        std::vector<uint32_t> result;
        size_t dif;
        if (shape1.size() > shape2.size()) {
            result = shape2;
            dif = shape1.size() - shape2.size();
        }
        else {
            result = shape1;
            dif = shape2.size() - shape1.size();
        }

        for (int i = 0; i < dif; i++)
            result.insert(result.begin(), 1);

        return result;
    }

    /**
     * Copies elements from a vector into a given array 'ptr' one by one using column-major order
     * @tparam T the type of element in the array
     * @param ptr a pointer to the target array
     * @param values a std::vector containing the values to be copied into the array
     * @param shape the dimension of the tensor
     */
    template<typename T>
    [[deprecated("Use fill_arr_r_major() instead.")]]
    void fill_arr_c_major(T* ptr, const std::vector<T>& values, const std::vector<ui32_t>& shape) {
        if (shape.size() < 2) {
            for (int i = 0; i < values.size(); i++)
                ptr[i] = values[i];

            return;
        }

        int idx = 0;
        const int row_num = shape[shape.size() - 2];
        const int col_num = shape[shape.size() - 1];
        while (idx < values.size()) {
            for (int i = 0; i < col_num; i++) {
                for (int j = 0; j < row_num; j++) {
                    *(ptr++) = values[idx + j * col_num + i];
                }
            }
            idx += row_num * col_num;
        }
    }

    /**
     * Get the stride for the tensor object in column-major form
     * @note The stride is the array of integers that describes the step size
     * between adjacent element along each dimension of a tensor
     * @param shape the shape of the tensor object
     * @return a vector containing the stride for each dimension of the tensor
     */
    [[deprecated("Use get_stride_r_major() instead.")]]
    inline std::vector<uint32_t> get_stride_c_major(const std::vector<uint32_t>& shape) {
        std::vector<uint32_t> stride;
        uint32_t stride_size = 1;
        stride.push_back(stride_size);
        if (shape.size() >= 2) {
            stride_size *= shape[shape.size() - 2];
            stride.push_back(stride_size);
            stride_size *= shape[shape.size() - 1];
            for (int i = static_cast<int>(shape.size()) - 3; i >= 0; i--) {
                stride.push_back(stride_size);
                stride_size *= shape[i];
            }

            std::ranges::reverse(stride);
        }

        return stride;
    }

    /**
     * Flatten the indices of the tensor to the index for a one-dimension array in column-major order
     * @param indices the indices to locate in the tensor
     * @param stride the stride of the tensor given its dimension
     * @return the flattened index of the indices provided for a one-dimension array in column-major order
     */
    [[deprecated("Use flatten_index_r_major() instead.")]]
    inline uint32_t flatten_index_c_major(const std::vector<uint32_t>& indices, const std::vector<uint32_t>& stride) {
        // exceptional case for indices of size 1
        if (indices.size() == 1)
            return indices[0];

        uint32_t findex = 0;
        for (int i = 0; i < indices.size() - 2; i++) {
            findex += indices[i] * stride[i];
        }

        // Address the remaining row-column dimension
        // Column index × row number + row index -> column major order index
        const auto col_index = indices[indices.size() - 1];
        const auto row_num = stride[indices.size() - 2];
        const auto row_index = indices[indices.size() - 2];
        findex += col_index * row_num + row_index;

        return findex;
    }
}


#endif //TENSOR_UTILS_H
