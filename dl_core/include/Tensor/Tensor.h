#ifndef TENSOR_H
#define TENSOR_H

#include <cmath>
#include <iomanip>
#include <memory>
#include <stack>
#include <vector>
#include "DeviceAllocator/DeviceAllocator.h"
#include "Dtypes/Dtype.h"
#include "Buffer/Buffer.h"
#include "Tensor_utils.h"

namespace cortex {

    class Autograd_graph;
    class BaseLayer;
    using BaseLayerPtr = std::shared_ptr<BaseLayer>;

    class Tensor : std::enable_shared_from_this<Tensor> {
    public:

        using pointer = std::shared_ptr<Tensor>;
        using weak_pointer = std::weak_ptr<Tensor>;

        /**
         * Constructor with specified device allocator
         * @param shape The shape of the tensor
         * @param dtype data type
         * @param alloc specified device allocator
         * @param requires_grad
         */
        Tensor(const std::vector<uint32_t>& shape, dtype dtype, const std::shared_ptr<DeviceAllocator>& alloc, bool requires_grad=true);

        /**
         * Constructor with specified device type.
         * @param shape The shape of the tensor.
         * @param dtype data type of each element in tensor
         * @param deviceType specified device type to allocate the tensor data.
         * @param requires_grad if the tensor tracks the gradient (false in default).
         */
        Tensor(const std::vector<uint32_t>& shape, dtype dtype, DeviceType deviceType, bool requires_grad=false);

        /**
         * Constructor of default device type
         * @param shape The shape of the tensor
         * @param dtype data type
         */
        Tensor(const std::vector<uint32_t>& shape, dtype dtype);

        /**
         * Constructing a tensor using the tensor buffer already given
         * @param shape the shape of the tensor
         * @param buffer shared tensor buffer
         * @param dtype the data type of element in the tensor
         * @param requires_grad whether tensor track gradient or not
         */
        Tensor(const std::vector<uint32_t>& shape, const std::shared_ptr<Buffer> &buffer, dtype dtype, bool requires_grad=true);

        /**
         * Shallow copy of current tensor
         * @param tensor shallow copied Tensor
         */
        Tensor(const Tensor& tensor);

        /**
         * Returns the size of the tensor
         * @return the size of the tensor
         */
        ui32_t size() const { return m_size; }

        /**
         * Return the shape vector of the tensor
         * @return the shape vector of the tensor
         */
        std::vector<uint32_t> shape() const { return m_shape; }

        /**
         * Return the stride vector of the tensor
         * @return the stride vector of the tensor
         */
        std::vector<uint32_t> stride() const { return m_stride; }

        /**
         * Returns the gradient of the tensor
         * @return the gradient of current tensor
         */
        std::shared_ptr<Tensor> grad() const { return m_grad; }

        /**
         * Checks if tensor requires gradients for backpropagation.
         * @return 'true' if the tensor requires gradients, otherwise 'false'
         */
        bool enable_grad() const { return m_requires_grad; }

        /**
         * Returns the data type of the tensor
         * @return the data type of the tensor
         */
        dtype get_dtype() const { return m_dtype; }

        /**
         * Returns the device type that the tensor is allocated on
         * @return the device type of the tensor source data
         */
        DeviceType get_device() const { return m_buffer->device_type(); }

        /**
         * Deep copy of the current tensor
         * @return the deep copied tensor
         */
        Tensor Copy() const;

        /**
         * @brief Returns a new tensor that is detached from the current computation graph.
         * This function creates a new tensor that shares the same underlying data as the original tensor,
         * but it does not track gradients.
         * @return A new tensor with the same data but without gradient tracking.
         */
        Tensor detach() const;

        /**
         * Release Memory
         */
        ~Tensor();

        /**
         * Returns a pointer to the raw data of the current tensor object
         * @tparam T The type of element in the tensor
         * @return the pointer to the raw data of the tensor
         */
        template <typename T>
        T* ptr() const;

        /**
         * Returns a pointer to the first element in the tensor
         * @tparam T The type of element in the tensor
         * @return a pointer to the first element in the tensor
         */
        template <typename T>
        T* begin() const;

        /**
         * Returns a pointer to the element past the last element of the tensor
         * @tparam T the type of the element in the tensor
         * @return a pointer to the element past the last element of the tensor
         */
        template <typename T>
        T* end() const;

        /**
         * Returns the gradient function of current tensor.
         * @return the reference of the gradient function, the layer that output current tensor.
         */
        std::shared_ptr<BaseLayer>& grad_func() {
            return m_layer;
        }

        /**
         * Returns the dimension of current tensor
         * @return the dimension of current tensor.
         */
        uint32_t dim() const {
            return m_shape.size();
        }

        /**
         * Get the element at the specified index of the tensor
         * @param index the index to be queried in the tensor
         * @return the reference to the element located at the index
         */
        f32_t& at(const std::vector<uint32_t>& index) const;

        /**
         * Initialize the tensor with specified values
         * @param initializer the values to be filled into the tensor
         */
        void initialize_with(const std::vector<f32_t>& initializer) const;

        /**
         * Convert all tensor data into the string
         * @return the converted string displaying contents of tensor
         */
        std::string to_string() const;

        /**
         * Fills the tensor with ones
         */
        void ones() const;

        /**
         * Fill the tensor with zeros
         */
        void zeros() const;

        /**
         * Returns a tensor object filled with ones with same shape
         * @return 
         */
        Tensor ones_like() const;

        /**
         * Fill the tensor with a value
         * @param data data to be filled into the tensor
         */
        void fill_(const f32_t& data) const;

        /**
         * Reshapes the tensor to the specified shape.
         * This function modifies the tensor's internal structure to match the given
         * shape. The total number of elements in the new shape must match the total
         * number of elements in the current tensor; otherwise, the behavior is undefined.
         * @param shape A vector representing the desired shape of the tensor
         */
        void reshape(const std::vector<uint32_t>& shape);

        /**
         * Check whether the other tensor has the same shape as self's.
         * If the other tensor doesn't have the same shape, then return a broadcast tensor.
         * If the other tensor has the same shape, then return itself.
         * @param other the other tensor to be compared with
         * @return
         */
        Tensor match_shape(const Tensor& other) const;

        /**
         * Broadcast the tensor to the target shape.
         * @param target_shape the target shape that current tensor to be broadcast to
         * @return the tensor broadcast of the target shape
         */
        Tensor broadcast_to(const std::vector<uint32_t>& target_shape) const;

        /**
         * Sum the tensor to the target shape
         * @param target_shape the target shape that current tensor to sum to
         * @return A new tensor with dimensions reduced to match the specified target shape.
         */
        Tensor sum_to(const std::vector<uint32_t>& target_shape) const;

        /**
         * Returns a transposed tensor of in-place modification
         * @param dim0 the first dimension to be transposed
         * @param dim1 the second dimension to be transposed
         * @return a transposed tensor
         */
        Tensor transpose(const uint32_t& dim0, const uint32_t &dim1);

        /**
         * Performs sum operation for every element in the tensor.
         * @return A tensor of shape {1} that contains the value of the sum of each element in the tensor.
         */
        Tensor sum() const;

        /**
         * Sets the gradient of current tensor to be zeros.
         * This function would usually be called after each iteration in training.
         */
        void zero_grad() const;

        /**
         * Creates the gradient of current tensor and set track tensor as true.
         */
        void requires_grad();

        /**
         * Performs the backward pass for backpropagation.
         * This function computes the gradients of all tensors that tracks gradients in the computational graph.
         */
        void backward();

        /**
         * Overloads the `+` operator to perform element-wise addition with another Tensor.
         * @note Based on the status in DLEngine, if the grad_mode is true, then gradient would be calculated.
         * if the grad_mode is false, then the system would start to execute inference.
         * if the grad_mode is false, then the system would start to execute inference.
         * @param tensor The other tensor to be added to the current tensor.
         * @return A result tensor object that equals to the current tensor + the other tensor
         */
        Tensor operator+(const Tensor& tensor) const;

        /**
         * Overloads the `+` operator to perform element-wise addition with scalar value.
         * @param scalar the scalar to be added to the current tensor.
         * @return A result tensor object that equals to the current tensor + scalar
         */
        Tensor operator+(const f32_t& scalar) const;

        /**
         * Overloads the '+' operator to perform element-wise addition with another scalar.
         * @param scalar the other scalar to add the current tensor.
         * @param tensor the tensor to be added by the scalar.
         * @return A shallow copy to the current tensor after the addition.
         */
        friend Tensor operator+(const f32_t& scalar, const Tensor& tensor);

        /**
         * Overloads the `+=` operator to perform element-wise addition with another Tensor.
         * @param tensor The other tensor to be added to the current tensor.
         * @return A shallow copy to the current tensor after the addition.
         */
        Tensor operator+=(const Tensor& tensor);

        /**
         * Overloads the `-` operator to perform element-wise subtraction between two Tensors.
         * @param tensor The other tensor to be subtracted to the current tensor.
         * @return A result tensor object that equals to the current tensor - the other tensor
         */
        Tensor operator-(const Tensor& tensor) const;

        /**
         * Overloads the `-` operator to perform element-wise subtraction with scalar value.
         * @param scalar the scalar to be subtracted to the current tensor.
         * @return A result tensor object that equals to the current tensor - scalar
         */
        Tensor operator-(const f32_t& scalar) const;

        /**
         * Overloads the '-' operator to perform element-wise subtraction with another scalar.
         * @param scalar the other scalar to subtract the current tensor.
         * @param tensor the tensor to be subtracted by the scalar.
         * @return A shallow copy to the current tensor after the subtraction.
         */
        friend Tensor operator-(const f32_t& scalar, const Tensor& tensor);

        /**
         * Overloads the `-=` operator to perform element-wise subtraction with another Tensor.
         * @param tensor The other tensor to be subtracted to the current tensor.
         * @return A shallow copy to the current tensor after the subtraction.
         */
        Tensor operator-=(const Tensor& tensor);

        /**
         * Overloads the `*` operator to perform element-wise multiplication between two Tensors.
         * @param tensor The other tensor to be multiplied to the current tensor.
         * @return A result tensor object that equals to the current tensor * the other tensor
         */
        Tensor operator*(const Tensor& tensor) const;

        /**
         * Overloads the `*` operator to perform element-wise multiplication with scalar value.
         * @param scalar the scalar to multiply the current tensor.
         * @return A result tensor object that equals to the current tensor * scalar
         */
        Tensor operator*(const f32_t &scalar) const;

        /**
         * Overloads the '*' operator to perform element-wise multiplication with another scalar
         * @param scalar the other scalar to be multiplied to the current tensor
         * @param tensor the tensor to be multiplied by the scalar
         * @return A shallow copy to the current tensor after the multiplication.
         */
        friend Tensor operator*(const f32_t& scalar, const Tensor& tensor);

        /**
         * Overloads the `*=` operator to perform element-wise multiplication with another Tensor.
         * @param tensor The other tensor to be multiplied to the current tensor.
         * @return A shallow copy to the current tensor after the multiplication.
         */
        Tensor operator*=(const Tensor& tensor);

        /**
         * Overloads the `/` operator to perform element-wise division between two Tensors.
         * @param tensor The other tensor to be divided to the current tensor.
         * @return A result tensor object that equals to the current tensor / the other tensor
         */
        Tensor operator/(const Tensor& tensor) const;

        /**
         * Overloads the `/` operator to perform element-wise division with scalar value.
         * @param scalar the scalar to be divided to the current tensor.
         * @return A result tensor object that equals to the current tensor / scalar
         */
        Tensor operator/(const f32_t &scalar) const;

        /**
         * Overloads the '/' operator to perform element-wise division with another scalar.
         * @param scalar the other scalar to divide the current tensor.
         * @param tensor the tensor to be divided by the scalar.
         * @return A shallow copy to the current tensor after the division.
         */
        friend Tensor operator/(const f32_t& scalar, const Tensor& tensor);

        /**
         * Overloads the `/=` operator to perform element-wise division with another Tensor.
         * @param tensor The other tensor to be divided to the current tensor.
         * @return A shallow copy of current tensor after the division.
         */
        Tensor operator/=(const Tensor& tensor);

        /**
         * Performs matrix multiplication with the given tensor.
         * @note Computes the matrix product of current tensor and the input tensor.
         * Note that the dimension of the tensors must be compatible for the matrix multiplication:
         * - For 2D matrix, the column number of current tensor must match the row number of the input tensor.
         * - For N-D tensor, the dimension other than row and column must support broadcast or the dimension must match together.
         * @param tensor the input tensor to multiply with.
         * @return A new tensor representing the result of the multiplication.
         */
        Tensor matmul(const Tensor& tensor) const;

        /**
         * Generate a tensor of given shape filled with one
         * @param shape the given shape of the generated tensor
         * @param dtype the data type of the generated tensor
         * @param device the device to allocate for the generated tensor
         * @return the generated tensor filled with one
         */
        static Tensor ones(const std::vector<uint32_t> &shape, dtype dtype = dtype::f32, DeviceType device = DeviceType::cpu);

        /**
         * Returns a 1-D tensor of size (end - start) / step with values from the interval [start, end)
         * taking the common difference step from the start.
         * @param start the starting value for the set of points (0 in default).
         * @param end the ending value for the set of points.
         * @param step the gap between each pair of adjacent points (1 in default).
         * @param device the device for the set of points (cpu in default).
         * @param requires_grad if tensor tracks gradient (false in default).
         * @return a 1-D tensor of size (end - start) / step with values from the interval [start, end)
         */
        static Tensor arange(const int& start, const int& end, const int& step = 1, DeviceType device = DeviceType::cpu, bool requires_grad = false);

    private:

        /**
         * Updates the shape of the tensor and changes the stride based on the shape
         * @param shape the dimension of the tensor
         */
        void update_shape(const std::vector<uint32_t>& shape);

    private:
        std::vector<uint32_t> m_shape;
        std::vector<uint32_t> m_stride;

        dtype m_dtype = dtype::None;
        std::shared_ptr<Buffer> m_buffer;
        uint32_t m_size = 0;

        bool m_requires_grad = false;
        BaseLayerPtr m_layer; // gradient function
        std::shared_ptr<Tensor> m_grad;

        std::shared_ptr<Autograd_graph> m_graph = nullptr;

        uint16_t m_printPrecision = 4;
    };

    /**
     * A utility function to create a tensor object
     * @param dim the dimension of the tensor
     * @param dtype the data type of element in the tensor
     * @param alloc the allocator of the data of the tensor
     * @param requires_grad (Optional) Indicates whether the tensor requires gradient tracking
     * @return std::shared_ptr<Tensor> A shared pointer to the newly created Tensor object.
     */
    inline Tensor::pointer tensor(const std::vector<uint32_t>& dim, dtype dtype, const std::shared_ptr<DeviceAllocator>& alloc, const bool requires_grad=true) {
        return std::make_shared<Tensor>(dim, dtype, alloc, requires_grad);
    }

    /**
     * A utility function to create a tensor object
     * @param dim the dimension of the tensor
     * @param dtype the data type of element in the tensor
     * @return std::shared_ptr<Tensor> A shared pointer to the newly created Tensor object.
     */
    inline Tensor::pointer tensor(const std::vector<uint32_t>& dim, dtype dtype) {
        return std::make_shared<Tensor>(dim, dtype);
    }

    template <typename T>
    T* Tensor::ptr() const {
        // Check whether the type is valid
        return static_cast<T*>(m_buffer->data());
    }

    template<typename T>
    T* Tensor::begin() const {
        return ptr<T>();
    }

    template<typename T>
    T* Tensor::end() const {
        return begin<T>() + size();
    }

    inline f32_t& Tensor::at(const std::vector<uint32_t> &index) const {
        if (index.size() != m_shape.size()) {
            throw std::runtime_error("Indexing Error (Tensor::at): Index vector size does not match with tensor's dimension");
        }

        // In-bounds check
        for (int i = 0; i < index.size(); i++) {
            if (index[i] >= m_shape[i])
                throw std::out_of_range("Index out of range! Index provided: " + vec_to_string(index) + ". Tensor shape given: " + vec_to_string(m_shape) + ".");
        }

        return ptr<f32_t>()[flatten_index_r_major(index, m_stride)];
    }

    inline std::string Tensor::to_string() const {
        std::ostringstream oss;
        auto* ptr = this->begin<f32_t>();

        // Set the width for each individual element
        // Find the max width for every element in the tensor
        const f32_t max_val = std::floor(*std::max_element(ptr, this->end<f32_t>()));
        ui32_t width = std::to_string(static_cast<i64_t>(max_val)).length() + 1;

        // Set the precision for the output
        oss << std::fixed << std::setprecision(this->m_printPrecision);
        width += m_printPrecision;

        std::stack<char> bracket_stack; // Stack to store the bracket needed in the output
        oss << "tensor(";

        // Add bracket to the output stream
        for (int i = 0; i < m_shape.size(); i++) {
            bracket_stack.push('[');
            oss << '[';
        } // After finish, the stream would be like [[[...

        /**
         * 1. Print Data
         * 2. When to move to next line
         * 3. When to add bracket
         */
        int cur_element_cnt = 0;
        while (cur_element_cnt < m_size) {
            const int col_num = m_shape[m_shape.size() - 1];

            // Take each row as basis, iterate through all row
            for (int j = 0; j < col_num; j++) {
                oss << std::setw(width) << std::right << *(ptr++);
                if (j < col_num - 1) {
                    oss << ", ";
                }
            }

            cur_element_cnt += col_num;
            // Add end bracket to the end of each row
            int end_bracket = 1;
            oss << std::string("]");
            bracket_stack.pop();

            // Add remaining bracket based on the stride
            for (int i = m_stride.size() - 3; i >= 0; i--) {
                if (cur_element_cnt % m_stride[i] == 0) {
                    oss << std::string("]");
                    bracket_stack.pop();
                    end_bracket++;
                }
            }

            // Wrap the lines
            if (cur_element_cnt + 1 < m_size) {
                oss << ",\n";
                if (end_bracket > 1) oss << "\n";
                // Compensate the spaces before each middle parenthesis.
                for (int r = 0; r < bracket_stack.size() + 7; r++) oss << " ";

                // Add middle parenthesis to the next line.
                for (int c = 0; c < end_bracket; c++) {
                    oss << "[";
                    bracket_stack.push('[');
                }
            }
        }

        while(!bracket_stack.empty()) {
            oss << "]";
            bracket_stack.pop();
        }

        oss << ")\n";
        return oss.str();
    }

    inline void Tensor::fill_(const f32_t& data) const {
        fill_with_value(m_buffer->data(), m_size, data, get_dtype_size(m_dtype));
    }

    inline Tensor operator+(const f32_t &scalar, const Tensor &tensor) {
        Tensor op_s({1}, tensor.m_dtype, tensor.m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return op_s.operator+(tensor);
    }

    inline Tensor operator-(const f32_t &scalar, const Tensor &tensor) {
        const Tensor op_s({1}, tensor.m_dtype, tensor.m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return op_s - tensor;
    }

    inline Tensor operator*(const f32_t &scalar, const Tensor &tensor) {
        const Tensor op_s({1}, tensor.m_dtype, tensor.m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return op_s * tensor;
    }

    inline Tensor operator/(const f32_t &scalar, const Tensor &tensor) {
        const Tensor op_s({1}, tensor.m_dtype, tensor.m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return op_s / tensor;
    }

    inline Tensor Tensor::arange(const int &start, const int &end, const int &step, const DeviceType device, const bool requires_grad) {
        // Initialize the data
        uint32_t size = (end - start) / step;
        Tensor ret({size}, dtype::f32, device, requires_grad);

        // Fill the data into a vector
        std::vector<f32_t> data;
        for (int i = start; i < end; i += step) {
            data.push_back(i);
        }
        ret.initialize_with(data);

        return ret;
    }

    inline void Tensor::initialize_with(const std::vector<f32_t>& initializer) const {
        fill_with_arr_r_major(ptr<f32_t>(), initializer);
    }
    
}


#endif //TENSOR_H
