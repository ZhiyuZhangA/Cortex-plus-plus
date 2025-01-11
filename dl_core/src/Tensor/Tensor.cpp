#include <iostream>
#include <queue>

#include "Tensor/Tensor.h"
#include "DLEngine/DLEngine.h"
#include "Layers/AddLayer.h"
#include "Layers/BroadcastLayer.h"
#include "Layers/DivLayer.h"
#include "Layers/MulLayer.h"
#include "Layers/SubLayer.h"
#include "Layers/SumToLayer.h"
#include "Layers/TransposeLayer.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Graphs/autograd_graph.h"

namespace cortex_core {
    Tensor::Tensor(const std::vector<uint32_t> &shape, const dtype dtype, const std::shared_ptr<DeviceAllocator> &alloc,
                   const bool requires_grad) :
    m_shape(shape),
    m_dtype(dtype),
    m_requires_grad(requires_grad) {

        // Count the number of elements
        this->m_size = num_elements(shape);

        // Initialize stride
        this->m_stride = get_stride_r_major(shape);

        // Allocate buffer
        m_buffer = std::make_shared<Buffer>(get_dtype_size(dtype) * m_size, alloc);
        m_buffer->allocate();

        // Default fill the tensor with zero
        fill_with_zero(m_buffer->data(), this->m_size, m_dtype);

        // Set the gradient
        if (!m_requires_grad) {
            m_grad = nullptr;
        }
        else {
            // Dynamically allocate space for the gradient
            m_grad = std::make_shared<Tensor>(shape, dtype::f32, alloc, false);
        }
        /** The reason using a pointer to represent gradient is that if the tensor doesn't requires
           * gradient, then having a tensor object as gradient is simply unnecessary.
           */
    }

    Tensor::Tensor(const std::vector<uint32_t> &shape, const dtype dtype, DeviceType deviceType, bool requires_grad) : Tensor(shape, dtype, DeviceAllocatorFactory::create_cpu_allocator(), requires_grad) { }

    Tensor::Tensor(const std::vector<uint32_t> &shape, const dtype dtype) : Tensor(shape, dtype, DeviceAllocatorFactory::create_cpu_allocator()) { }

    Tensor::Tensor(const std::vector<uint32_t> &shape, const std::shared_ptr<Buffer> &buffer, dtype dtype, bool requires_grad) {
        this->m_buffer = buffer;
        this->m_dtype = dtype;
        this->m_shape = shape;
        this->m_stride = get_stride_r_major(shape);
        this->m_size = num_elements(shape);
        this->m_layer = nullptr;
        this->m_requires_grad = requires_grad;

        // Set a new gradient
        if (!m_requires_grad) {
            m_grad = nullptr;
        }
        else {
            // Dynamically allocate space for the gradient
            m_grad = std::make_shared<Tensor>(shape, dtype::f32, buffer->device_alloc(), false);
        }
    }

    Tensor::Tensor(const Tensor &tensor)  : enable_shared_from_this(tensor) {
        // Shallow Copy
        this->m_buffer = tensor.m_buffer;
        this->m_shape = tensor.m_shape;
        this->m_stride = tensor.m_stride;
        this->m_dtype = tensor.m_dtype;
        this->m_size = tensor.m_size;
        this->m_requires_grad = tensor.m_requires_grad;
        this->m_layer = tensor.m_layer;
        this->m_grad = tensor.m_grad;
    }

    Tensor::~Tensor() = default;

    void Tensor::ones() const {
        fill_with_ones(m_buffer->data(), this->m_size, m_dtype);
    }

    void Tensor::zeros() const {
        fill_with_zero(m_buffer->data(), this->m_size, m_dtype);
    }

    Tensor Tensor::ones_like() const {
        Tensor copy(this->m_shape, this->m_dtype, this->get_device(), this->m_requires_grad);
        copy.ones();
        return copy;
    }

    void Tensor::update_shape(const std::vector<uint32_t> &shape) {
        // Update the dimension and the stride of the tensor
        this->m_shape = shape;
        this->m_stride = get_stride_r_major(this->m_shape);
    }

    Tensor Tensor::Copy() const {
        Tensor copied(m_shape, m_dtype, m_buffer->device_type());
        copied.m_buffer = m_buffer;
        return copied;
    }

    Tensor Tensor::detach() const {
        // Stop tracking the gradient in default
        Tensor copied(m_shape, m_buffer, m_dtype, false);

        return copied;
    }

    Tensor Tensor::ones(const std::vector<uint32_t> &shape, const dtype dtype, const DeviceType device) {
        Tensor tensor(shape, dtype, device, false);
        tensor.fill_(1.0);
        return tensor;
    }

    void Tensor::reshape(const std::vector<uint32_t> &shape) {
        // Check whether the shape is legal
        if (num_elements(shape) != this->m_size) {
            throw std::invalid_argument("Tensor::reshape(): invalid shape");
        }

        this->m_shape = shape;
        this->m_stride = get_stride_r_major(this->m_shape);
    }

    Tensor Tensor::match_shape(const Tensor &other) const {
        if (this->m_shape == other.m_shape) {
            return other;
        }
        else {
            // Check whether the other support broadcast operation

            // Create a broadcast tensor object and create a broadcast layer.
            return other;
        }
    }

    Tensor Tensor::broadcast_to(const std::vector<uint32_t> &target_shape) const {
        // Check whether current tensor support broadcast
        if (target_shape == this->m_shape) {
            throw std::invalid_argument("Tensor::broadcast_to: No Need to broadcast! Target shape is the same as current shape!");
        }

        Tensor ret(target_shape, this->m_dtype, this->get_device(), this->m_requires_grad);

        const int tar_size = num_elements(target_shape);
        std::vector<f32_t> filled_buffer(tar_size);
        for (int i = 0; i < tar_size; i++) {
            int original_idx = 0;
            int idx = i;
            for (int j = target_shape.size() - 1; j >= 0; j--) {
                int _dim = (j < m_shape.size()) ? m_shape[j] : 1;
                int _stride = (j < m_shape.size()) ? m_stride[j] : 1;
                int pos = idx % target_shape[j];
                if (_dim != 1) {
                    original_idx += pos * _stride;
                }
                idx /= target_shape[j];
            }
            filled_buffer[i] = this->ptr<f32_t>()[original_idx];
        }

        ret.initialize_with(filled_buffer);

        // Link the tensor to the layer if tracks gradient
        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<BroadcastLayer>(ret.m_dtype, ret.get_device(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::sum_to(const std::vector<uint32_t> &target_shape) const {
        // Extend the shape if the dimension doesn't match
        auto res_shape = target_shape;
        if (this->m_shape.size() != target_shape.size()) {
            res_shape = extend_shape(target_shape, this->m_shape);
        }

        // Defining the result Tensor with extending shape
        Tensor ret(target_shape, this->m_dtype, this->get_device(), this->m_requires_grad);

        std::vector<size_t> indices(this->m_shape.size(), 0);
        std::vector<f32_t> filled_data(size());

        for (size_t i = 0; i < size(); i++) {
            // Get the corresponding indices in the original tensor
            size_t remaining = i;
            for (int j = this->m_shape.size() - 1; j >= 0; j--) {
                indices[j] = remaining % m_shape[j];
                remaining /= m_shape[j];
            }

            std::vector<uint32_t> target_indices(res_shape.size());
            for (size_t j = 0; j < res_shape.size(); j++) {
                target_indices[j] = (res_shape[j] == m_shape[j]) ? indices[j] : 0;
            }

            ret.at(target_indices) += *(ptr<f32_t>() + i);
        }

        ret.reshape(target_shape);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<SumToLayer>(ret.m_dtype, ret.get_device(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::transpose(const uint32_t& dim0, const uint32_t& dim1) {
        // In-place modification
        // Get the transposed shape
        std::vector<uint32_t> transpose_dims(this->m_shape);
        std::swap(transpose_dims[dim0], transpose_dims[dim1]);

        // Create a new tensor with same data source
        Tensor ret(transpose_dims, this->m_buffer, m_dtype, this->m_requires_grad);
        // Transpose the matrix
        get_transpose_kernel(this->m_buffer->device_type())(*this, ret, dim0, dim1, false);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<TransposeLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::sum() const {
        Tensor ret({1}, this->m_dtype, this->get_device(), this->m_requires_grad);
        get_sum_kernel(this->get_device())(*this, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<SumToLayer>(this->m_dtype, this->get_device(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_output(ret);
        }
        return ret;
    }

    void Tensor::zero_grad() const {
        if (this->enable_grad())
            this->m_grad->zeros();
    }

    void Tensor::requires_grad() {
        if (m_grad == nullptr) {
            m_grad = std::make_shared<Tensor>(m_shape, dtype::f32, m_buffer->device_alloc(), false);
            m_requires_grad = true;
        }
    }

    void Tensor::backward() {
        if (!DLEngine::is_grad_mode()) {
            std::clog << "Tensor::backward does not support inference mode" << std::endl;
            return;
        }

        if (m_graph == nullptr) {
            m_graph = std::make_shared<Autograd_graph>(*this);
        }
        m_graph->backward(*this);
    }

    Tensor Tensor::operator+(const Tensor& tensor) const {
        // Broadcast First
        const std::vector<uint32_t> target_shape = broadcast_shape(this->m_shape, tensor.m_shape);
        const Tensor this_broadcast = (this->m_shape != target_shape && this->m_size != 1) ? this->broadcast_to(target_shape) : *this;
        const Tensor other_broadcast = (tensor.m_shape != target_shape && tensor.m_size != 1) ? tensor.broadcast_to(target_shape) : tensor;
        Tensor ret(target_shape, m_dtype, m_buffer->device_alloc(), this->m_requires_grad || other_broadcast.m_requires_grad);
        get_add_kernel(m_buffer->device_type())(this_broadcast, other_broadcast, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<AddLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(this_broadcast);
            ret.m_layer->add_input(other_broadcast);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator+(const f32_t &scalar) const {
        Tensor op_s({1}, m_dtype, m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return *this + op_s;
    }

    Tensor Tensor::operator+=(const Tensor &tensor) {
        const Tensor other = match_shape(tensor);
        Tensor ret(this->m_shape, this->m_buffer, m_dtype, m_requires_grad);
        get_add_kernel(m_buffer->device_type())(*this, other, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<SubLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_input(other);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator-(const Tensor &tensor) const {
        // Broadcast First
        const std::vector<uint32_t> target_shape = broadcast_shape(this->m_shape, tensor.m_shape);
        const Tensor this_broadcast = (this->m_shape != target_shape && this->m_size != 1) ? this->broadcast_to(target_shape) : *this;
        const Tensor other_broadcast = (tensor.m_shape != target_shape && tensor.m_size != 1) ? tensor.broadcast_to(target_shape) : tensor;
        Tensor ret(target_shape, m_dtype, m_buffer->device_alloc(), this->m_requires_grad || other_broadcast.m_requires_grad);
        get_sub_kernel(m_buffer->device_type())(this_broadcast, other_broadcast, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<SubLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(this_broadcast);
            ret.m_layer->add_input(other_broadcast);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator-(const f32_t &scalar) const {
        Tensor op_s({1}, m_dtype, m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return *this - op_s;
    }

    Tensor Tensor::operator-=(const Tensor &tensor) {
        const Tensor other = match_shape(tensor);
        Tensor ret(this->m_shape, this->m_buffer, m_dtype, m_requires_grad);
        get_sub_kernel(m_buffer->device_type())(*this, other, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<SubLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_input(other);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator*(const Tensor &tensor) const {
        // Broadcast First
        const std::vector<uint32_t> target_shape = broadcast_shape(this->m_shape, tensor.m_shape);
        const Tensor this_broadcast = (this->m_shape != target_shape && this->m_size != 1) ? this->broadcast_to(target_shape) : *this;
        const Tensor other_broadcast = (tensor.m_shape != target_shape && tensor.m_size != 1) ? tensor.broadcast_to(target_shape) : tensor;
        Tensor ret(target_shape, m_dtype, m_buffer->device_alloc(), this->m_requires_grad || other_broadcast.m_requires_grad);
        get_mul_kernel(m_buffer->device_type())(this_broadcast, other_broadcast, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<MulLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(this_broadcast);
            ret.m_layer->add_input(other_broadcast);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator*(const f32_t &scalar) const {
        Tensor op_s({1}, m_dtype, m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return *this * op_s;
    }

    Tensor Tensor::operator*=(const Tensor &tensor) {
        const Tensor other = match_shape(tensor);
        Tensor ret(this->m_shape, this->m_buffer, m_dtype, m_requires_grad);
        get_mul_kernel(m_buffer->device_type())(*this, other, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<MulLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_input(other);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator/(const Tensor &tensor) const {
        // Broadcast First
        const std::vector<uint32_t> target_shape = broadcast_shape(this->m_shape, tensor.m_shape);
        const Tensor this_broadcast = (this->m_shape != target_shape && this->m_size != 1) ? this->broadcast_to(target_shape) : *this;
        const Tensor other_broadcast = (tensor.m_shape != target_shape && tensor.m_size != 1) ? tensor.broadcast_to(target_shape) : tensor;
        Tensor ret(target_shape, m_dtype, m_buffer->device_alloc(), this->m_requires_grad || other_broadcast.m_requires_grad);
        get_div_kernel(m_buffer->device_type())(this_broadcast, other_broadcast, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<DivLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(this_broadcast);
            ret.m_layer->add_input(other_broadcast);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }

    Tensor Tensor::operator/(const f32_t &scalar) const {
        Tensor op_s({1}, m_dtype, m_buffer->device_type(), false);
        op_s.fill_(scalar);
        return this->operator/(op_s);
    }

    Tensor Tensor::operator/=(const Tensor &tensor) {
        const Tensor other = match_shape(tensor);
        Tensor ret(this->m_shape, this->m_buffer, m_dtype, m_requires_grad);
        get_div_kernel(m_buffer->device_type())(*this, other, ret);

        if (DLEngine::is_grad_mode()) {
            ret.m_layer = std::make_shared<DivLayer>(this->m_dtype, this->m_buffer->device_type(), false);
            ret.m_layer->add_input(*this);
            ret.m_layer->add_input(other);
            ret.m_layer->add_output(ret);
        }

        return ret;
    }
}
