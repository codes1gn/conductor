
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

// Include choreo.h for host compilation
// This provides the choreo::spanned_view and choreo::spanned_data types
#include "choreo.h"

// Forward declaration of the auto-generated choreo host function
// This matches the signature from choreo -gs output and is already compiled in the object file
extern choreo::spanned_data<choreo::f32, 2> kernel_6788ff95(const choreo::spanned_view<choreo::f32, 2>& input_0, const choreo::spanned_view<choreo::f32, 2>& input_1);


// Host wrapper function that can be called from Python via ctypes
extern "C" {
    int execute_kernel(void* input_data_0, size_t* input_shape_0, void* input_data_1, size_t* input_shape_1, void* output_data, size_t* output_shape) {
        try {

            // Create spanned_view for input 0
            choreo::f32* input_0_ptr = static_cast<choreo::f32*>(input_data_0);
            choreo::mdspan<2> input_0_shape{input_shape_0[0], input_shape_0[1]};
            choreo::spanned_view<choreo::f32, 2> input_0_view(input_0_ptr, input_0_shape);
            // Create spanned_view for input 1
            choreo::f32* input_1_ptr = static_cast<choreo::f32*>(input_data_1);
            choreo::mdspan<2> input_1_shape{input_shape_1[0], input_shape_1[1]};
            choreo::spanned_view<choreo::f32, 2> input_1_view(input_1_ptr, input_1_shape);

            // Call the auto-generated choreo host function
            auto result = kernel_6788ff95(input_0_view, input_1_view);

            // Copy result data to output buffer
            choreo::f32* output_ptr = static_cast<choreo::f32*>(output_data);

            // Calculate total number of elements from result shape
            size_t total_elements = result.element_count();

            std::memcpy(output_ptr, result.data(), total_elements * sizeof(choreo::f32));

            // Set output shape
            output_shape[0] = result.shape()[0];
            output_shape[1] = result.shape()[1];

            return 0;  // Success

        } catch (const std::exception& e) {
            std::cerr << "Kernel execution error: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cerr << "Unknown kernel execution error" << std::endl;
            return -2;
        }
    }
}
