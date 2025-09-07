"""
GCU Host Interface for Real Device Execution.

This module implements the host-side integration for executing compiled choreo
kernels on GCU hardware, following the patterns from choreo-op examples.
It handles tensor marshalling between PyTorch and choreo formats using
the authentic choreo host-side data structures.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Optional, Union
import torch
import numpy as np
from dataclasses import dataclass

from ..utils.exceptions import DeviceError, ExecutionError
from ..config.logging import get_logger
from ..config.debug_artifacts import get_debug_manager
from ..config.debug_tracer import get_debug_tracer, trace_host_wrapper

logger = get_logger(__name__)


@dataclass
class ChoreoTensorDescriptor:
    """Describes a tensor in choreo format for host-device communication."""
    data_ptr: int           # Pointer to tensor data
    shape: list[int]        # Tensor dimensions
    dtype: str              # Choreo data type (f32, s32, etc.)
    element_size: int       # Size of each element in bytes
    total_bytes: int        # Total memory size
    rank: int               # Number of dimensions


class ChoreoHostInterface:
    """
    Host-side interface for choreo kernel execution.
    
    This class handles the conversion between PyTorch tensors and choreo
    data structures, following the patterns from choreo-op examples.
    """
    
    def __init__(self):
        """Initialize the choreo host interface."""
        self.dtype_mapping = {
            torch.float32: ('f32', 4),
            torch.float16: ('f16', 2),
            torch.int32: ('s32', 4),
            torch.int64: ('s64', 8),
            torch.bool: ('bool', 1),
        }
        
    def tensor_to_choreo_descriptor(self, tensor: torch.Tensor) -> ChoreoTensorDescriptor:
        """
        Convert PyTorch tensor to choreo tensor descriptor.
        
        Args:
            tensor: PyTorch tensor to convert
            
        Returns:
            ChoreoTensorDescriptor with choreo-compatible information
        """
        if tensor.dtype not in self.dtype_mapping:
            raise DeviceError(f"Unsupported tensor dtype: {tensor.dtype}")
        
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        choreo_dtype, element_size = self.dtype_mapping[tensor.dtype]
        
        return ChoreoTensorDescriptor(
            data_ptr=tensor.data_ptr(),
            shape=list(tensor.shape),
            dtype=choreo_dtype,
            element_size=element_size,
            total_bytes=tensor.numel() * element_size,
            rank=len(tensor.shape)
        )
    
    def generate_host_wrapper_code(self, function_name: str, input_descriptors: list[ChoreoTensorDescriptor],
                                   output_descriptor: ChoreoTensorDescriptor) -> str:
        """
        Generate C++ host wrapper code that interfaces with choreo __co__ function.
        
        This follows the pattern from choreo-op examples where host code creates
        spanviews and calls the __co__ function.
        
        Args:
            function_name: Name of the choreo __co__ function
            input_descriptors: List of input tensor descriptors
            output_descriptor: Output tensor descriptor
            
        Returns:
            C++ code string for the host wrapper
        """
        # Generate includes and function declaration for host code
        # Based on choreo -gs analysis, we include choreo.h for host compilation only
        wrapper_code = f'''
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

// Include choreo.h for host compilation
// This provides the choreo::spanned_view and choreo::spanned_data types
#include "choreo.h"

// Forward declaration of the auto-generated choreo host function
// This matches the signature from choreo -gs output and is already compiled in the object file
extern choreo::spanned_data<choreo::{output_descriptor.dtype}, {output_descriptor.rank}> {function_name}('''

        # Generate function signature parameters for the choreo function
        param_strs = []
        for i, desc in enumerate(input_descriptors):
            param_strs.append(f"const choreo::spanned_view<choreo::{desc.dtype}, {desc.rank}>& input_{i}")

        wrapper_code += ", ".join(param_strs) + ");\n\n"
        
        # Generate host wrapper function that calls the auto-generated choreo function
        wrapper_code += f'''
// Host wrapper function that can be called from Python via ctypes
extern "C" {{
    int execute_kernel('''

        # Generate wrapper function parameters
        wrapper_params = []
        for i, desc in enumerate(input_descriptors):
            wrapper_params.append(f"void* input_data_{i}")
            wrapper_params.append(f"size_t* input_shape_{i}")
        wrapper_params.append("void* output_data")
        wrapper_params.append("size_t* output_shape")

        wrapper_code += ", ".join(wrapper_params) + ") {\n"
        wrapper_code += "        try {\n"

        # Create choreo::spanned_view objects from the input data
        for i, desc in enumerate(input_descriptors):
            wrapper_code += f'''
            // Create spanned_view for input {i}
            choreo::{desc.dtype}* input_{i}_ptr = static_cast<choreo::{desc.dtype}*>(input_data_{i});'''

            # Generate shape using mdspan (SimpleArray) with proper initialization
            shape_elements = [f"input_shape_{i}[{j}]" for j in range(desc.rank)]
            wrapper_code += f'''
            choreo::mdspan<{desc.rank}> input_{i}_shape{{{", ".join(shape_elements)}}};
            choreo::spanned_view<choreo::{desc.dtype}, {desc.rank}> input_{i}_view(input_{i}_ptr, input_{i}_shape);'''

        wrapper_code += f'''

            // Call the auto-generated choreo host function
            auto result = {function_name}('''

        # Generate function call arguments
        call_args = [f"input_{i}_view" for i in range(len(input_descriptors))]
        wrapper_code += ", ".join(call_args) + ");\n"

        wrapper_code += f'''
            // Copy result data to output buffer
            choreo::{output_descriptor.dtype}* output_ptr = static_cast<choreo::{output_descriptor.dtype}*>(output_data);

            // Calculate total number of elements from result shape
            size_t total_elements = result.element_count();

            std::memcpy(output_ptr, result.data(), total_elements * sizeof(choreo::{output_descriptor.dtype}));

            // Set output shape'''

        for j in range(output_descriptor.rank):
            wrapper_code += f'''
            output_shape[{j}] = result.shape()[{j}];'''

        wrapper_code += '''

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
'''

        # Debug tracing: Print host wrapper details
        debug_tracer = get_debug_tracer()
        if debug_tracer.is_enabled():
            # Create function signature for display
            param_strs = []
            for i, desc in enumerate(input_descriptors):
                param_strs.append(f"const choreo::spanned_view<choreo::{desc.dtype}, {desc.rank}>& input_{i}")
            function_signature = f"choreo::spanned_data<choreo::{output_descriptor.dtype}, {output_descriptor.rank}> {function_name}({', '.join(param_strs)})"

            # Create buffer info
            buffer_info = {}
            for i, desc in enumerate(input_descriptors):
                buffer_info[f"input_{i}"] = {
                    'size': desc.total_bytes,
                    'alignment': 'default',
                    'memory_type': 'device',
                    'dtype': desc.dtype,
                    'shape': desc.shape
                }
            buffer_info['output'] = {
                'size': output_descriptor.total_bytes,
                'alignment': 'default',
                'memory_type': 'device',
                'dtype': output_descriptor.dtype,
                'shape': output_descriptor.shape
            }

            trace_host_wrapper(wrapper_code, function_signature, buffer_info)

        return wrapper_code
    
    def compile_host_wrapper(self, wrapper_code: str, object_file_path: str, output_path: str, kernel_name: str = "unknown") -> None:
        """
        Compile the host wrapper C++ code with the choreo object file.

        Args:
            wrapper_code: C++ wrapper code to compile
            object_file_path: Path to the choreo object file
            output_path: Path for the output shared library
            kernel_name: Name of the kernel for debug artifacts
        """
        # Get debug manager and save host wrapper code
        debug_manager = get_debug_manager()
        debug_wrapper_path = debug_manager.save_host_wrapper(kernel_name, wrapper_code)

        # Use debug path as the wrapper source file (instead of temporary file)
        wrapper_cpp_path = str(debug_wrapper_path)

        logger.info(f"Host wrapper C++ code saved to: {wrapper_cpp_path}")

        # Also save the choreo object file to debug directory
        debug_manager.save_intermediate_object(kernel_name, object_file_path)
        
        try:
            # Based on choreo -gs analysis, compile host wrapper using regular C++ compiler
            # The choreo object file contains the host function, we just need to link against it
            from .topscc_utils import get_topscc_environment

            topscc_env = get_topscc_environment()
            include_dirs = [f'{os.path.dirname(os.path.abspath(__file__))}/../../choreo-headers']

            # Use g++ for host wrapper compilation (not topscc device compilation)
            # This matches the approach shown in choreo -gs output
            compile_cmd = [
                'g++',
                '-shared',
                '-fPIC',
                '-std=c++17',
                f'-I{include_dirs[0]}',
                wrapper_cpp_path,
                object_file_path,
                '-o', output_path
            ]

            # Add topscc library paths if available for GCU runtime linking
            if topscc_env.is_available():
                compile_cmd.extend([
                    f'-L{topscc_env.topscc_lib}',
                    '-ltopsrt',
                    '-lrtcu',
                    '-lefrt',
                    f'-Wl,-rpath,{topscc_env.topscc_lib}'
                ])

            logger.debug(f"Compiling host wrapper: {' '.join(compile_cmd)}")
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise DeviceError(f"Host wrapper compilation failed: {result.stderr}")

            # Save the final compiled shared library to debug directory
            debug_manager.save_shared_library(kernel_name, output_path)
            logger.info(f"Compiled shared library saved to debug directory")

        except subprocess.TimeoutExpired:
            raise DeviceError("Host wrapper compilation timed out")
        except FileNotFoundError:
            raise DeviceError("Compiler not found. Please ensure topscc or g++ is installed and in PATH.")

        # Note: We no longer delete the wrapper file - it's saved in debug directory for inspection
    
    def create_output_tensor(self, descriptor: ChoreoTensorDescriptor, device: str = 'cpu') -> torch.Tensor:
        """
        Create a PyTorch tensor for output based on choreo descriptor.
        
        Args:
            descriptor: Choreo tensor descriptor
            device: PyTorch device ('cpu' or 'cuda')
            
        Returns:
            Empty PyTorch tensor with correct shape and dtype
        """
        # Map choreo dtype back to PyTorch dtype
        dtype_reverse_mapping = {
            'f32': torch.float32,
            'f16': torch.float16,
            's32': torch.int32,
            's64': torch.int64,
            'bool': torch.bool,
        }
        
        torch_dtype = dtype_reverse_mapping.get(descriptor.dtype, torch.float32)
        return torch.empty(descriptor.shape, dtype=torch_dtype, device=device)
    
    def load_shared_library(self, library_path: str) -> ctypes.CDLL:
        """
        Load the compiled shared library for execution.
        
        Args:
            library_path: Path to the shared library
            
        Returns:
            Loaded ctypes library
        """
        try:
            lib = ctypes.CDLL(library_path)
            
            # Set up the execute_kernel function signature
            lib.execute_kernel.restype = ctypes.c_int
            # The exact argument types will be set dynamically based on tensor descriptors
            
            return lib
            
        except OSError as e:
            raise DeviceError(f"Failed to load shared library {library_path}: {e}")
    
    def execute_kernel_with_library(self, lib: ctypes.CDLL, input_tensors: list[torch.Tensor],
                                    output_tensor: torch.Tensor) -> int:
        """
        Execute the kernel using the loaded shared library.
        
        Args:
            lib: Loaded ctypes library
            input_tensors: List of input PyTorch tensors
            output_tensor: Output PyTorch tensor
            
        Returns:
            Execution result code (0 for success)
        """
        # Prepare arguments for the C function
        args = []
        
        # Add input tensor data and shapes
        for tensor in input_tensors:
            # Ensure tensor is contiguous
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Add data pointer
            args.append(ctypes.cast(tensor.data_ptr(), ctypes.c_void_p))
            
            # Add shape array
            shape_array = (ctypes.c_size_t * len(tensor.shape))(*tensor.shape)
            args.append(shape_array)
        
        # Add output tensor data and shape
        if not output_tensor.is_contiguous():
            output_tensor = output_tensor.contiguous()
        
        args.append(ctypes.cast(output_tensor.data_ptr(), ctypes.c_void_p))
        output_shape_array = (ctypes.c_size_t * len(output_tensor.shape))(*output_tensor.shape)
        args.append(output_shape_array)
        
        # Execute the kernel
        try:
            result = lib.execute_kernel(*args)
            return result
        except Exception as e:
            raise ExecutionError(f"Kernel execution failed: {e}")
