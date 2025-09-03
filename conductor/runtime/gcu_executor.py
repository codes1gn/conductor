"""
GCU Device Execution Engine.

This module implements the host-side integration for executing compiled choreo
kernels on GCU hardware. It handles tensor marshalling between PyTorch and
choreo formats, device memory management, and kernel execution coordination.
"""

import ctypes
import os
import tempfile
import subprocess
from typing import List, Any, Dict, Optional, Tuple
import torch
import numpy as np

from .loader import CompiledArtifact, ExecutableKernel
from .gcu_host_interface import ChoreoHostInterface, ChoreoTensorDescriptor
from ..utils.exceptions import DeviceError, ExecutionError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GCUKernelExecutor:
    """
    Executes compiled choreo kernels on GCU hardware.

    This class provides the main interface for executing PyTorch operations
    on GCU hardware using compiled choreo kernels, following choreo-op patterns.
    """

    def __init__(self, artifact: CompiledArtifact):
        """
        Initialize GCU kernel executor.

        Args:
            artifact: Compiled choreo artifact to execute
        """
        self.artifact = artifact
        self.host_interface = ChoreoHostInterface()
        self._shared_library = None
        self._wrapper_path = None
        self._is_loaded = False
        
    def load_kernel(self) -> None:
        """Load the compiled choreo kernel for execution following choreo-op patterns."""
        if self._is_loaded:
            return

        try:
            # Create host wrapper that interfaces with the choreo __co__ function
            self._create_host_wrapper()

            # Load the shared library using ctypes
            self._shared_library = self.host_interface.load_shared_library(self._wrapper_path)
            self._is_loaded = True
            logger.info(f"Successfully loaded GCU kernel: {self.artifact.entry_point}")

        except Exception as e:
            raise DeviceError(f"Failed to load GCU kernel: {e}")
    
    def _create_host_wrapper(self) -> None:
        """
        Create a host wrapper that interfaces with the choreo __co__ function.

        This generates a C++ wrapper following choreo-op patterns that properly
        calls the __co__ function with choreo spanview data structures.
        """
        # Find the choreo object file
        object_path = self._find_choreo_object_file()

        # Determine the actual number of inputs from the DSL content
        input_descriptors = self._get_input_descriptors_from_dsl()

        # Create output descriptor (assume same shape as first input for now)
        dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
        output_desc = self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)

        # Generate wrapper code using the host interface
        wrapper_code = self.host_interface.generate_host_wrapper_code(
            self.artifact.entry_point,
            input_descriptors,
            output_desc
        )

        # Compile wrapper with the choreo object file
        self._wrapper_path = tempfile.mktemp(suffix='.so')
        self.host_interface.compile_host_wrapper(wrapper_code, object_path, self._wrapper_path, self.artifact.entry_point)

    def _get_input_descriptors_from_dsl(self) -> List[ChoreoTensorDescriptor]:
        """
        Extract input descriptors from the DSL content to match the actual function signature.

        Returns:
            List of ChoreoTensorDescriptor objects matching the DSL function parameters
        """
        if 'dsl_content' not in self.artifact.metadata:
            # Fallback to single input if DSL content not available
            dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
            return [self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)]

        dsl_content = self.artifact.metadata['dsl_content']

        # Parse the function signature to determine number and types of inputs
        import re

        # Look for the __co__ function signature
        # Pattern: __co__ auto function_name(param1, param2, ...)
        pattern = r'__co__\s+auto\s+\w+\s*\(([^)]*)\)'
        match = re.search(pattern, dsl_content)

        if not match:
            # Fallback to single input
            dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
            return [self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)]

        params_str = match.group(1).strip()

        if not params_str:
            # No parameters
            return []

        # Parse parameters more carefully to handle commas inside brackets
        # Pattern: dtype [shape] name
        param_pattern = r'(\w+)\s*\[([^\]]*)\]\s*(\w+)'
        params = self._split_parameters_safely(params_str)

        input_descriptors = []
        for param in params:
            param_match = re.match(param_pattern, param)
            if param_match:
                dtype_str, shape_str, param_name = param_match.groups()

                # Convert choreo dtype to torch dtype
                torch_dtype = self._choreo_dtype_to_torch(dtype_str)

                # Parse shape - for now assume 2D [N, M]
                # TODO: Make this more robust for different shapes
                dummy_tensor = torch.randn(2, 3, dtype=torch_dtype)
                input_descriptors.append(self.host_interface.tensor_to_choreo_descriptor(dummy_tensor))
            else:
                # Fallback for unparseable parameter
                dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
                input_descriptors.append(self.host_interface.tensor_to_choreo_descriptor(dummy_tensor))

        return input_descriptors

    def _choreo_dtype_to_torch(self, choreo_dtype: str) -> torch.dtype:
        """Convert choreo dtype string to torch dtype."""
        mapping = {
            'f32': torch.float32,
            'f16': torch.float16,
            's32': torch.int32,
            's64': torch.int64,
            'bool': torch.bool,
        }
        return mapping.get(choreo_dtype, torch.float32)

    def _split_parameters_safely(self, params_str: str) -> List[str]:
        """
        Split parameter string safely, handling commas inside brackets.

        Args:
            params_str: Parameter string like "f32 [N, M] lhs, s32 [K] rhs"

        Returns:
            List of individual parameter strings
        """
        params = []
        current_param = ""
        bracket_depth = 0

        for char in params_str:
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == ',' and bracket_depth == 0:
                # This comma is a parameter separator, not inside brackets
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        # Add the last parameter
        if current_param.strip():
            params.append(current_param.strip())

        return params

    def _get_expected_input_count(self) -> int:
        """Get the expected number of inputs from the DSL function signature."""
        input_descriptors = self._get_input_descriptors_from_dsl()
        return len(input_descriptors)

    def _find_choreo_object_file(self) -> str:
        """Find the choreo object file corresponding to the shared library."""
        # First, try to get object file path from metadata
        if 'object_file_path' in self.artifact.metadata:
            object_path = self.artifact.metadata['object_file_path']
            if os.path.exists(object_path):
                return object_path

        # Fallback: derive from shared library path
        base_name = os.path.splitext(self.artifact.path)[0]
        object_path = f"{base_name}.o"

        if os.path.exists(object_path):
            return object_path

        # Try to find the DSL file and recompile
        dsl_path = self.artifact.metadata.get('dsl_file_path')
        if not dsl_path:
            # Look for DSL source file (.co extension for Choreo DSL source)
            dsl_path = f"{base_name}.co"

        if os.path.exists(dsl_path):
            self._recompile_object_file(dsl_path, object_path)
            return object_path

        raise DeviceError(f"Cannot find choreo object file for {self.artifact.path}")


    
    def _compile_wrapper(self, wrapper_cpp_path: str, output_path: str) -> None:
        """Compile the wrapper C++ code with the choreo object file."""
        # Find the original object file
        base_name = os.path.splitext(self.artifact.path)[0]
        object_path = f"{base_name}.o"
        
        if not os.path.exists(object_path):
            # Try to recompile from DSL if object file is missing
            # Look for DSL source file (.co extension for Choreo DSL source)
            dsl_path = f"{base_name}.co"
            if os.path.exists(dsl_path):
                self._recompile_object_file(dsl_path, object_path)
            else:
                raise DeviceError(f"Cannot find object file or DSL source for {self.artifact.path}")
        
        # Compile wrapper with choreo object file using topscc
        from ..utils.topscc_utils import compile_with_topscc, get_topscc_environment

        topscc_env = get_topscc_environment()
        include_dirs = [f'{os.path.dirname(os.path.abspath(__file__))}/../../choreo-headers']

        if topscc_env.is_available():
            logger.debug("Using topscc for wrapper compilation")
            success, error_msg = compile_with_topscc(
                [wrapper_cpp_path, object_path],
                output_path,
                include_dirs=include_dirs,
                extra_flags=['-shared'],
                timeout=60,
                host_code=True
            )

            if not success:
                raise DeviceError(f"topscc wrapper compilation failed: {error_msg}")
        else:
            # Fallback to g++ if topscc is not available
            logger.warning("topscc not available, falling back to g++ for wrapper compilation")
            compile_cmd = [
                'g++',
                '-shared',
                '-fPIC',
                '-std=c++17',
                f'-I{include_dirs[0]}',
                wrapper_cpp_path,
                object_path,
                '-o', output_path
            ]

            try:
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    raise DeviceError(f"Wrapper compilation failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                raise DeviceError("Wrapper compilation timed out")
            except FileNotFoundError:
                raise DeviceError("g++ compiler not found. Please ensure g++ is installed and in PATH.")
    
    def _recompile_object_file(self, dsl_path: str, object_path: str) -> None:
        """Recompile object file from DSL source if needed."""
        compile_cmd = [
            'choreo',
            '-c',
            '-fpic',
            dsl_path,
            '-o', object_path
        ]
        
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise DeviceError(f"Choreo recompilation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise DeviceError("Choreo recompilation timed out")
        except FileNotFoundError:
            raise DeviceError("choreo compiler not found. Please ensure choreo is installed and in PATH.")
    
    def execute(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Execute the kernel on GCU hardware with PyTorch tensors following choreo-op patterns.

        Args:
            inputs: List of input PyTorch tensors

        Returns:
            List of output PyTorch tensors

        Raises:
            DeviceError: If execution fails
        """
        if not self._is_loaded:
            self.load_kernel()

        if not inputs:
            raise DeviceError("No input tensors provided")

        try:
            # Validate inputs match expected number from DSL
            expected_inputs = self._get_expected_input_count()
            if len(inputs) != expected_inputs:
                raise DeviceError(f"Expected {expected_inputs} inputs, got {len(inputs)}")

            # Create output tensor with same shape as first input
            if inputs:
                output_tensor = torch.empty_like(inputs[0])
            else:
                # No inputs case - create a default output tensor
                output_tensor = torch.empty((2, 3), dtype=torch.float32)

            # Execute using the host interface
            result_code = self.host_interface.execute_kernel_with_library(
                self._shared_library, inputs, output_tensor
            )

            if result_code != 0:
                raise ExecutionError(f"Kernel execution failed with code {result_code}")

            logger.debug(f"Successfully executed GCU kernel with {len(inputs)} inputs")
            return [output_tensor]

        except Exception as e:
            raise ExecutionError(f"GCU kernel execution failed: {e}")
    
    def unload(self) -> None:
        """Unload the kernel and clean up resources."""
        if self._shared_library:
            # ctypes libraries don't need explicit unloading
            self._shared_library = None

        # Clean up wrapper file
        if self._wrapper_path and os.path.exists(self._wrapper_path):
            try:
                os.unlink(self._wrapper_path)
            except:
                pass

        self._is_loaded = False
        logger.debug("GCU kernel unloaded")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload()
