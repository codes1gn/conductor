"""
GCU Device Execution Engine.

This module implements the host-side integration for executing compiled choreo
kernels on GCU hardware. It handles tensor marshalling between PyTorch and
choreo formats, device memory management, and kernel execution coordination.
"""

from __future__ import annotations

import os
import tempfile
import subprocess
import torch

from .loader import CompiledArtifact
from .host_wrapper import ChoreoHostInterface, ChoreoTensorDescriptor
from ..utils.exceptions import DeviceError, ExecutionError
from ..utils.logging import get_logger
from ..utils.type_mapping import choreo_to_torch_dtype, get_type_mapper
from ..utils.symbolic_shapes import get_symbolic_shape_resolver, resolve_symbolic_shape, infer_shape_context

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
            self.artifact.entry_point, input_descriptors, output_desc
        )

        # Compile wrapper with the choreo object file
        self._wrapper_path = tempfile.mktemp(suffix=".so")
        self.host_interface.compile_host_wrapper(
            wrapper_code, object_path, self._wrapper_path, self.artifact.entry_point
        )

    def _get_input_descriptors_from_dsl(self) -> list[ChoreoTensorDescriptor]:
        """
        Extract input descriptors from the DSL content to match the actual function signature.

        Returns:
            List of ChoreoTensorDescriptor objects matching the DSL function parameters
        """
        if "dsl_content" not in self.artifact.metadata:
            # Fallback to single input if DSL content not available
            dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
            return [self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)]

        dsl_content = self.artifact.metadata["dsl_content"]

        # Parse the function signature to determine number and types of inputs
        import re

        # Look for the __co__ function signature
        # Pattern: __co__ return_type function_name(param1, param2, ...)
        # The return type can be 'auto' or a type with shape like 'f32 [16, 32]'
        pattern = r"__co__\s+(?:auto|\w+(?:\s*\[[^\]]*\])?)\s+(\w+)\s*\(([^)]*)\)"
        match = re.search(pattern, dsl_content)

        if not match:
            # Fallback to single input
            dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
            return [self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)]

        params_str = match.group(2).strip()

        if not params_str:
            # No parameters
            return []

        # Parse parameters more carefully to handle commas inside brackets
        # Pattern: dtype [shape] name
        param_pattern = r"(\w+)\s*\[([^\]]*)\]\s*(\w+)"
        params = self._split_parameters_safely(params_str)

        input_descriptors = []
        for param in params:
            param_match = re.match(param_pattern, param)
            if param_match:
                dtype_str, shape_str, param_name = param_match.groups()

                # Convert choreo dtype to torch dtype
                torch_dtype = self._choreo_dtype_to_torch(dtype_str)

                # Parse shape from the DSL signature with context
                context = self._get_shape_context()
                shape = self._parse_shape_from_dsl(shape_str, context)
                dummy_tensor = torch.randn(*shape, dtype=torch_dtype)
                input_descriptors.append(
                    self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)
                )
            else:
                # Fallback for unparseable parameter
                dummy_tensor = torch.randn(2, 3, dtype=torch.float32)
                input_descriptors.append(
                    self.host_interface.tensor_to_choreo_descriptor(dummy_tensor)
                )

        return input_descriptors

    def _choreo_dtype_to_torch(self, choreo_dtype: str) -> torch.dtype:
        """Convert choreo dtype string to torch dtype using centralized type mapping."""
        try:
            return choreo_to_torch_dtype(choreo_dtype)
        except ValueError as e:
            logger.warning(f"Unsupported Choreo dtype {choreo_dtype}, defaulting to float32: {e}")
            return torch.float32

    def _validate_and_cast_tensor(self, tensor: torch.Tensor, expected_dtype: str, param_name: str) -> torch.Tensor:
        """
        Validate and cast tensor to expected dtype with comprehensive error handling.

        Args:
            tensor: Input tensor to validate/cast
            expected_dtype: Expected Choreo dtype string
            param_name: Parameter name for error messages

        Returns:
            Tensor cast to the correct dtype

        Raises:
            ValueError: If casting fails or types are incompatible
        """
        try:
            target_torch_dtype = self._choreo_dtype_to_torch(expected_dtype)

            # Check if tensor is already the correct type
            if tensor.dtype == target_torch_dtype:
                return tensor

            # Attempt to cast
            logger.debug(f"Casting {param_name} from {tensor.dtype} to {target_torch_dtype}")
            return tensor.to(target_torch_dtype)

        except Exception as e:
            raise ValueError(f"Failed to cast {param_name} to {expected_dtype}: {e}")

    def _get_supported_dtypes(self) -> list[str]:
        """Get list of supported Choreo dtypes."""
        type_mapper = get_type_mapper()
        return [dtype.value for dtype in type_mapper.get_supported_choreo_types()]

    def _validate_input_tensors(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Validate input tensors and cast them to expected dtypes if needed.

        Args:
            inputs: List of input tensors

        Returns:
            List of validated/cast tensors

        Raises:
            DeviceError: If validation fails
        """
        try:
            # Get expected input types from DSL if available
            expected_dtypes = self._get_expected_input_dtypes()

            validated_inputs = []
            for i, tensor in enumerate(inputs):
                if i < len(expected_dtypes):
                    expected_dtype = expected_dtypes[i]
                    validated_tensor = self._validate_and_cast_tensor(
                        tensor, expected_dtype, f"input_{i}"
                    )
                    validated_inputs.append(validated_tensor)
                else:
                    # No expected dtype info, use tensor as-is
                    validated_inputs.append(tensor)

            return validated_inputs

        except Exception as e:
            logger.warning(f"Input validation failed, using original tensors: {e}")
            return inputs

    def _get_expected_input_dtypes(self) -> list[str]:
        """
        Extract expected input dtypes from DSL signature.

        Returns:
            List of expected Choreo dtype strings
        """
        if "dsl_content" not in self.artifact.metadata:
            return []

        dsl_content = self.artifact.metadata["dsl_content"]

        # Parse function signature to extract parameter types
        import re
        pattern = r"__co__\s+\w+\s+\w+\s*\((.*?)\)"
        match = re.search(pattern, dsl_content)

        if not match:
            return []

        params_str = match.group(1).strip()
        if not params_str:
            return []

        dtypes = []
        # Parse parameters like "f32 [16, 32] lhs, f32 [16, 32] rhs" using safe splitting
        params = self._split_parameters_safely(params_str)
        for param in params:
            param = param.strip()
            if param:
                # Extract dtype using regex to handle complex patterns
                import re
                dtype_match = re.match(r'^(\w+)\s+\[', param)
                if dtype_match:
                    dtype = dtype_match.group(1)
                    dtypes.append(dtype)
                else:
                    # Fallback: extract first word
                    parts = param.split()
                    if parts:
                        dtype = parts[0]
                        dtypes.append(dtype)

        return dtypes

    def _parse_shape_from_dsl(self, shape_str: str, context: Optional[Dict[str, int]] = None) -> list[int]:
        """
        Parse shape string from DSL signature with robust symbolic shape handling.

        Args:
            shape_str: Shape string like "16, 32" or "N, M"
            context: Optional context for symbolic dimension resolution

        Returns:
            List of integers representing the shape
        """
        try:
            # Use robust symbolic shape resolver
            resolved_shape = resolve_symbolic_shape(shape_str, context)
            return resolved_shape if resolved_shape else [16, 32]
        except Exception as e:
            logger.warning(f"Failed to parse shape '{shape_str}': {e}, using fallback")
            # Fallback to simple parsing
            dimensions = [dim.strip() for dim in shape_str.split(",")]
            shape = []

            for dim in dimensions:
                if dim.isdigit():
                    shape.append(int(dim))
                else:
                    # Symbolic dimension - use a default size
                    shape.append(16)  # Default size for symbolic dimensions

            return shape if shape else [16, 32]  # Default shape if parsing fails

    def _get_shape_context(self) -> Dict[str, int]:
        """
        Get shape context for symbolic dimension resolution.

        Returns:
            Context dictionary with known dimension values
        """
        context = {}

        # Try to infer context from artifact metadata if available
        if hasattr(self, 'artifact') and self.artifact and hasattr(self.artifact, 'metadata'):
            metadata = self.artifact.metadata

            # Look for example inputs in metadata
            if 'example_inputs' in metadata:
                try:
                    example_inputs = metadata['example_inputs']
                    if example_inputs:
                        context = infer_shape_context(example_inputs)
                except Exception as e:
                    logger.debug(f"Failed to infer context from example inputs: {e}")

            # Look for input shapes in metadata
            if 'input_shapes' in metadata:
                try:
                    input_shapes = metadata['input_shapes']
                    if input_shapes and len(input_shapes) > 0:
                        first_shape = input_shapes[0]
                        if len(first_shape) >= 1:
                            context.setdefault("N", first_shape[0])
                        if len(first_shape) >= 2:
                            context.setdefault("M", first_shape[1])
                        if len(first_shape) >= 3:
                            context.setdefault("K", first_shape[2])
                except Exception as e:
                    logger.debug(f"Failed to infer context from input shapes: {e}")

        # Add some reasonable defaults if no context found
        if not context:
            context = {
                "N": 16,  # Batch size
                "M": 32,  # Feature dimension
                "K": 32,  # Another feature dimension
                "L": 64,  # Sequence length
                "H": 64,  # Height
                "W": 64,  # Width
                "C": 32,  # Channels
            }

        logger.debug(f"Shape context: {context}")
        return context

    def _validate_input_shapes(self, inputs: List[torch.Tensor]) -> bool:
        """
        Validate input tensor shapes against expected DSL signature shapes.

        Args:
            inputs: List of input tensors

        Returns:
            True if shapes are compatible, False otherwise
        """
        try:
            # Get expected shapes from DSL
            expected_shapes = self._get_expected_input_shapes()

            if len(expected_shapes) != len(inputs):
                logger.warning(f"Shape count mismatch: expected {len(expected_shapes)}, got {len(inputs)}")
                return False

            resolver = get_symbolic_shape_resolver()

            for i, (expected_shape_str, input_tensor) in enumerate(zip(expected_shapes, inputs)):
                actual_shape = list(input_tensor.shape)

                if not resolver.validate_shape_compatibility(expected_shape_str, actual_shape):
                    logger.warning(f"Input {i} shape mismatch: expected {expected_shape_str}, got {actual_shape}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Shape validation failed: {e}")
            return False

    def _get_expected_input_shapes(self) -> List[str]:
        """
        Extract expected input shapes from DSL signature.

        Returns:
            List of expected shape strings
        """
        if "dsl_content" not in self.artifact.metadata:
            return []

        dsl_content = self.artifact.metadata["dsl_content"]

        # Parse function signature to extract parameter shapes
        import re
        pattern = r"__co__\s+\w+\s+\w+\s*\((.*?)\)"
        match = re.search(pattern, dsl_content)

        if not match:
            return []

        params_str = match.group(1).strip()
        if not params_str:
            return []

        shapes = []
        # Parse parameters like "f32 [16, 32] lhs, f32 [N, M] rhs"
        params = self._split_parameters_safely(params_str)
        for param in params:
            param = param.strip()
            if param:
                # Extract shape using regex
                import re
                shape_match = re.search(r'\[([^\]]+)\]', param)
                if shape_match:
                    shape_str = shape_match.group(1)
                    shapes.append(shape_str)

        return shapes

    def _split_parameters_safely(self, params_str: str) -> list[str]:
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
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "," and bracket_depth == 0:
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
        if "object_file_path" in self.artifact.metadata:
            object_path = self.artifact.metadata["object_file_path"]
            if os.path.exists(object_path):
                return object_path

        # Fallback: derive from shared library path
        base_name = os.path.splitext(self.artifact.path)[0]
        object_path = f"{base_name}.o"

        if os.path.exists(object_path):
            return object_path

        # Try to find the DSL file and recompile
        dsl_path = self.artifact.metadata.get("dsl_file_path")
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
        from .topscc_utils import compile_with_topscc, get_topscc_environment

        topscc_env = get_topscc_environment()
        include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/../../choreo-headers"]

        if topscc_env.is_available():
            logger.debug("Using topscc for wrapper compilation")
            success, error_msg = compile_with_topscc(
                [wrapper_cpp_path, object_path],
                output_path,
                include_dirs=include_dirs,
                extra_flags=["-shared"],
                timeout=60,
                host_code=True,
            )

            if not success:
                raise DeviceError(f"topscc wrapper compilation failed: {error_msg}")
        else:
            # Fallback to g++ if topscc is not available
            logger.warning("topscc not available, falling back to g++ for wrapper compilation")
            compile_cmd = [
                "g++",
                "-shared",
                "-fPIC",
                "-std=c++17",
                f"-I{include_dirs[0]}",
                wrapper_cpp_path,
                object_path,
                "-o",
                output_path,
            ]

            try:
                result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    raise DeviceError(f"Wrapper compilation failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                raise DeviceError("Wrapper compilation timed out")
            except FileNotFoundError:
                raise DeviceError(
                    "g++ compiler not found. Please ensure g++ is installed and in PATH."
                )

    def _recompile_object_file(self, dsl_path: str, object_path: str) -> None:
        """Recompile object file from DSL source if needed."""
        compile_cmd = ["choreo", "-c", "-fpic", dsl_path, "-o", object_path]

        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                raise DeviceError(f"Choreo recompilation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise DeviceError("Choreo recompilation timed out")
        except FileNotFoundError:
            raise DeviceError(
                "choreo compiler not found. Please ensure choreo is installed and in PATH."
            )

    def execute(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
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

            # Validate input shapes against expected symbolic shapes
            if not self._validate_input_shapes(inputs):
                logger.warning("Input shape validation failed, proceeding with execution")

            # Validate and potentially cast input tensors to expected dtypes
            validated_inputs = self._validate_input_tensors(inputs)

            # Create output tensor with same shape as first input
            if validated_inputs:
                output_tensor = torch.empty_like(validated_inputs[0])
            else:
                # No inputs case - create a default output tensor
                output_tensor = torch.empty((2, 3), dtype=torch.float32)

            # Execute using the host interface with validated inputs
            result_code = self.host_interface.execute_kernel_with_library(
                self._shared_library, validated_inputs, output_tensor
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
            os.unlink(self._wrapper_path)

        self._is_loaded = False
        logger.debug("GCU kernel unloaded")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload()
