"""
Function Signature Generation Module.

This module provides modular components for generating function signatures
in various DSL formats, improving maintainability and reusability.
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..utils.logging import get_logger

# Avoid importing graph modules here to prevent circular imports.
# Types are available via postponed evaluation from __future__ import annotations.

logger = get_logger(__name__)


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    dtype: str
    shape: Optional[list[int]] = None
    is_input: bool = True
    is_output: bool = False
    qualifiers: list[str] = None

    def __post_init__(self):
        if self.qualifiers is None:
            self.qualifiers = []


@dataclass
class SignatureInfo:
    """Complete function signature information."""

    function_name: str
    parameters: list[ParameterInfo]
    return_type: str = "auto"
    qualifiers: list[str] = None
    attributes: dict[str, Any] = None

    def __post_init__(self):
        if self.qualifiers is None:
            self.qualifiers = []
        if self.attributes is None:
            self.attributes = {}


class SignatureGenerator(ABC):
    """Abstract base class for function signature generators."""

    @abstractmethod
    def generate_signature(self, signature_info: SignatureInfo) -> str:
        """Generate function signature string."""
        pass

    @abstractmethod
    def generate_parameter(self, param_info: ParameterInfo) -> str:
        """Generate parameter string."""
        pass

    @abstractmethod
    def get_dtype_mapping(self) -> dict[str, str]:
        """Get data type mapping for this target."""
        pass


class ChoreoSignatureGenerator(SignatureGenerator):
    """Function signature generator for Choreo DSL."""

    def generate_signature(self, signature_info: SignatureInfo) -> str:
        """Generate Choreo __co__ function signature."""
        # Generate parameter list
        param_strings = []
        for param in signature_info.parameters:
            param_str = self.generate_parameter(param)
            param_strings.append(param_str)

        # Combine qualifiers
        qualifiers = " ".join(signature_info.qualifiers) if signature_info.qualifiers else ""
        if qualifiers:
            qualifiers += " "

        # Generate complete signature
        param_list = ", ".join(param_strings)
        return f"__{qualifiers}co__ {signature_info.return_type} {signature_info.function_name}({param_list})"

    def generate_parameter(self, param_info: ParameterInfo) -> str:
        """Generate Choreo parameter string."""
        dtype = self._map_dtype(param_info.dtype)

        # Handle array parameters
        if param_info.shape:
            shape_str = ", ".join(map(str, param_info.shape))
            return f"{dtype} [{shape_str}] {param_info.name}"
        else:
            return f"{dtype} {param_info.name}"

    def get_dtype_mapping(self) -> dict[str, str]:
        """Get Choreo data type mapping."""
        return {
            "float32": "f32",
            "float64": "f64",
            "int32": "s32",
            "int64": "s64",
            "uint32": "u32",
            "uint64": "u64",
            "bool": "bool",
            "float": "f32",
            "int": "s32",
        }

    def _map_dtype(self, dtype: str) -> str:
        """Map PyTorch dtype to Choreo dtype."""
        mapping = self.get_dtype_mapping()
        return mapping.get(dtype, "f32")


class CudaSignatureGenerator(SignatureGenerator):
    """Function signature generator for CUDA kernels."""

    def generate_signature(self, signature_info: SignatureInfo) -> str:
        """Generate CUDA kernel signature."""
        param_strings = []
        for param in signature_info.parameters:
            param_str = self.generate_parameter(param)
            param_strings.append(param_str)

        qualifiers = (
            " ".join(signature_info.qualifiers) if signature_info.qualifiers else "__global__"
        )
        param_list = ", ".join(param_strings)

        return f"{qualifiers} {signature_info.return_type} {signature_info.function_name}({param_list})"

    def generate_parameter(self, param_info: ParameterInfo) -> str:
        """Generate CUDA parameter string."""
        dtype = self._map_dtype(param_info.dtype)

        # CUDA uses pointers for array parameters
        if param_info.shape:
            return f"{dtype}* {param_info.name}"
        else:
            return f"{dtype} {param_info.name}"

    def get_dtype_mapping(self) -> dict[str, str]:
        """Get CUDA data type mapping."""
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "int64": "long long",
            "uint32": "unsigned int",
            "uint64": "unsigned long long",
            "bool": "bool",
            "float": "float",
            "int": "int",
        }

    def _map_dtype(self, dtype: str) -> str:
        """Map PyTorch dtype to CUDA dtype."""
        mapping = self.get_dtype_mapping()
        return mapping.get(dtype, "float")


class CPPSignatureGenerator(SignatureGenerator):
    """Function signature generator for C++ functions."""

    def generate_signature(self, signature_info: SignatureInfo) -> str:
        """Generate C++ function signature."""
        param_strings = []
        for param in signature_info.parameters:
            param_str = self.generate_parameter(param)
            param_strings.append(param_str)

        qualifiers = " ".join(signature_info.qualifiers) if signature_info.qualifiers else ""
        if qualifiers:
            qualifiers += " "

        param_list = ", ".join(param_strings)
        return (
            f"{qualifiers}{signature_info.return_type} {signature_info.function_name}({param_list})"
        )

    def generate_parameter(self, param_info: ParameterInfo) -> str:
        """Generate C++ parameter string."""
        dtype = self._map_dtype(param_info.dtype)

        # C++ uses pointers or references for array parameters
        if param_info.shape:
            return f"{dtype}* {param_info.name}"
        else:
            return f"{dtype} {param_info.name}"

    def get_dtype_mapping(self) -> dict[str, str]:
        """Get C++ data type mapping."""
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "bool": "bool",
            "float": "float",
            "int": "int",
        }

    def _map_dtype(self, dtype: str) -> str:
        """Map PyTorch dtype to C++ dtype."""
        mapping = self.get_dtype_mapping()
        return mapping.get(dtype, "float")


class SignatureBuilder:
    """Builder for constructing function signatures from DAGs."""

    def __init__(self, generator: SignatureGenerator):
        self.generator = generator

    def build_from_dag(
        self, dag: ComputationDAG, function_name: str, qualifiers: Optional[list[str]] = None
    ) -> SignatureInfo:
        """Build signature info from computation DAG."""
        parameters = []

        # Add input parameters
        for buf in dag.inputs:
            param = ParameterInfo(
                name=buf.name,
                dtype=buf.dtype or "float32",
                shape=buf.shape,
                is_input=True,
                is_output=False,
            )
            parameters.append(param)

        # For some targets, we might need to add output parameters
        # (Choreo handles this differently with return values)

        return SignatureInfo(
            function_name=function_name,
            parameters=parameters,
            return_type="auto",
            qualifiers=qualifiers or [],
        )

    def build_kernel_signature(
        self,
        operation: str,
        input_shapes: list[list[int]],
        output_shapes: list[list[int]],
        dtype: str = "float32",
    ) -> SignatureInfo:
        """Build signature for device kernel."""
        parameters = []

        # Add input parameters
        for i, shape in enumerate(input_shapes):
            param = ParameterInfo(name=f"input_{i}", dtype=dtype, shape=shape, is_input=True)
            parameters.append(param)

        # Add output parameters
        for i, shape in enumerate(output_shapes):
            param = ParameterInfo(name=f"output_{i}", dtype=dtype, shape=shape, is_output=True)
            parameters.append(param)

        # Add dimension parameters for kernels
        if operation in ["matmul", "conv2d"]:
            dim_params = self._get_dimension_parameters(operation, input_shapes)
            parameters.extend(dim_params)

        return SignatureInfo(
            function_name=f"{operation}_kernel",
            parameters=parameters,
            return_type="void",
            qualifiers=["__co_device__", 'extern "C"'],
        )

    def _get_dimension_parameters(
        self, operation: str, shapes: list[list[int]]
    ) -> list[ParameterInfo]:
        """Get dimension parameters for specific operations."""
        params = []

        if operation == "matmul":
            params.extend(
                [
                    ParameterInfo("m", "int32"),
                    ParameterInfo("k", "int32"),
                    ParameterInfo("n", "int32"),
                ]
            )
        elif operation == "conv2d":
            params.extend(
                [
                    ParameterInfo("batch", "int32"),
                    ParameterInfo("in_channels", "int32"),
                    ParameterInfo("out_channels", "int32"),
                    ParameterInfo("height", "int32"),
                    ParameterInfo("width", "int32"),
                    ParameterInfo("kernel_h", "int32"),
                    ParameterInfo("kernel_w", "int32"),
                ]
            )

        return params

    def generate_signature_string(self, signature_info: SignatureInfo) -> str:
        """Generate signature string using the configured generator."""
        return self.generator.generate_signature(signature_info)


class SignatureRegistry:
    """Registry for managing signature generators."""

    def __init__(self):
        self.generators: dict[str, SignatureGenerator] = {}
        self._initialize_builtin_generators()

    def register_generator(self, name: str, generator: SignatureGenerator) -> None:
        """Register a signature generator."""
        self.generators[name] = generator
        logger.info(f"Registered signature generator: {name}")

    def get_generator(self, name: str) -> Optional[SignatureGenerator]:
        """Get signature generator by name."""
        return self.generators.get(name)

    def create_builder(self, target: str) -> Optional[SignatureBuilder]:
        """Create signature builder for target."""
        generator = self.get_generator(target)
        if generator:
            return SignatureBuilder(generator)
        return None

    def _initialize_builtin_generators(self) -> None:
        """Initialize built-in signature generators."""
        self.register_generator("choreo", ChoreoSignatureGenerator())
        self.register_generator("cuda", CudaSignatureGenerator())
        self.register_generator("cpp", CPPSignatureGenerator())


# Global signature registry
signature_registry = SignatureRegistry()


def get_signature_registry() -> SignatureRegistry:
    """Get the global signature registry."""
    return signature_registry


def create_signature_builder(target: str = "choreo") -> Optional[SignatureBuilder]:
    """Create signature builder for target."""
    return signature_registry.create_builder(target)
