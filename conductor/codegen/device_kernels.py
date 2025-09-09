"""
Device Kernel Implementation for Conductor.

This module provides authentic device kernel implementations for complex operations
like matmul, conv2d, and attention, following choreo-op patterns and syntax.
"""

from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..graph.graph_nodes import ConductorNode

logger = get_logger(__name__)


@dataclass
class KernelMetadata:
    """Metadata for device kernel generation."""

    operation: str
    template_vars: dict[str, Any]
    includes: list[str]
    parameters: list[str]
    local_memory_size: Optional[int] = None
    shared_memory_size: Optional[int] = None


class DeviceKernel(ABC):
    """Abstract base class for device kernel implementations."""

    @abstractmethod
    def get_operation_name(self) -> str:
        """Get the operation name this kernel handles."""
        pass

    @abstractmethod
    def generate_kernel_declaration(self, node: ConductorNode) -> list[str]:
        """Generate the __co_device__ kernel declaration."""
        pass

    @abstractmethod
    def generate_kernel_implementation(self, node: ConductorNode) -> list[str]:
        """Generate the kernel implementation code."""
        pass

    @abstractmethod
    def get_kernel_call_syntax(
        self, node: ConductorNode, input_vars: list[str], output_vars: list[str]
    ) -> str:
        """Generate the call syntax for this kernel in __co__ functions."""
        pass

    def get_required_includes(self) -> list[str]:
        """Get required includes for this kernel."""
        return []

    def get_metadata(self, node: ConductorNode) -> KernelMetadata:
        """Get kernel metadata for optimization."""
        return KernelMetadata(
            operation=self.get_operation_name(),
            template_vars={},
            includes=self.get_required_includes(),
            parameters=[],
        )


class MatMulDeviceKernel(DeviceKernel):
    """Device kernel for matrix multiplication operations."""

    def get_operation_name(self) -> str:
        return "matmul"

    def generate_kernel_declaration(self, node: ConductorNode) -> list[str]:
        """Generate matmul kernel declaration following choreo-op patterns."""
        return [
            '__co_device__ extern "C" void matmul_kernel(float* lhs, float* rhs, float* out, int m, int k, int n) {'
        ]

    def generate_kernel_implementation(self, node: ConductorNode) -> list[str]:
        """Generate optimized matmul kernel implementation."""
        # Based on choreo-op/matmul-gcu-only-test.co
        return [
            '__co_device__ extern "C" void matmul_kernel(float* lhs, float* rhs, float* out, int m, int k, int n) {',
            "  // Initialize output to zero",
            "  for (int i = 0; i < m * n; ++i) {",
            "    out[i] = 0.0f;",
            "  }",
            "  ",
            "  // Perform matrix multiplication",
            "  for (int i = 0; i < m; ++i) {",
            "    for (int j = 0; j < n; ++j) {",
            "      for (int z = 0; z < k; ++z) {",
            "        out[i * n + j] += lhs[i * k + z] * rhs[z * n + j];",
            "      }",
            "    }",
            "  }",
            "}",
        ]

    def get_kernel_call_syntax(
        self, node: ConductorNode, input_vars: list[str], output_vars: list[str]
    ) -> str:
        """Generate call syntax for matmul kernel."""
        if len(input_vars) >= 2 and len(output_vars) >= 1:
            return f"call matmul_kernel({input_vars[0]}.data, {input_vars[1]}.data, {output_vars[0]}.data, m, k, n);"
        return f"call matmul_kernel(lhs.data, rhs.data, out.data, m, k, n);"

    def get_metadata(self, node: ConductorNode) -> KernelMetadata:
        return KernelMetadata(
            operation="matmul",
            template_vars={"m": 64, "k": 64, "n": 64},
            includes=[],
            parameters=["m", "k", "n"],
            local_memory_size=64 * 64 * 4,  # Assume float32
            shared_memory_size=128 * 64 * 4,
        )


class Conv2DDeviceKernel(DeviceKernel):
    """Device kernel for 2D convolution operations."""

    def get_operation_name(self) -> str:
        return "conv2d"

    def generate_kernel_declaration(self, node: ConductorNode) -> list[str]:
        return [
            '__co_device__ extern "C" void conv2d_kernel(float* input, float* weight, float* output, '
            "int batch, int in_channels, int out_channels, int height, int width, int kernel_h, int kernel_w) {"
        ]

    def generate_kernel_implementation(self, node: ConductorNode) -> list[str]:
        """Generate conv2d kernel implementation."""
        return [
            '__co_device__ extern "C" void conv2d_kernel(float* input, float* weight, float* output,',
            "                                              int batch, int in_channels, int out_channels,",
            "                                              int height, int width, int kernel_h, int kernel_w) {",
            "  // Simple convolution implementation",
            "  for (int b = 0; b < batch; ++b) {",
            "    for (int oc = 0; oc < out_channels; ++oc) {",
            "      for (int h = 0; h < height - kernel_h + 1; ++h) {",
            "        for (int w = 0; w < width - kernel_w + 1; ++w) {",
            "          float sum = 0.0f;",
            "          for (int ic = 0; ic < in_channels; ++ic) {",
            "            for (int kh = 0; kh < kernel_h; ++kh) {",
            "              for (int kw = 0; kw < kernel_w; ++kw) {",
            "                int input_idx = b * in_channels * height * width +",
            "                               ic * height * width +",
            "                               (h + kh) * width + (w + kw);",
            "                int weight_idx = oc * in_channels * kernel_h * kernel_w +",
            "                                ic * kernel_h * kernel_w +",
            "                                kh * kernel_w + kw;",
            "                sum += input[input_idx] * weight[weight_idx];",
            "              }",
            "            }",
            "          }",
            "          int output_idx = b * out_channels * (height - kernel_h + 1) * (width - kernel_w + 1) +",
            "                          oc * (height - kernel_h + 1) * (width - kernel_w + 1) +",
            "                          h * (width - kernel_w + 1) + w;",
            "          output[output_idx] = sum;",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
        ]

    def get_kernel_call_syntax(
        self, node: ConductorNode, input_vars: list[str], output_vars: list[str]
    ) -> str:
        """Generate call syntax for conv2d kernel."""
        if len(input_vars) >= 2 and len(output_vars) >= 1:
            return f"call conv2d_kernel({input_vars[0]}.data, {input_vars[1]}.data, {output_vars[0]}.data, batch, in_channels, out_channels, height, width, kernel_h, kernel_w);"
        return f"call conv2d_kernel(input.data, weight.data, output.data, batch, in_channels, out_channels, height, width, kernel_h, kernel_w);"


class AttentionDeviceKernel(DeviceKernel):
    """Device kernel for attention operations (Flash Attention style)."""

    def get_operation_name(self) -> str:
        return "attention"

    def generate_kernel_declaration(self, node: ConductorNode) -> list[str]:
        return [
            '__co_device__ extern "C" void flash_attention_kernel(float* q, float* k, float* v, float* output, '
            "int seq_len, int head_dim, float* max_value) {"
        ]

    def generate_kernel_implementation(self, node: ConductorNode) -> list[str]:
        """Generate flash attention kernel based on choreo-op/flashattention-gcu-only-test.co."""
        return [
            '__co_device__ extern "C" void flash_attention_kernel(float* q, float* k, float* v, float* output,',
            "                                                       int seq_len, int head_dim, float* max_value) {",
            "  // Simplified Flash Attention implementation",
            "  for (int i = 0; i < seq_len; ++i) {",
            "    // Compute attention scores",
            "    float max_score = -1e9f;",
            "    for (int j = 0; j < seq_len; ++j) {",
            "      float score = 0.0f;",
            "      for (int d = 0; d < head_dim; ++d) {",
            "        score += q[i * head_dim + d] * k[j * head_dim + d];",
            "      }",
            "      if (score > max_score) max_score = score;",
            "    }",
            "    ",
            "    // Compute softmax and weighted sum",
            "    float sum_exp = 0.0f;",
            "    for (int j = 0; j < seq_len; ++j) {",
            "      float score = 0.0f;",
            "      for (int d = 0; d < head_dim; ++d) {",
            "        score += q[i * head_dim + d] * k[j * head_dim + d];",
            "      }",
            "      sum_exp += expf(score - max_score);",
            "    }",
            "    ",
            "    for (int d = 0; d < head_dim; ++d) {",
            "      float weighted_sum = 0.0f;",
            "      for (int j = 0; j < seq_len; ++j) {",
            "        float score = 0.0f;",
            "        for (int dim = 0; dim < head_dim; ++dim) {",
            "          score += q[i * head_dim + dim] * k[j * head_dim + dim];",
            "        }",
            "        float attention_weight = expf(score - max_score) / sum_exp;",
            "        weighted_sum += attention_weight * v[j * head_dim + d];",
            "      }",
            "      output[i * head_dim + d] = weighted_sum;",
            "    }",
            "    ",
            "    max_value[i] = max_score;",
            "  }",
            "}",
        ]

    def get_kernel_call_syntax(
        self, node: ConductorNode, input_vars: list[str], output_vars: list[str]
    ) -> str:
        """Generate call syntax for attention kernel."""
        if len(input_vars) >= 3 and len(output_vars) >= 1:
            return f"call flash_attention_kernel({input_vars[0]}.data, {input_vars[1]}.data, {input_vars[2]}.data, {output_vars[0]}.data, seq_len, head_dim, max_value.data);"
        return f"call flash_attention_kernel(q.data, k.data, v.data, output.data, seq_len, head_dim, max_value.data);"


class ReLUDeviceKernel(DeviceKernel):
    """Device kernel for ReLU activation."""

    def get_operation_name(self) -> str:
        return "relu"

    def generate_kernel_declaration(self, node: ConductorNode) -> list[str]:
        return [
            "template<int size>",
            "__co_device__ void relu_kernel(float* input, float* output) {",
        ]

    def generate_kernel_implementation(self, node: ConductorNode) -> list[str]:
        """Generate ReLU kernel implementation."""
        return [
            "template<int size>",
            "__co_device__ void relu_kernel(float* input, float* output) {",
            "  for (int i = 0; i < size; ++i) {",
            "    output[i] = input[i] > 0.0f ? input[i] : 0.0f;",
            "  }",
            "}",
        ]

    def get_kernel_call_syntax(
        self, node: ConductorNode, input_vars: list[str], output_vars: list[str]
    ) -> str:
        """Generate call syntax for ReLU kernel."""
        if len(input_vars) >= 1 and len(output_vars) >= 1:
            return f"call relu_kernel<buffer_size>({input_vars[0]}.data, {output_vars[0]}.data);"
        return f"call relu_kernel<buffer_size>(input.data, output.data);"


class DeviceKernelRegistry:
    """Registry for managing device kernels."""

    def __init__(self):
        self.kernels: dict[str, DeviceKernel] = {}
        self._initialize_builtin_kernels()

    def register_kernel(self, kernel: DeviceKernel) -> None:
        """Register a device kernel."""
        op_name = kernel.get_operation_name()
        self.kernels[op_name] = kernel
        logger.info(f"Registered device kernel: {op_name}")

    def get_kernel(self, operation: str) -> Optional[DeviceKernel]:
        """Get device kernel for operation."""
        return self.kernels.get(operation)

    def has_kernel(self, operation: str) -> bool:
        """Check if kernel exists for operation."""
        return operation in self.kernels

    def list_supported_operations(self) -> list[str]:
        """List all operations with device kernels."""
        return list(self.kernels.keys())

    def _initialize_builtin_kernels(self) -> None:
        """Initialize built-in device kernels."""
        self.register_kernel(MatMulDeviceKernel())
        self.register_kernel(Conv2DDeviceKernel())
        self.register_kernel(AttentionDeviceKernel())
        self.register_kernel(ReLUDeviceKernel())
        logger.info("Initialized built-in device kernels")


def get_device_kernel_registry() -> DeviceKernelRegistry:
    """Get the device kernel registry from the global context."""
    from ..context import ensure_context_initialized
    context = ensure_context_initialized()
    return context.get_device_kernel_registry()


def register_device_kernel(kernel: DeviceKernel) -> None:
    """Register a device kernel globally."""
    get_device_kernel_registry().register_kernel(kernel)
