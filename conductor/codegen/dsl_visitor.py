"""
DSL Generation Visitor Pattern Implementation.

This module implements the Visitor Pattern for DSL generation following
LLVM-style codegen design patterns for better extensibility and maintainability.
"""

from __future__ import annotations

from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Forward-referenced types are resolved at runtime by future annotations.
# Avoid importing graph modules here to prevent circular imports.
from .device_kernels import device_kernel_registry
from .function_signature import create_signature_builder
from ..graph.parallel_fusion import analyze_parallel_fusion, ParallelStrategy
from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodegenContext:
    """Context for code generation with visitor state."""
    indent_level: int = 0
    current_function: Optional[str] = None
    local_vars: dict[str, str] = None
    parallel_factor: int = 4
    buffer_size: int = 16
    
    def __post_init__(self):
        if self.local_vars is None:
            self.local_vars = {}
    
    def indent(self, text: str) -> str:
        """Apply indentation to text."""
        return "  " * self.indent_level + text
    
    def push_indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1
    
    def pop_indent(self) -> None:
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)


class DSLVisitor(ABC):
    """Abstract base class for DSL generation visitors."""
    
    @abstractmethod
    def visit_dag(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Visit a computation DAG."""
        pass
    
    @abstractmethod
    def visit_node(self, node: ConductorNode, context: CodegenContext) -> list[str]:
        """Visit a computation node."""
        pass
    
    @abstractmethod
    def visit_buffer(self, buffer: Buffer, context: CodegenContext) -> list[str]:
        """Visit a buffer declaration."""
        pass


class ChoreoFunctionVisitor(DSLVisitor):
    """Visitor for generating Choreo __co__ functions."""
    
    def visit_dag(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate complete __co__ function from DAG."""
        lines = []
        
        # Generate function signature
        lines.extend(self._visit_function_signature(dag, context))
        
        # Generate function body
        context.push_indent()
        
        # Output buffer declarations
        lines.extend(self._visit_output_declarations(dag, context))
        
        # Parallel computation structure
        lines.extend(self._visit_parallel_structure(dag, context))
        
        # Return statement
        lines.extend(self._visit_return_statement(dag, context))
        
        context.pop_indent()
        lines.append("}")
        
        return lines
    
    def visit_node(self, node: ConductorNode, context: CodegenContext) -> list[str]:
        """Generate code for a single computation node."""
        # Check if we have a device kernel for this operation
        kernel = device_kernel_registry.get_kernel(node.op_name)
        if kernel:
            return self._visit_kernel_node(node, kernel, context)
        else:
            return self._visit_elementwise_node(node, context)
    
    def visit_buffer(self, buffer: Buffer, context: CodegenContext) -> list[str]:
        """Generate buffer declaration."""
        dtype = self._get_choreo_dtype(buffer.dtype)
        if buffer.shape:
            shape_str = ", ".join(map(str, buffer.shape))
            return [context.indent(f"{dtype} [{shape_str}] {buffer.name};")]
        else:
            return [context.indent(f"{dtype} {buffer.name};")]
    
    def _visit_function_signature(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate function signature using modular signature builder."""
        # Use modular signature generation for better maintainability
        signature_builder = create_signature_builder("choreo")
        if not signature_builder:
            # Fallback to manual generation
            return self._generate_signature_fallback(dag, context)

        function_name = context.current_function or "conductor_kernel"
        signature_info = signature_builder.build_from_dag(dag, function_name)
        signature_str = signature_builder.generate_signature_string(signature_info)

        return [f"{signature_str} {{"]

    def _generate_signature_fallback(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Fallback signature generation."""
        input_params = []
        for buf in dag.inputs:
            dtype = self._get_choreo_dtype(buf.dtype)
            if buf.shape:
                shape_str = ", ".join(map(str, buf.shape))
                input_params.append(f"{dtype} [{shape_str}] {buf.name}")
            else:
                input_params.append(f"{dtype} {buf.name}")

        function_name = context.current_function or "conductor_kernel"
        input_str = ", ".join(input_params)
        return [f"__co__ auto {function_name}({input_str}) {{"]
    
    def _visit_output_declarations(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate output buffer declarations."""
        lines = []
        for buf in dag.outputs:
            lines.extend(self.visit_buffer(buf, context))
        return lines
    
    def _visit_parallel_structure(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate optimized parallel computation structure with fusion."""
        lines = []

        # Analyze parallel fusion opportunities
        fusion_plan = analyze_parallel_fusion(dag)

        # Update context with optimized parameters
        context.parallel_factor = fusion_plan.parallel_factor
        context.buffer_size = fusion_plan.chunk_size

        # Generate strategy-specific parallel structure
        if fusion_plan.strategy == ParallelStrategy.PIPELINE_PARALLEL:
            lines.extend(self._visit_pipeline_parallel_structure(dag, fusion_plan, context))
        elif fusion_plan.strategy == ParallelStrategy.TASK_PARALLEL:
            lines.extend(self._visit_task_parallel_structure(dag, fusion_plan, context))
        elif fusion_plan.strategy == ParallelStrategy.HYBRID_PARALLEL:
            lines.extend(self._visit_hybrid_parallel_structure(dag, fusion_plan, context))
        else:  # DATA_PARALLEL (default)
            lines.extend(self._visit_data_parallel_structure(dag, fusion_plan, context))

        return lines

    def _visit_data_parallel_structure(self, dag: ComputationDAG, fusion_plan, context: CodegenContext) -> list[str]:
        """Generate data parallel structure (default)."""
        lines = []

        # Parallel directive with optimized factor
        lines.append(context.indent(f"parallel p by {fusion_plan.parallel_factor} {{"))
        context.push_indent()

        # Foreach loop for chunking with optimized chunk size
        lines.extend(self._visit_chunking_loop(dag, context, fusion_plan.chunk_size))

        context.pop_indent()
        lines.append(context.indent("}"))

        return lines

    def _visit_pipeline_parallel_structure(self, dag: ComputationDAG, fusion_plan, context: CodegenContext) -> list[str]:
        """Generate pipeline parallel structure."""
        lines = []

        if fusion_plan.pipeline_stages:
            # Generate pipeline stages
            lines.append(context.indent(f"parallel p by {fusion_plan.parallel_factor} {{"))
            context.push_indent()

            for stage_idx, stage_nodes in enumerate(fusion_plan.pipeline_stages):
                lines.append(context.indent(f"// Pipeline stage {stage_idx + 1}"))
                lines.append(context.indent(f"foreach index in [{fusion_plan.chunk_size}] {{"))
                context.push_indent()

                # Generate stage-specific computation
                lines.extend(self._visit_stage_computation(stage_nodes, context))

                context.pop_indent()
                lines.append(context.indent("}"))

                if stage_idx < len(fusion_plan.pipeline_stages) - 1:
                    lines.append(context.indent("// Pipeline synchronization"))
                    lines.append(context.indent("barrier;"))

            context.pop_indent()
            lines.append(context.indent("}"))
        else:
            # Fallback to data parallel
            lines.extend(self._visit_data_parallel_structure(dag, fusion_plan, context))

        return lines

    def _visit_task_parallel_structure(self, dag: ComputationDAG, fusion_plan, context: CodegenContext) -> list[str]:
        """Generate task parallel structure."""
        lines = []

        # Task parallel with cluster-based distribution
        lines.append(context.indent(f"parallel p by {fusion_plan.parallel_factor} {{"))
        context.push_indent()

        # Distribute clusters across parallel tasks
        for cluster_idx, cluster in enumerate(fusion_plan.clusters):
            lines.append(context.indent(f"if (p == {cluster_idx % fusion_plan.parallel_factor}) {{"))
            context.push_indent()

            lines.append(context.indent(f"foreach index in [{fusion_plan.chunk_size}] {{"))
            context.push_indent()

            # Generate cluster computation
            lines.extend(self._visit_cluster_computation(cluster, context))

            context.pop_indent()
            lines.append(context.indent("}"))

            context.pop_indent()
            lines.append(context.indent("}"))

        context.pop_indent()
        lines.append(context.indent("}"))

        return lines

    def _visit_hybrid_parallel_structure(self, dag: ComputationDAG, fusion_plan, context: CodegenContext) -> list[str]:
        """Generate hybrid parallel structure."""
        lines = []

        # Combine data and task parallelism
        lines.append(context.indent(f"parallel p by {fusion_plan.parallel_factor} {{"))
        context.push_indent()

        # Use different strategies for different cluster types
        for cluster in fusion_plan.clusters:
            if cluster.cluster_type.value == "compute_bound":
                # Use task parallelism for compute-bound clusters
                lines.append(context.indent(f"// Task parallel for {cluster.cluster_type.value}"))
                lines.extend(self._visit_cluster_computation(cluster, context))
            else:
                # Use data parallelism for other clusters
                lines.append(context.indent(f"// Data parallel for {cluster.cluster_type.value}"))
                lines.append(context.indent(f"foreach index in [{fusion_plan.chunk_size}] {{"))
                context.push_indent()
                lines.extend(self._visit_cluster_computation(cluster, context))
                context.pop_indent()
                lines.append(context.indent("}"))

        context.pop_indent()
        lines.append(context.indent("}"))

        return lines
    
    def _visit_chunking_loop(self, dag: ComputationDAG, context: CodegenContext, chunk_size: Optional[int] = None) -> list[str]:
        """Generate chunking loop structure."""
        lines = []

        # Use provided chunk size or determine from input shapes
        if chunk_size is None:
            chunk_size = self._determine_chunk_size(dag)

        lines.append(context.indent(f"foreach index in [{chunk_size}] {{"))
        context.push_indent()

        # DMA loads
        lines.extend(self._visit_dma_loads(dag, context))

        # Local computation
        lines.extend(self._visit_local_computation(dag, context))

        # DMA stores
        lines.extend(self._visit_dma_stores(dag, context))

        context.pop_indent()
        lines.append(context.indent("}"))

        return lines

    def _visit_stage_computation(self, stage_nodes: list[ConductorNode], context: CodegenContext) -> list[str]:
        """Generate computation for a pipeline stage."""
        lines = []

        # Generate computation for each node in the stage
        for node in stage_nodes:
            lines.extend(self.visit_node(node, context))

        return lines

    def _visit_cluster_computation(self, cluster, context: CodegenContext) -> List[str]:
        """Generate computation for a fusion cluster."""
        lines = []

        # Generate computation for each node in the cluster
        for node in cluster.nodes:
            lines.extend(self.visit_node(node, context))

        return lines
    
    def _visit_dma_loads(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate DMA load operations."""
        lines = []
        load_vars = []
        
        for i, buf in enumerate(dag.inputs):
            if i == 0:
                load_var = "lf"
            elif i == 1:
                load_var = "rf"
            else:
                load_var = f"input_{i}_load"
            
            load_vars.append(load_var)
            lines.append(context.indent(f"{load_var} = dma.copy.async {buf.name}.chunkat(p, index) => local;"))
        
        # Wait for all loads
        if load_vars:
            lines.append(context.indent(f"wait {', '.join(load_vars)};"))
            lines.append(context.indent(""))
        
        # Store load variables in context
        context.local_vars['input_vars'] = load_vars
        
        return lines
    
    def _visit_local_computation(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate local computation."""
        lines = []

        # Load input data first
        input_vars = context.local_vars.get('input_vars', ['lf', 'rf'])

        # Declare local output buffer
        if dag.outputs:
            output_buf = dag.outputs[0]
            dtype = self._get_choreo_dtype(output_buf.dtype)
            lines.append(context.indent(f"local {dtype} [{input_vars[0]}.span] l1_out;"))
            lines.append(context.indent(""))

        # Generate computation for each node
        for node in dag.nodes:
            lines.extend(self.visit_node(node, context))

        return lines
    
    def _visit_dma_stores(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate DMA store operations."""
        lines = []
        
        if dag.outputs:
            output_buf = dag.outputs[0]
            lines.append(context.indent(""))
            lines.append(context.indent(f"dma.copy l1_out => {output_buf.name}.chunkat(p, index);"))
        
        return lines
    
    def _visit_return_statement(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate return statement."""
        if dag.outputs:
            return [context.indent(f"return {dag.outputs[0].name};")]
        return []
    
    def _visit_kernel_node(self, node: ConductorNode, kernel, context: CodegenContext) -> list[str]:
        """Generate code for a node with device kernel."""
        input_vars = context.local_vars.get('input_vars', ['lf', 'rf'])
        output_vars = ['l1_out']
        
        kernel_call = kernel.get_kernel_call_syntax(node, input_vars, output_vars)
        return [context.indent(kernel_call)]
    
    def _visit_elementwise_node(self, node: ConductorNode, context: CodegenContext) -> list[str]:
        """Generate code for elementwise operations."""
        lines = []
        input_vars = context.local_vars.get('input_vars', ['lf', 'rf'])
        
        lines.append(context.indent("foreach i in [l1_out.span] {"))
        context.push_indent()
        
        # Generate operation-specific code using proper Choreo syntax
        if node.op_name == "add" and len(input_vars) >= 2:
            lines.append(context.indent(f"l1_out.at(i) = {input_vars[0]}.at(i) + {input_vars[1]}.at(i);"))
        elif node.op_name == "mul" and len(input_vars) >= 2:
            lines.append(context.indent(f"l1_out.at(i) = {input_vars[0]}.at(i) * {input_vars[1]}.at(i);"))
        elif node.op_name == "relu" and len(input_vars) >= 1:
            lines.append(context.indent(f"l1_out.at(i) = {input_vars[0]}.at(i) > 0.0f ? {input_vars[0]}.at(i) : 0.0f;"))
        else:
            # Generic fallback
            lines.append(context.indent(f"l1_out.at(i) = {input_vars[0]}.at(i); // {node.op_name}"))
        
        context.pop_indent()
        lines.append(context.indent("}"))
        
        return lines
    
    def _determine_chunk_size(self, dag: ComputationDAG) -> int:
        """Determine appropriate chunk size for computation."""
        if dag.inputs and dag.inputs[0].shape:
            shape = dag.inputs[0].shape
            if len(shape) >= 2:
                return min(shape[-1], 4)  # Use last dimension, capped at 4
            else:
                return min(shape[0], 16)
        return 4  # Default chunk size
    
    def _get_choreo_dtype(self, dtype: Optional[str]) -> str:
        """Convert PyTorch dtype to Choreo dtype."""
        if dtype is None:
            return "f32"
        
        dtype_map = {
            "float32": "f32",
            "float64": "f64", 
            "int32": "s32",
            "int64": "s64",
            "uint32": "u32",
            "uint64": "u64"
        }
        return dtype_map.get(dtype, "f32")


class ChoreoKernelVisitor(DSLVisitor):
    """Visitor for generating Choreo __cok__ kernel sections."""
    
    def visit_dag(self, dag: ComputationDAG, context: CodegenContext) -> list[str]:
        """Generate __cok__ section for device kernels."""
        kernel_nodes = [node for node in dag.nodes if device_kernel_registry.has_kernel(node.op_name)]
        
        if not kernel_nodes:
            return []
        
        lines = []
        lines.append("__cok__ {")
        
        for node in kernel_nodes:
            lines.extend(self.visit_node(node, context))
        
        lines.append("}")
        return lines
    
    def visit_node(self, node: ConductorNode, context: CodegenContext) -> list[str]:
        """Generate device kernel implementation for node."""
        kernel = device_kernel_registry.get_kernel(node.op_name)
        if kernel:
            lines = [f"// Device kernel for {node.op_name}"]
            lines.extend(kernel.generate_kernel_implementation(node))
            lines.append("")
            return lines
        return []
    
    def visit_buffer(self, buffer: Buffer, context: CodegenContext) -> list[str]:
        """Not used in kernel visitor."""
        return []


class DSLCodeGenerator:
    """Main DSL code generator using visitor pattern."""
    
    def __init__(self):
        self.function_visitor = ChoreoFunctionVisitor()
        self.kernel_visitor = ChoreoKernelVisitor()
    
    def generate_dsl(self, dag: ComputationDAG, function_name: str = "conductor_kernel") -> str:
        """Generate complete DSL using visitor pattern."""
        context = CodegenContext(current_function=function_name)
        
        lines = []
        
        # Header
        lines.extend(self._generate_header())
        
        # Kernel section
        kernel_lines = self.kernel_visitor.visit_dag(dag, context)
        if kernel_lines:
            lines.append("")
            lines.extend(kernel_lines)
        
        # Function section
        lines.append("")
        lines.extend(self.function_visitor.visit_dag(dag, context))
        
        return '\n'.join(lines)
    
    def _generate_header(self) -> list[str]:
        """Generate DSL header."""
        return [
            "// Generated Choreo DSL",
            "// Auto-generated from PyTorch FX Graph via Conductor",
            "",
            '#include "choreo.h"',
        ]


# Global DSL code generator instance
dsl_codegen = DSLCodeGenerator()
