"""
Complete End-to-End GCU Compilation Pipeline

This module provides a comprehensive pipeline that takes PyTorch programs and makes them
runnable on GCU hardware through the Conductor framework with automatic debugging.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
import torch.fx as fx

from ..runtime.gcu_backend import GCUBackend
from ..runtime.choreo_jit import ChoreoJITCompiler
from ..utils.debug_artifacts import get_debug_manager
from ..utils.topscc_utils import get_topscc_environment

logger = logging.getLogger(__name__)


class GCUPipelineError(Exception):
    """Exception raised by GCU pipeline operations."""
    pass


class AutoDebugger:
    """Automatic debugging system for GCU pipeline issues."""

    def __init__(self):
        self.debug_manager = get_debug_manager()

    def analyze_choreo_error(self, error_msg: str, kernel_name: str) -> Dict[str, Any]:
        """
        Automatically analyze choreo runtime errors using debug artifacts.

        Args:
            error_msg: The error message from choreo runtime
            kernel_name: Name of the kernel that failed

        Returns:
            Analysis results with diagnosis and suggested fixes
        """
        analysis = {
            'error_type': 'unknown',
            'diagnosis': '',
            'suggested_fixes': [],
            'debug_info': {},
            'resolution_attempted': False,
            'resolution_success': False
        }

        # Get debug artifacts for analysis
        artifact_summary = self.debug_manager.get_artifact_summary(kernel_name)

        # Analyze different types of errors
        if "zero is detected" in error_msg and "mdspan" in error_msg:
            analysis['error_type'] = 'dimension_validation'
            analysis['diagnosis'] = 'Choreo kernel dimension validation failed - tensor dimensions do not match kernel expectations'
            analysis['suggested_fixes'] = [
                'Check input tensor shapes match choreo kernel dimension requirements',
                'Inspect choreo DSL source for expected dimensions',
                'Use choreo -es to examine auto-generated function signature',
                'Regenerate kernel with correct tensor dimensions'
            ]

        elif "undefined symbol" in error_msg.lower():
            analysis['error_type'] = 'linking_error'
            analysis['diagnosis'] = 'Host wrapper linking failed - missing symbols from GCU runtime or choreo object'
            analysis['suggested_fixes'] = [
                'Check GCU runtime library linking',
                'Verify choreo object file compilation',
                'Ensure proper host/device compilation separation',
                'Recompile with correct library paths'
            ]

        elif "compilation failed" in error_msg.lower():
            analysis['error_type'] = 'compilation_error'
            analysis['diagnosis'] = 'Host wrapper or choreo compilation failed'
            analysis['suggested_fixes'] = [
                'Check C++ syntax in generated host wrapper',
                'Verify choreo DSL syntax',
                'Use choreo -gs to examine compilation pipeline',
                'Regenerate host wrapper with corrected approach'
            ]

        elif "runtime check failed" in error_msg.lower():
            analysis['error_type'] = 'runtime_validation'
            analysis['diagnosis'] = 'Choreo runtime validation failed - input data or dimensions invalid'
            analysis['suggested_fixes'] = [
                'Validate input tensor data and shapes',
                'Check choreo kernel dimension requirements',
                'Inspect choreo DSL source for validation logic',
                'Use existing choreo-op examples as reference'
            ]

        # Add debug artifact information
        analysis['debug_info'] = {
            'artifacts_available': {k: v['exists'] for k, v in artifact_summary.items()},
            'kernel_name': kernel_name,
            'error_message': error_msg
        }

        return analysis

    def attempt_automatic_resolution(self, analysis: Dict[str, Any],
                                   kernel_name: str,
                                   input_shapes: List[Tuple[int, ...]]) -> bool:
        """
        Attempt automatic resolution of common issues.

        Args:
            analysis: Error analysis results
            kernel_name: Name of the kernel
            input_shapes: Input tensor shapes

        Returns:
            True if resolution was attempted and may have succeeded
        """
        analysis['resolution_attempted'] = True

        if analysis['error_type'] == 'dimension_validation':
            # Try to regenerate kernel with correct dimensions
            try:
                kernel_manager = ChoreoKernelManager()

                # Generate new kernel with correct dimensions
                new_dsl = kernel_manager.generate_elementwise_kernel(
                    'add', kernel_name, input_shapes, input_shapes[0]
                )

                # Save the corrected DSL
                dsl_path = self.debug_manager.debug_dir / f"{kernel_name}_corrected.co"
                with open(dsl_path, 'w') as f:
                    f.write(new_dsl)

                logger.info(f"Generated corrected kernel DSL: {dsl_path}")
                analysis['resolution_success'] = True
                return True

            except Exception as e:
                logger.warning(f"Failed to regenerate kernel: {e}")
                return False

        elif analysis['error_type'] == 'linking_error':
            # Try to recompile with different linking approach
            logger.info("Linking error detected - manual recompilation may be needed")
            return False

        return False
    
    def inspect_choreo_signature(self, kernel_name: str) -> Optional[str]:
        """Use choreo -es to inspect auto-generated function signature."""
        artifact_summary = self.debug_manager.get_artifact_summary(kernel_name)

        if not artifact_summary['dsl_source']['exists']:
            logger.warning(f"DSL source not available for {kernel_name}")
            return None

        dsl_path = artifact_summary['dsl_source']['path']

        try:
            result = subprocess.run(
                ['choreo', '-es', dsl_path],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                logger.info(f"Successfully inspected choreo signature for {kernel_name}")
                return result.stdout
            else:
                logger.warning(f"choreo -es failed for {kernel_name}: {result.stderr}")
                return None

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to run choreo -es for {kernel_name}: {e}")
            return None

    def inspect_choreo_pipeline(self, kernel_name: str) -> Optional[str]:
        """Use choreo -gs to examine complete compilation pipeline."""
        artifact_summary = self.debug_manager.get_artifact_summary(kernel_name)

        if not artifact_summary['dsl_source']['exists']:
            logger.warning(f"DSL source not available for {kernel_name}")
            return None

        dsl_path = artifact_summary['dsl_source']['path']
        # Use debug directory for pipeline script
        debug_manager = get_debug_manager()
        pipeline_script_path = debug_manager.debug_dir / f'choreo_pipeline_debug_{kernel_name}.sh'

        try:
            result = subprocess.run(
                ['choreo', '-gs', dsl_path, '-o', str(pipeline_script_path)],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # Read the generated pipeline script
                with open(str(pipeline_script_path), 'r') as f:
                    pipeline_content = f.read()
                logger.info(f"Successfully generated choreo pipeline script: {pipeline_script_path}")
                return pipeline_content
            else:
                logger.warning(f"choreo -gs failed for {kernel_name}: {result.stderr}")
                return None

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to run choreo -gs for {kernel_name}: {e}")
            return None

    def cross_reference_with_examples(self, operation: str) -> Optional[str]:
        """Cross-reference with choreo-op examples to understand proper patterns."""
        choreo_op_dir = Path(__file__).parent.parent.parent / "choreo-op"

        if not choreo_op_dir.exists():
            logger.warning("choreo-op directory not found")
            return None

        # Find relevant example files
        relevant_files = []
        for co_file in choreo_op_dir.glob("**/*.co"):
            if operation.lower() in co_file.name.lower() or 'elemwise' in co_file.name.lower():
                relevant_files.append(co_file)

        if not relevant_files:
            logger.warning(f"No relevant examples found for operation: {operation}")
            return None

        # Read the most relevant example
        example_file = relevant_files[0]
        try:
            with open(example_file, 'r') as f:
                content = f.read()
            logger.info(f"Found relevant example: {example_file}")
            return content
        except Exception as e:
            logger.warning(f"Failed to read example file {example_file}: {e}")
            return None

    def generate_debugging_report(self, kernel_name: str, error_msg: str) -> str:
        """Generate comprehensive debugging report."""
        report = f"""
=== GCU PIPELINE DEBUGGING REPORT ===
Kernel: {kernel_name}
Error: {error_msg}

=== ERROR ANALYSIS ===
"""

        # Perform error analysis
        analysis = self.analyze_choreo_error(error_msg, kernel_name)
        report += f"Error Type: {analysis['error_type']}\n"
        report += f"Diagnosis: {analysis['diagnosis']}\n"
        report += f"Suggested Fixes:\n"
        for fix in analysis['suggested_fixes']:
            report += f"  - {fix}\n"

        # Add choreo signature inspection
        signature = self.inspect_choreo_signature(kernel_name)
        if signature:
            report += f"\n=== CHOREO FUNCTION SIGNATURE ===\n{signature}\n"

        # Add debug artifacts info
        artifact_summary = self.debug_manager.get_artifact_summary(kernel_name)
        report += f"\n=== DEBUG ARTIFACTS ===\n"
        for artifact_type, info in artifact_summary.items():
            status = "✓" if info['exists'] else "✗"
            size_info = f" ({info['size']} bytes)" if info['exists'] else ""
            report += f"{status} {artifact_type}: {info['path']}{size_info}\n"

        # Add example cross-reference
        example_content = self.cross_reference_with_examples('elementwise')
        if example_content:
            report += f"\n=== RELEVANT EXAMPLE PATTERN ===\n"
            report += f"(First 20 lines of relevant example)\n"
            lines = example_content.split('\n')[:20]
            for i, line in enumerate(lines, 1):
                report += f"{i:2d}: {line}\n"

        return report


class ErrorResolutionStrategy:
    """Automated error diagnosis and resolution system."""

    def __init__(self):
        self.auto_debugger = AutoDebugger()
        self.kernel_manager = ChoreoKernelManager()

    def diagnose_and_resolve(self, error_msg: str, kernel_name: str,
                           model: torch.nn.Module,
                           example_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Comprehensive error diagnosis and resolution.

        Args:
            error_msg: Error message from execution
            kernel_name: Name of the failed kernel
            model: Original PyTorch model
            example_inputs: Example input tensors

        Returns:
            Resolution results with status and actions taken
        """
        resolution_result = {
            'diagnosis': {},
            'resolution_attempted': False,
            'resolution_actions': [],
            'success': False,
            'new_kernel_generated': False,
            'debugging_report': ''
        }

        # Step 1: Analyze the error
        analysis = self.auto_debugger.analyze_choreo_error(error_msg, kernel_name)
        resolution_result['diagnosis'] = analysis

        # Step 2: Generate debugging report
        debugging_report = self.auto_debugger.generate_debugging_report(kernel_name, error_msg)
        resolution_result['debugging_report'] = debugging_report

        # Step 3: Attempt resolution based on error type
        if analysis['error_type'] == 'dimension_validation':
            resolution_result.update(self._resolve_dimension_error(
                kernel_name, example_inputs, analysis
            ))

        elif analysis['error_type'] == 'compilation_error':
            resolution_result.update(self._resolve_compilation_error(
                kernel_name, model, example_inputs, analysis
            ))

        elif analysis['error_type'] == 'linking_error':
            resolution_result.update(self._resolve_linking_error(
                kernel_name, analysis
            ))

        elif analysis['error_type'] == 'runtime_validation':
            resolution_result.update(self._resolve_runtime_error(
                kernel_name, example_inputs, analysis
            ))

        return resolution_result

    def _resolve_dimension_error(self, kernel_name: str,
                               example_inputs: List[torch.Tensor],
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dimension validation errors."""
        result = {
            'resolution_attempted': True,
            'resolution_actions': [],
            'success': False,
            'new_kernel_generated': False
        }

        try:
            # Extract input shapes from example inputs
            input_shapes = [tuple(tensor.shape) for tensor in example_inputs]

            # Generate new kernel with correct dimensions
            new_dsl = self.kernel_manager.generate_elementwise_kernel(
                'add', f"{kernel_name}_corrected", input_shapes, input_shapes[0]
            )

            # Save corrected DSL
            debug_manager = self.auto_debugger.debug_manager
            corrected_path = debug_manager.debug_dir / f"{kernel_name}_corrected.co"

            with open(corrected_path, 'w') as f:
                f.write(new_dsl)

            result['resolution_actions'].append(f"Generated corrected kernel DSL: {corrected_path}")
            result['new_kernel_generated'] = True

            # Inspect the corrected kernel
            signature = self.auto_debugger.inspect_choreo_signature(f"{kernel_name}_corrected")
            if signature:
                result['resolution_actions'].append("Inspected corrected kernel signature")

            result['success'] = True

        except Exception as e:
            result['resolution_actions'].append(f"Failed to generate corrected kernel: {e}")

        return result

    def _resolve_compilation_error(self, kernel_name: str,
                                 model: torch.nn.Module,
                                 example_inputs: List[torch.Tensor],
                                 analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve compilation errors."""
        result = {
            'resolution_attempted': True,
            'resolution_actions': [],
            'success': False,
            'new_kernel_generated': False
        }

        try:
            # Check if it's a host wrapper compilation issue
            if "host wrapper" in analysis['debug_info'].get('error_message', '').lower():
                result['resolution_actions'].append("Detected host wrapper compilation issue")

                # Generate pipeline inspection
                pipeline_info = self.auto_debugger.inspect_choreo_pipeline(kernel_name)
                if pipeline_info:
                    result['resolution_actions'].append("Generated choreo pipeline inspection")

            # Try to regenerate with different approach
            input_shapes = [tuple(tensor.shape) for tensor in example_inputs]
            new_dsl = self.kernel_manager.generate_elementwise_kernel(
                'add', f"{kernel_name}_recompiled", input_shapes, input_shapes[0]
            )

            debug_manager = self.auto_debugger.debug_manager
            recompiled_path = debug_manager.debug_dir / f"{kernel_name}_recompiled.co"

            with open(recompiled_path, 'w') as f:
                f.write(new_dsl)

            result['resolution_actions'].append(f"Generated recompiled kernel: {recompiled_path}")
            result['new_kernel_generated'] = True
            result['success'] = True

        except Exception as e:
            result['resolution_actions'].append(f"Failed to resolve compilation error: {e}")

        return result

    def _resolve_linking_error(self, kernel_name: str,
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve linking errors."""
        result = {
            'resolution_attempted': True,
            'resolution_actions': [],
            'success': False
        }

        # Check GCU runtime availability
        topscc_env = get_topscc_environment()
        if topscc_env.is_available():
            result['resolution_actions'].append("GCU runtime environment is available")
        else:
            result['resolution_actions'].append("WARNING: GCU runtime environment not available")

        # Generate pipeline inspection for linking analysis
        pipeline_info = self.auto_debugger.inspect_choreo_pipeline(kernel_name)
        if pipeline_info:
            result['resolution_actions'].append("Generated pipeline inspection for linking analysis")
            result['success'] = True

        return result

    def _resolve_runtime_error(self, kernel_name: str,
                             example_inputs: List[torch.Tensor],
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve runtime validation errors."""
        result = {
            'resolution_attempted': True,
            'resolution_actions': [],
            'success': False,
            'new_kernel_generated': False
        }

        try:
            # Validate input tensor properties
            for i, tensor in enumerate(example_inputs):
                result['resolution_actions'].append(
                    f"Input {i}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
                )

            # Cross-reference with working examples
            example_content = self.auto_debugger.cross_reference_with_examples('elementwise')
            if example_content:
                result['resolution_actions'].append("Found relevant working example for comparison")

            # Generate corrected kernel based on input analysis
            input_shapes = [tuple(tensor.shape) for tensor in example_inputs]
            new_dsl = self.kernel_manager.generate_elementwise_kernel(
                'add', f"{kernel_name}_runtime_corrected", input_shapes, input_shapes[0]
            )

            debug_manager = self.auto_debugger.debug_manager
            corrected_path = debug_manager.debug_dir / f"{kernel_name}_runtime_corrected.co"

            with open(corrected_path, 'w') as f:
                f.write(new_dsl)

            result['resolution_actions'].append(f"Generated runtime-corrected kernel: {corrected_path}")
            result['new_kernel_generated'] = True
            result['success'] = True

        except Exception as e:
            result['resolution_actions'].append(f"Failed to resolve runtime error: {e}")

        return result


class ChoreoKernelManager:
    """Manages Choreo kernel implementations and DSL generation."""
    
    def __init__(self):
        self.choreo_op_dir = Path(__file__).parent.parent.parent / "choreo-op"
        self.debug_manager = get_debug_manager()
    
    def get_existing_kernel(self, operation: str) -> Optional[Path]:
        """
        Find existing .co file for the given operation.
        
        Args:
            operation: Operation name (e.g., 'add', 'mul', 'elementwise')
            
        Returns:
            Path to existing .co file or None
        """
        # Check choreo-op directory for existing implementations
        if self.choreo_op_dir.exists():
            for co_file in self.choreo_op_dir.glob("**/*.co"):
                if operation.lower() in co_file.name.lower():
                    return co_file
        
        return None
    
    def generate_elementwise_kernel(self, operation: str, kernel_name: str,
                                  input_shapes: List[Tuple[int, ...]],
                                  output_shape: Tuple[int, ...]) -> str:
        """
        Generate Choreo DSL source for elementwise operations based on existing patterns.

        Args:
            operation: Operation type ('add', 'mul', etc.)
            kernel_name: Name for the generated kernel
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape

        Returns:
            Generated Choreo DSL source code following proper Choreo patterns
        """
        # Use the first input shape as reference (assuming all inputs have same shape for elementwise)
        if not input_shapes:
            raise ValueError("At least one input shape required")

        ref_shape = input_shapes[0]
        rank = len(ref_shape)
        shape_str = ", ".join(map(str, ref_shape))

        # Generate kernel operation
        if operation == 'add':
            kernel_body = "c[i] = a[i] + b[i];"
        elif operation == 'mul':
            kernel_body = "c[i] = a[i] * b[i];"
        elif operation == 'sub':
            kernel_body = "c[i] = a[i] - b[i];"
        else:
            kernel_body = f"c[i] = a[i]; // {operation} not implemented"

        # Calculate tiling factors for parallel execution
        # Use simple tiling strategy based on the existing pattern
        if rank >= 3:
            parallel_dim = ref_shape[0]
            foreach_dims = ref_shape[1:]
            tiling_factor = min(4, foreach_dims[-1]) if foreach_dims else 1
        elif rank == 2:
            parallel_dim = ref_shape[0]
            foreach_dims = [ref_shape[1]]
            tiling_factor = min(4, ref_shape[1])
        else:
            parallel_dim = ref_shape[0] if ref_shape else 1
            foreach_dims = []
            tiling_factor = 1

        # Generate proper Choreo DSL following the existing pattern
        dsl_source = f'''// Auto-generated Choreo DSL for {operation} operation
// Kernel: {kernel_name}
// Input shapes: {input_shapes}
// Output shape: {output_shape}

__cok__ {{ /// kernel program
__co_device__ extern "C" void kernel(int * a, int * b, int * c, int n) {{
  for (int i = 0; i < n; ++i)
    {kernel_body}
}}
}} /// end of kernel decl

__co__ s32 [{shape_str}] {kernel_name}(s32 [{shape_str}] lhs, s32 [{shape_str}] rhs) {{ /// device program
  s32[lhs.span] output; // use same shape as lhs

  parallel p by {parallel_dim} {{  // p is sip_index'''

        # Add foreach loops based on dimensions
        if rank >= 2:
            dsl_source += f'''
    foreach x in {foreach_dims[0]}'''
            if rank >= 3:
                dsl_source += f'''
      foreach y in {tiling_factor} {{'''
            else:
                dsl_source += f''' {{'''
        else:
            dsl_source += f''' {{'''

        # Add DMA operations and kernel call
        if rank >= 3:
            dsl_source += f'''
        lhs_load = dma.copy.async lhs.chunkat(p, x, y) => local;
        rhs_load = dma.copy.async rhs.chunkat(p, x, y) => local;
        wait lhs_load, rhs_load;

        local s32[lhs_load.span] l1_out;

        call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);

        out_store = dma.copy.async l1_out => output.chunkat(p, x, y);
        wait out_store;
      }}'''
        elif rank >= 2:
            dsl_source += f'''
        lhs_load = dma.copy.async lhs.chunkat(p, x) => local;
        rhs_load = dma.copy.async rhs.chunkat(p, x) => local;
        wait lhs_load, rhs_load;

        local s32[lhs_load.span] l1_out;

        call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);

        out_store = dma.copy.async l1_out => output.chunkat(p, x);
        wait out_store;
      }}'''
        else:
            dsl_source += f'''
        lhs_load = dma.copy.async lhs.chunkat(p) => local;
        rhs_load = dma.copy.async rhs.chunkat(p) => local;
        wait lhs_load, rhs_load;

        local s32[lhs_load.span] l1_out;

        call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);

        out_store = dma.copy.async l1_out => output.chunkat(p);
        wait out_store;
      }}'''

        dsl_source += f'''
  }}
  return output;
}}

int main() {{ /// host program
  auto a = choreo::make_spandata<choreo::s32>({shape_str});
  auto b = choreo::make_spandata<choreo::s32>({shape_str});
  a.fill_random(-10, 10);
  b.fill_random(-10, 10);

  auto res = {kernel_name}(a.view(), b.view());

  // Verification
  for (size_t i = 0; i < res.element_count(); ++i) {{
    if (a.data()[i] + b.data()[i] != res.data()[i]) {{
      choreo::choreo_assert(false, "values are not equal.");
    }}
  }}

  std::cout << "Test Passed" << std::endl;
  return 0;
}}'''

        return dsl_source.strip()


class GCUPipeline:
    """Complete end-to-end GCU compilation pipeline."""
    
    def __init__(self):
        self.backend = GCUBackend()
        self.kernel_manager = ChoreoKernelManager()
        self.auto_debugger = AutoDebugger()
        self.debug_manager = get_debug_manager()
        
    def compile_pytorch_model(self, model: torch.nn.Module, 
                            example_inputs: List[torch.Tensor],
                            enable_debugging: bool = True) -> 'CompiledGCUModel':
        """
        Compile a PyTorch model for GCU execution.
        
        Args:
            model: PyTorch model to compile
            example_inputs: Example input tensors for tracing
            enable_debugging: Whether to enable automatic debugging
            
        Returns:
            Compiled model ready for GCU execution
        """
        logger.info(f"Starting GCU compilation pipeline for model: {type(model).__name__}")
        
        try:
            # Step 1: Trace the model
            traced_model = fx.symbolic_trace(model)
            logger.info(f"Successfully traced model with {len(list(traced_model.graph.nodes))} nodes")
            
            # Step 2: Compile for GCU
            compiled_func = self.backend.compile_graph(traced_model, example_inputs)
            kernel_name = compiled_func.artifact.entry_point
            
            logger.info(f"Successfully compiled model for GCU: {kernel_name}")
            
            # Step 3: Create compiled model wrapper
            compiled_model = CompiledGCUModel(
                compiled_func=compiled_func,
                kernel_name=kernel_name,
                auto_debugger=self.auto_debugger if enable_debugging else None,
                debug_manager=self.debug_manager
            )
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"GCU compilation pipeline failed: {e}")
            
            # Automatic error analysis if debugging is enabled
            if enable_debugging:
                try:
                    # Try to get kernel name from error context
                    kernel_name = getattr(e, 'kernel_name', 'unknown')
                    analysis = self.auto_debugger.analyze_choreo_error(str(e), kernel_name)
                    
                    logger.error(f"Error analysis: {analysis['diagnosis']}")
                    for fix in analysis['suggested_fixes']:
                        logger.error(f"  Suggested fix: {fix}")
                        
                except Exception as debug_error:
                    logger.error(f"Failed to analyze error: {debug_error}")
            
            raise GCUPipelineError(f"GCU compilation failed: {e}") from e


class CompiledGCUModel:
    """Wrapper for compiled GCU model with automatic debugging."""

    def __init__(self, compiled_func, kernel_name: str,
                 auto_debugger: Optional[AutoDebugger] = None,
                 debug_manager = None,
                 input_shapes: Optional[List[Tuple[int, ...]]] = None):
        self.compiled_func = compiled_func
        self.kernel_name = kernel_name
        self.auto_debugger = auto_debugger
        self.debug_manager = debug_manager
        self.input_shapes = input_shapes or []

    def __call__(self, *args, **kwargs):
        """Execute the compiled model with automatic error handling and resolution."""
        try:
            return self.compiled_func(*args, **kwargs)

        except Exception as e:
            logger.error(f"GCU model execution failed: {e}")

            # Automatic debugging if enabled
            if self.auto_debugger:
                analysis = self.auto_debugger.analyze_choreo_error(str(e), self.kernel_name)

                logger.error(f"Automatic error analysis:")
                logger.error(f"  Error type: {analysis['error_type']}")
                logger.error(f"  Diagnosis: {analysis['diagnosis']}")

                for fix in analysis['suggested_fixes']:
                    logger.error(f"  Suggested fix: {fix}")

                # Attempt automatic resolution
                if analysis['error_type'] in ['dimension_validation', 'runtime_validation']:
                    logger.info("Attempting automatic error resolution...")

                    resolution_success = self.auto_debugger.attempt_automatic_resolution(
                        analysis, self.kernel_name, self.input_shapes
                    )

                    if resolution_success:
                        logger.info("Automatic resolution may have succeeded - manual recompilation needed")
                    else:
                        logger.warning("Automatic resolution failed")

                # Provide additional debugging information
                if analysis['error_type'] == 'dimension_validation':
                    signature = self.auto_debugger.inspect_choreo_signature(self.kernel_name)
                    if signature:
                        logger.error(f"Choreo function signature:\n{signature}")

                elif analysis['error_type'] in ['compilation_error', 'linking_error']:
                    pipeline = self.auto_debugger.inspect_choreo_pipeline(self.kernel_name)
                    if pipeline:
                        logger.error(f"Choreo compilation pipeline available for inspection")

            raise

    def get_debug_artifacts(self) -> Dict[str, Any]:
        """Get debug artifacts for manual inspection."""
        if self.debug_manager:
            return self.debug_manager.get_artifact_summary(self.kernel_name)
        return {}

    def get_error_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Get detailed error analysis for debugging."""
        if self.auto_debugger:
            return self.auto_debugger.analyze_choreo_error(error_msg, self.kernel_name)
        return {}


# High-level PyTorch integration functions
def compile_for_gcu(model: torch.nn.Module,
                   example_inputs: List[torch.Tensor],
                   enable_debugging: bool = True) -> CompiledGCUModel:
    """
    High-level function to compile PyTorch model for GCU execution.

    Args:
        model: PyTorch model to compile
        example_inputs: Example input tensors for tracing
        enable_debugging: Whether to enable automatic debugging

    Returns:
        Compiled model ready for GCU execution
    """
    pipeline = GCUPipeline()
    return pipeline.compile_pytorch_model(model, example_inputs, enable_debugging)


def test_gcu_pipeline(model: torch.nn.Module,
                     example_inputs: List[torch.Tensor],
                     enable_debugging: bool = True) -> Tuple[bool, str]:
    """
    Test the complete GCU pipeline with a PyTorch model.

    Args:
        model: PyTorch model to test
        example_inputs: Example input tensors
        enable_debugging: Whether to enable debugging

    Returns:
        Tuple of (success, message)
    """
    try:
        # Get PyTorch reference
        with torch.no_grad():
            reference = model(*example_inputs)

        # Compile for GCU
        compiled_model = compile_for_gcu(model, example_inputs, enable_debugging)

        # Test execution
        result = compiled_model(*example_inputs)

        # Compare results
        if isinstance(result, (list, tuple)):
            result = result[0]

        if isinstance(reference, (list, tuple)):
            reference = reference[0]

        max_diff = torch.max(torch.abs(reference - result)).item()

        if max_diff < 1e-5:
            return True, f"Pipeline test successful - results match perfectly (max_diff: {max_diff:.2e})"
        elif max_diff < 1e-2:
            return True, f"Pipeline test successful - results match within tolerance (max_diff: {max_diff:.2e})"
        else:
            return False, f"Pipeline test failed - results differ significantly (max_diff: {max_diff:.2e})"

    except Exception as e:
        return False, f"Pipeline test failed with error: {e}"
