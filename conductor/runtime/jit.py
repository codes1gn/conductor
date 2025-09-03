"""
Just-in-time compilation pipeline.

This module manages the dynamic compilation workflow from FX Graph
to executable artifacts, including caching and error handling.
"""

import subprocess
import tempfile
import os
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
import torch
from .loader import CompiledArtifact, ExecutableKernel
from ..codegen.graph import GraphAnalyzer
from ..codegen.fusion import FusionEngine
from ..codegen.choreo_dsl import ChoreoDSLGenerator
from ..utils.exceptions import CompilationError, UnsupportedOperationError, DeviceError, get_fallback_handler
from ..utils.logging import get_logger


class JITCompiler:
    """
    Manages just-in-time compilation workflow.
    
    This class orchestrates the complete JIT compilation pipeline,
    from FX Graph processing to executable artifact generation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_mb: int = 1024):
        """
        Initialize JIT compiler with required components.
        
        Args:
            cache_dir: Directory for compilation cache (default: system temp)
            max_cache_size_mb: Maximum cache size in megabytes
        """
        from ..utils.caching import CompilationCache
        
        self._cache = CompilationCache(cache_dir, max_cache_size_mb)
        self._graph_analyzer = GraphAnalyzer()
        self._fusion_engine = FusionEngine()
        self._dsl_generator = ChoreoDSLGenerator()
        self._logger = get_logger(__name__)
        
    def compile_graph(self, graph_module: torch.fx.GraphModule) -> CompiledArtifact:
        """
        Complete JIT compilation pipeline from FX Graph to executable.
        
        Args:
            graph_module: PyTorch FX Graph to compile
            
        Returns:
            CompiledArtifact ready for execution
            
        Raises:
            CompilationError: If compilation fails
            UnsupportedOperationError: If graph contains unsupported operations
        """
        self._logger.info(f"Starting JIT compilation for graph with {len(list(graph_module.graph.nodes))} nodes")
        
        try:
            # Generate graph signature for caching
            graph_hash = self._generate_graph_hash(graph_module)
            self._logger.debug(f"Graph hash: {graph_hash}")
            
            # Check cache first
            cached_artifact = self._cache.get(graph_hash)
            if cached_artifact is not None:
                self._logger.info("Found cached compilation result")
                # Validate cached artifact is still accessible
                if cached_artifact.is_valid():
                    return cached_artifact
                else:
                    self._logger.warning("Cached artifact is invalid, removing from cache")
                    self._cache.invalidate(graph_hash)
            
            # Step 1: Convert FX Graph to internal DAG
            self._logger.debug("Converting FX Graph to internal DAG")
            dag = self._graph_analyzer.parse_fx_graph(graph_module)
            
            # Validate graph correctness
            if not dag.validate_graph_correctness():
                raise CompilationError("Invalid graph structure detected", "", "")
            
            # Step 2: Apply fusion optimizations
            self._logger.debug("Applying fusion optimizations")
            fusion_clusters = self._fusion_engine.identify_fusion_opportunities(dag)
            self._logger.info(f"Identified {len(fusion_clusters)} fusion opportunities")
            
            # Apply fusion optimizations to the DAG
            for cluster in fusion_clusters:
                self._fusion_engine.optimize_buffer_usage(cluster)
                # Update nodes with fusion group information
                for node in cluster.nodes:
                    node.fusion_group = cluster
            
            # Step 3: Generate Choreo DSL
            self._logger.debug("Generating Choreo DSL")
            function_name = f"kernel_{graph_hash[:8]}"
            dsl_content = self._dsl_generator.generate_dsl_file(dag, function_name)

            # Write DSL to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
                dsl_file.write(dsl_content)
                dsl_file_path = dsl_file.name

            try:
                # Step 4: Invoke Choreo compiler
                self._logger.debug(f"Invoking Choreo compiler on {dsl_file_path}")
                artifact_path = self.invoke_choreo_compiler(dsl_file_path)
                
                # Step 5: Create compiled artifact
                artifact = CompiledArtifact(
                    path=artifact_path,
                    artifact_type='shared_library',
                    entry_point=function_name,
                    metadata={
                        'graph_hash': graph_hash,
                        'node_count': len(dag.nodes),
                        'fusion_clusters': len(fusion_clusters),
                        'dsl_content': dsl_content,
                        'function_name': function_name
                    }
                )
                
                # Cache the result
                if not self.cache_compilation_result(graph_hash, artifact):
                    self._logger.warning("Failed to cache compilation result")
                
                self._logger.info("JIT compilation completed successfully")
                return artifact
                
            finally:
                # Clean up temporary DSL file
                try:
                    os.unlink(dsl_file_path)
                except OSError:
                    pass
                    
        except UnsupportedOperationError as e:
            self._logger.warning(f"Unsupported operation encountered: {e}")
            # Generate diagnostic report
            diagnostics = self.get_diagnostic_info(e, graph_module)
            self._logger.debug(f"Diagnostic info: {diagnostics}")
            raise
        except CompilationError as e:
            self._logger.error(f"Compilation failed: {e}")
            # Generate comprehensive error report
            error_report = self.generate_error_report(e, graph_module)
            self._logger.debug(f"Error report:\n{error_report}")
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error during JIT compilation: {e}")
            # Wrap unexpected errors with diagnostic information
            compilation_error = CompilationError(f"JIT compilation failed: {e}", "", str(e))
            error_report = self.generate_error_report(compilation_error, graph_module)
            self._logger.debug(f"Error report:\n{error_report}")
            raise compilation_error
            
    def invoke_choreo_compiler(self, dsl_file: str) -> str:
        """
        Call external Choreo compiler to compile DSL to shared library.

        Args:
            dsl_file: Path to Choreo DSL file to compile

        Returns:
            Path to compiled shared library

        Raises:
            CompilationError: If compilation fails
        """
        self._logger.info(f"Invoking Choreo compiler for {dsl_file}")

        if not os.path.exists(dsl_file):
            raise CompilationError(f"DSL file not found: {dsl_file}", "", "")

        # Generate output paths
        base_name = os.path.splitext(dsl_file)[0]
        object_path = f"{base_name}.o"
        shared_lib_path = f"{base_name}.so"

        # Step 1: Compile to object file using choreo -c -fpic
        compile_cmd = [
            'choreo',
            '-c',           # Compile only, don't link
            '-fpic',        # Generate position-independent code for shared library
            dsl_file,
            '-o', object_path
        ]

        try:
            self._logger.debug(f"Running Choreo compiler: {' '.join(compile_cmd)}")

            # Step 1: Run Choreo compiler to generate object file
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                check=False
            )

            if result.returncode != 0:
                # Read DSL content for error context
                try:
                    with open(dsl_file, 'r') as f:
                        dsl_content = f.read()
                except:
                    dsl_content = "Unable to read DSL file"

                error_msg = f"Choreo compilation failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout}"

                raise CompilationError(
                    error_msg,
                    dsl_code=dsl_content,
                    compiler_output=result.stderr or result.stdout
                )

            # Verify object file was created
            if not os.path.exists(object_path) or os.path.getsize(object_path) == 0:
                raise CompilationError(
                    f"Choreo compiler succeeded but object file {object_path} was not created or is empty",
                    dsl_code="",
                    compiler_output=result.stdout
                )

            # Step 2: Link object file into shared library using gcc/clang
            link_cmd = [
                'gcc',  # or 'clang' depending on system
                '-shared',
                '-fPIC',
                object_path,
                '-o', shared_lib_path
            ]

            self._logger.debug(f"Linking shared library: {' '.join(link_cmd)}")
            link_result = subprocess.run(
                link_cmd,
                capture_output=True,
                text=True,
                timeout=30,  # Linking should be fast
                check=False
            )

            if link_result.returncode != 0:
                error_msg = f"Linking failed with return code {link_result.returncode}"
                if link_result.stderr:
                    error_msg += f"\nStderr: {link_result.stderr}"
                if link_result.stdout:
                    error_msg += f"\nStdout: {link_result.stdout}"

                raise CompilationError(
                    error_msg,
                    dsl_code="",
                    compiler_output=link_result.stderr or link_result.stdout
                )

            # Verify shared library was created
            if not os.path.exists(shared_lib_path) or os.path.getsize(shared_lib_path) == 0:
                raise CompilationError(
                    f"Linking succeeded but shared library {shared_lib_path} was not created or is empty",
                    dsl_code="",
                    compiler_output=link_result.stdout
                )

            # Clean up object file
            try:
                os.remove(object_path)
            except:
                pass  # Don't fail if cleanup fails

            self._logger.info(f"Choreo compilation and linking successful: {shared_lib_path}")
            return shared_lib_path

        except subprocess.TimeoutExpired:
            raise CompilationError(
                "Choreo compilation timed out after 300 seconds",
                dsl_code="",
                compiler_output=""
            )
        except FileNotFoundError:
            raise CompilationError(
                "Choreo compiler not found in PATH. Please ensure Choreo is installed and available.",
                dsl_code="",
                compiler_output=""
            )
        except Exception as e:
            if isinstance(e, CompilationError):
                raise
            raise CompilationError(f"Unexpected error during compilation: {e}", "", str(e))
            
    def load_compiled_artifact(self, artifact_path: str) -> ExecutableKernel:
        """
        Load compiled shared library using ctypes.
        
        Args:
            artifact_path: Path to compiled shared library
            
        Returns:
            ExecutableKernel ready for execution
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            self._logger.debug(f"Loading compiled artifact: {artifact_path}")
            kernel = ExecutableKernel.load_from_file(artifact_path)
            self._logger.debug("Artifact loaded successfully")
            return kernel
        except Exception as e:
            self._logger.error(f"Failed to load artifact {artifact_path}: {e}")
            raise RuntimeError(f"Failed to load compiled artifact: {e}")
        
    def cache_compilation_result(self, graph_hash: str, artifact: CompiledArtifact) -> bool:
        """
        Cache compiled artifacts for reuse.
        
        Args:
            graph_hash: Unique identifier for the graph
            artifact: Compiled artifact to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        success = self._cache.put(graph_hash, artifact)
        if success:
            self._logger.debug(f"Cached compilation result for graph {graph_hash}")
        else:
            self._logger.warning(f"Failed to cache compilation result for graph {graph_hash}")
        return success
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get compilation cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return self._cache.get_stats()
        
    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self._cache.clear()
        self._logger.info("Compilation cache cleared")
        
    def invalidate_cache_entry(self, graph_hash: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            graph_hash: Graph hash to invalidate
            
        Returns:
            True if entry was invalidated, False if not found
        """
        success = self._cache.invalidate(graph_hash)
        if success:
            self._logger.debug(f"Invalidated cache entry for graph {graph_hash}")
        return success
        
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """
        Validate cache integrity and return report.
        
        Returns:
            Dictionary containing validation results
        """
        stats = self.get_cache_stats()
        validation_results = {
            'total_entries': stats['entries'],
            'valid_entries': 0,
            'invalid_entries': 0,
            'missing_files': 0,
            'corrupted_entries': 0
        }
        
        # This would require access to cache internals for full validation
        # For now, just return basic stats
        validation_results['valid_entries'] = stats['entries']
        
        return validation_results
        
    def get_diagnostic_info(self, error: Exception, graph_module: Optional[torch.fx.GraphModule] = None) -> Dict[str, Any]:
        """
        Collect comprehensive diagnostic information for error analysis.
        
        Args:
            error: Exception that occurred
            graph_module: Optional FX Graph that caused the error
            
        Returns:
            Dictionary containing diagnostic information
        """
        diagnostics = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time(),
            'cache_stats': self.get_cache_stats(),
            'fallback_stats': get_fallback_handler().get_fallback_stats()
        }
        
        # Add error-specific diagnostics
        if isinstance(error, CompilationError):
            diagnostics.update(self._get_compilation_error_diagnostics(error))
        elif isinstance(error, UnsupportedOperationError):
            diagnostics.update(self._get_unsupported_operation_diagnostics(error))
        elif isinstance(error, DeviceError):
            diagnostics.update(self._get_device_error_diagnostics(error))
        
        # Add graph-specific diagnostics
        if graph_module is not None:
            diagnostics.update(self._get_graph_diagnostics(graph_module))
        
        # Add system information
        diagnostics.update(self._get_system_diagnostics())
        
        return diagnostics
        
    def _get_compilation_error_diagnostics(self, error: CompilationError) -> Dict[str, Any]:
        """Get diagnostics specific to compilation errors."""
        diagnostics = {
            'dsl_code_length': len(error.dsl_code) if error.dsl_code else 0,
            'compiler_output_length': len(error.compiler_output) if error.compiler_output else 0,
            'compiler_errors': error.get_compiler_errors(),
        }
        
        # Analyze DSL code if available
        if error.dsl_code:
            diagnostics.update(self._analyze_dsl_code(error.dsl_code))
        
        # Analyze compiler output if available
        if error.compiler_output:
            diagnostics.update(self._analyze_compiler_output(error.compiler_output))
        
        return diagnostics
        
    def _get_unsupported_operation_diagnostics(self, error: UnsupportedOperationError) -> Dict[str, Any]:
        """Get diagnostics specific to unsupported operation errors."""
        return {
            'unsupported_operation': error.operation,
            'unsupported_reason': error.reason,
            'suggested_alternatives': self._get_operation_alternatives(error.operation)
        }
        
    def _get_device_error_diagnostics(self, error: DeviceError) -> Dict[str, Any]:
        """Get diagnostics specific to device errors."""
        return {
            'device_id': error.device_id,
            'device_available': self._check_device_availability(),
            'memory_info': self._get_device_memory_info()
        }
        
    def _get_graph_diagnostics(self, graph_module: torch.fx.GraphModule) -> Dict[str, Any]:
        """Get diagnostics about the FX Graph."""
        nodes = list(graph_module.graph.nodes)
        
        # Count operation types
        op_counts = {}
        for node in nodes:
            op_type = node.op
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Identify complex operations
        complex_ops = []
        for node in nodes:
            if node.op == 'call_function':
                if hasattr(node.target, '__name__'):
                    op_name = node.target.__name__
                    if op_name in ['matmul', 'conv2d', 'linear', 'bmm']:
                        complex_ops.append(op_name)
        
        return {
            'graph_node_count': len(nodes),
            'graph_operation_counts': op_counts,
            'complex_operations': complex_ops,
            'graph_hash': self._generate_graph_hash(graph_module)
        }
        
    def _get_system_diagnostics(self) -> Dict[str, Any]:
        """Get system-level diagnostic information."""
        import platform
        import sys
        
        diagnostics = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'torch_version': torch.__version__ if hasattr(torch, '__version__') else 'unknown'
        }
        
        # Add memory information if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            diagnostics.update({
                'system_memory_total': memory.total,
                'system_memory_available': memory.available,
                'system_memory_percent': memory.percent
            })
        except ImportError:
            diagnostics['memory_info'] = 'psutil not available'
        
        return diagnostics
        
    def _analyze_dsl_code(self, dsl_code: str) -> Dict[str, Any]:
        """Analyze DSL code for common issues."""
        lines = dsl_code.split('\n')
        
        analysis = {
            'dsl_line_count': len(lines),
            'dsl_function_count': len([line for line in lines if 'function' in line]),
            'dsl_buffer_declarations': len([line for line in lines if any(scope in line for scope in ['local', 'shared', 'global'])]),
            'dsl_operations': len([line for line in lines if '=' in line and not line.strip().startswith('//')])
        }
        
        # Check for common DSL issues
        issues = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('//'):
                # Check for syntax issues
                if line.endswith('='):
                    issues.append(f"Line {i+1}: Incomplete assignment")
                if '(' in line and ')' not in line:
                    issues.append(f"Line {i+1}: Unmatched parentheses")
        
        analysis['dsl_syntax_issues'] = issues
        return analysis
        
    def _analyze_compiler_output(self, compiler_output: str) -> Dict[str, Any]:
        """Analyze compiler output for error patterns."""
        lines = compiler_output.split('\n')
        
        analysis = {
            'compiler_output_lines': len(lines),
            'error_lines': len([line for line in lines if 'error:' in line.lower()]),
            'warning_lines': len([line for line in lines if 'warning:' in line.lower()]),
        }
        
        # Extract specific error types
        error_types = set()
        for line in lines:
            if 'error:' in line.lower():
                # Simple pattern matching for common error types
                if 'syntax' in line.lower():
                    error_types.add('syntax_error')
                elif 'undefined' in line.lower():
                    error_types.add('undefined_symbol')
                elif 'type' in line.lower():
                    error_types.add('type_error')
                else:
                    error_types.add('unknown_error')
        
        analysis['compiler_error_types'] = list(error_types)
        return analysis
        
    def _get_operation_alternatives(self, operation: str) -> List[str]:
        """Get suggested alternatives for unsupported operations."""
        alternatives = {
            'custom_op': ['Use standard PyTorch operations', 'Implement as fusion of supported ops'],
            'complex_indexing': ['Use simpler indexing patterns', 'Break into multiple steps'],
            'dynamic_shapes': ['Use static shapes where possible', 'Add shape constraints'],
            'control_flow': ['Use torch.where for conditional logic', 'Avoid loops in traced code']
        }
        
        return alternatives.get(operation, ['Check documentation for supported operations'])
        
    def _check_device_availability(self) -> bool:
        """Check if GCU device is available."""
        # This would check actual device availability
        # For now, return a placeholder
        return True
        
    def _get_device_memory_info(self) -> Dict[str, Any]:
        """Get device memory information."""
        # This would query actual device memory
        # For now, return placeholder
        return {
            'total_memory': 'unknown',
            'free_memory': 'unknown',
            'used_memory': 'unknown'
        }
        
    def generate_error_report(self, error: Exception, graph_module: Optional[torch.fx.GraphModule] = None) -> str:
        """
        Generate a comprehensive error report for debugging.
        
        Args:
            error: Exception that occurred
            graph_module: Optional FX Graph that caused the error
            
        Returns:
            Formatted error report string
        """
        diagnostics = self.get_diagnostic_info(error, graph_module)
        
        report_lines = [
            "=" * 80,
            "CONDUCTOR COMPILATION ERROR REPORT",
            "=" * 80,
            f"Error Type: {diagnostics['error_type']}",
            f"Error Message: {diagnostics['error_message']}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diagnostics['timestamp']))}",
            "",
            "SYSTEM INFORMATION:",
            f"  Python Version: {diagnostics.get('python_version', 'unknown')}",
            f"  Platform: {diagnostics.get('platform', 'unknown')}",
            f"  PyTorch Version: {diagnostics.get('torch_version', 'unknown')}",
            ""
        ]
        
        # Add graph information if available
        if 'graph_node_count' in diagnostics:
            report_lines.extend([
                "GRAPH INFORMATION:",
                f"  Node Count: {diagnostics['graph_node_count']}",
                f"  Operation Counts: {diagnostics['graph_operation_counts']}",
                f"  Complex Operations: {diagnostics['complex_operations']}",
                f"  Graph Hash: {diagnostics['graph_hash']}",
                ""
            ])
        
        # Add error-specific information
        if isinstance(error, CompilationError):
            report_lines.extend([
                "COMPILATION ERROR DETAILS:",
                f"  DSL Code Length: {diagnostics.get('dsl_code_length', 0)} characters",
                f"  Compiler Output Length: {diagnostics.get('compiler_output_length', 0)} characters",
                f"  Compiler Errors: {len(diagnostics.get('compiler_errors', []))}",
            ])
            
            if diagnostics.get('compiler_errors'):
                report_lines.append("  Error Messages:")
                for err in diagnostics['compiler_errors'][:5]:  # Show first 5 errors
                    report_lines.append(f"    - {err}")
            
            if diagnostics.get('dsl_syntax_issues'):
                report_lines.append("  DSL Syntax Issues:")
                for issue in diagnostics['dsl_syntax_issues'][:5]:  # Show first 5 issues
                    report_lines.append(f"    - {issue}")
            
            report_lines.append("")
        
        elif isinstance(error, UnsupportedOperationError):
            report_lines.extend([
                "UNSUPPORTED OPERATION DETAILS:",
                f"  Operation: {diagnostics.get('unsupported_operation', 'unknown')}",
                f"  Reason: {diagnostics.get('unsupported_reason', 'not specified')}",
                "  Suggested Alternatives:",
            ])
            
            for alt in diagnostics.get('suggested_alternatives', []):
                report_lines.append(f"    - {alt}")
            
            report_lines.append("")
        
        # Add cache and fallback statistics
        cache_stats = diagnostics.get('cache_stats', {})
        fallback_stats = diagnostics.get('fallback_stats', {})
        
        report_lines.extend([
            "CACHE STATISTICS:",
            f"  Cache Entries: {cache_stats.get('entries', 0)}",
            f"  Cache Size: {cache_stats.get('total_size_mb', 0):.2f} MB",
            "",
            "FALLBACK STATISTICS:",
            f"  Total Fallbacks: {fallback_stats.get('total_fallbacks', 0)}",
            f"  Most Common Reason: {fallback_stats.get('most_common_reason', 'none')}",
            "",
            "SUGGESTED ACTIONS:",
        ])
        
        # Add suggested actions based on error type
        if isinstance(error, CompilationError):
            report_lines.extend([
                "  1. Check the generated DSL code for syntax errors",
                "  2. Verify that the Conductor compiler is properly installed",
                "  3. Try simplifying the model to isolate the issue",
                "  4. Enable debug logging for more detailed information"
            ])
        elif isinstance(error, UnsupportedOperationError):
            report_lines.extend([
                "  1. Use the suggested alternatives listed above",
                "  2. Check if the operation can be decomposed into supported operations",
                "  3. Consider using torch.compile with a different backend",
                "  4. File an issue if this operation should be supported"
            ])
        else:
            report_lines.extend([
                "  1. Check the error message for specific guidance",
                "  2. Verify system requirements and dependencies",
                "  3. Try with a simpler model to isolate the issue",
                "  4. Enable debug logging for more information"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
        
    def _generate_graph_hash(self, graph_module: torch.fx.GraphModule) -> str:
        """
        Generate unique hash for FX Graph for caching purposes.
        
        Args:
            graph_module: FX Graph to hash
            
        Returns:
            Unique hash string
        """
        # Create a comprehensive hash based on graph structure and content
        hash_components = []
        
        # Include graph structure
        graph_str = str(graph_module.graph)
        hash_components.append(graph_str)
        
        # Include node details for more precise hashing
        for node in graph_module.graph.nodes:
            node_info = f"{node.op}:{node.target}:{node.args}:{node.kwargs}"
            hash_components.append(node_info)
        
        # Include module parameters if any
        try:
            for name, param in graph_module.named_parameters():
                # Include parameter shape and dtype, but not values (for performance)
                param_info = f"{name}:{param.shape}:{param.dtype}"
                hash_components.append(param_info)
        except:
            # If parameter access fails, continue without them
            pass
        
        # Create final hash
        combined_str = "|".join(hash_components)
        return hashlib.sha256(combined_str.encode()).hexdigest()[:16]  # Use first 16 chars