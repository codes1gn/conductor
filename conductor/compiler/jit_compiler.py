"""
Choreo just-in-time compilation pipeline.

This module manages the dynamic compilation workflow from FX Graph
to executable artifacts using the real Choreo compiler.
"""

import subprocess
import tempfile
import os
import hashlib
from typing import Optional, Dict, Any, List
import torch
from .loader import CompiledArtifact, ExecutableKernel
from ..graph.graph_analyzer import GraphAnalyzer
from ..graph.fusion import FusionEngine
from ..codegen.dslgen import ChoreoDslGen
from ..utils.exceptions import CompilationError, UnsupportedOperationError
from ..config.logging import get_logger
from ..config.debug_artifacts import get_debug_manager
from ..config.debug_tracer import get_debug_tracer, trace_internal_dag, trace_choreo_dsl

logger = get_logger(__name__)


class ChoreoJITCompiler:
    """
    Manages just-in-time compilation workflow using Choreo compiler.
    
    This class orchestrates the complete JIT compilation pipeline,
    from FX Graph processing to executable Choreo artifacts.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_mb: int = 1024):
        """
        Initialize Choreo JIT compiler with required components.
        
        Args:
            cache_dir: Directory for compilation cache (default: system temp)
            max_cache_size_mb: Maximum cache size in megabytes
        """
        from ..config.caching import CompilationCache
        
        self._cache = CompilationCache(cache_dir, max_cache_size_mb)
        self._graph_analyzer = GraphAnalyzer()
        self._fusion_engine = FusionEngine()
        self._dsl_generator = ChoreoDslGen()
        self.compilation_timeout = 300  # 5 minutes
        
    def compile_graph(self, graph_module: torch.fx.GraphModule, example_inputs: Optional[List[torch.Tensor]] = None) -> CompiledArtifact:
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
        logger.info(f"Starting Choreo JIT compilation for graph with {len(list(graph_module.graph.nodes))} nodes")
        debug_tracer = get_debug_tracer()
        
        try:
            # Generate graph signature for caching
            graph_hash = self._generate_graph_hash(graph_module)
            logger.debug(f"Graph hash: {graph_hash}")
            
            # Check cache first (skip if debug mode)
            if not debug_tracer.is_enabled():
                cached_artifact = self._cache.get(graph_hash)
                if cached_artifact is not None:
                    logger.info("Found cached compilation result")
                    if cached_artifact.is_valid():
                        return cached_artifact
                    else:
                        logger.warning("Cached artifact is invalid, removing from cache")
                        self._cache.invalidate(graph_hash)
            
            # Step 1: Convert FX Graph to internal DAG with shape information
            logger.debug("Converting FX Graph to internal DAG")
            dag = self._graph_analyzer.parse_fx_graph(graph_module, example_inputs)

            # Debug tracing: Print internal DAG representation
            if debug_tracer.is_enabled():
                trace_internal_dag(dag)

            # Validate graph correctness
            if not dag.validate_graph_correctness():
                raise CompilationError("Invalid graph structure detected", "", "")
            
            # Step 2: Apply fusion optimizations
            logger.debug("Applying fusion optimizations")
            fusion_clusters = self._fusion_engine.identify_fusion_opportunities(dag)
            logger.info(f"Identified {len(fusion_clusters)} fusion opportunities")
            
            # Apply fusion optimizations to the DAG
            for cluster in fusion_clusters:
                self._fusion_engine.optimize_buffer_usage(cluster)
                for node in cluster.nodes:
                    node.fusion_group = cluster
            
            # Step 3: Generate Choreo DSL
            logger.debug("Generating Choreo DSL")
            function_name = f"kernel_{graph_hash[:8]}"
            dsl_content = self._dsl_generator.generate_dsl_file(dag, function_name)

            # Debug tracing: Print generated Choreo DSL code
            if debug_tracer.is_enabled():
                # Extract kernel code if present
                kernel_code = None
                if "__cok__" in dsl_content:
                    lines = dsl_content.split('\n')
                    in_kernel = False
                    kernel_lines = []
                    for line in lines:
                        if "__cok__" in line:
                            in_kernel = True
                        if in_kernel:
                            kernel_lines.append(line)
                        if in_kernel and line.strip() == "}":
                            break
                    kernel_code = '\n'.join(kernel_lines) if kernel_lines else None

                trace_choreo_dsl(dsl_content, kernel_code)

            # Save DSL source to debug directory
            debug_manager = get_debug_manager()
            debug_dsl_path = debug_manager.save_dsl_source(function_name, dsl_content)
            logger.info(f"Choreo DSL source saved to: {debug_dsl_path}")

            # Write DSL to temporary file for compilation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.co', delete=False) as dsl_file:
                dsl_file.write(dsl_content)
                dsl_file_path = dsl_file.name

            # Also save the actual compilation DSL file to debug directory for reference
            # This ensures we can see the exact file that was compiled, including any
            # potential differences from the debug version
            compilation_dsl_debug_path = debug_manager.save_compilation_dsl_source(function_name, dsl_file_path)
            
            try:
                # Step 4: Invoke Choreo compiler
                logger.debug(f"Invoking Choreo compiler on {dsl_file_path}")
                artifact_path = self.invoke_choreo_compiler(dsl_file_path, function_name)
                
                # Step 5: Create compiled artifact
                # Include object file path for host wrapper compilation
                base_name = os.path.splitext(dsl_file_path)[0]
                object_file_path = f"{base_name}.o"

                artifact = CompiledArtifact(
                    path=artifact_path,
                    artifact_type='shared_library',
                    entry_point=function_name,
                    metadata={
                        'graph_hash': graph_hash,
                        'node_count': len(dag.nodes),
                        'fusion_clusters': len(fusion_clusters),
                        'dsl_content': dsl_content,
                        'function_name': function_name,
                        'object_file_path': object_file_path,
                        'dsl_file_path': dsl_file_path,
                        'compilation_dsl_path': str(compilation_dsl_debug_path),
                        'debug_dsl_path': str(debug_dsl_path)
                    }
                )
                
                # Cache the result (skip if debug mode)
                if not debug_tracer.is_enabled():
                    if not self.cache_compilation_result(graph_hash, artifact):
                        logger.warning("Failed to cache compilation result")
                
                logger.info("Choreo JIT compilation completed successfully")
                return artifact
                
            finally:
                # Clean up temporary DSL file
                try:
                    os.unlink(dsl_file_path)
                except OSError:
                    pass
                    
        except UnsupportedOperationError as e:
            logger.warning(f"Unsupported operation encountered: {e}")
            raise
        except CompilationError as e:
            logger.error(f"Choreo compilation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Choreo JIT compilation: {e}")
            compilation_error = CompilationError(f"Choreo JIT compilation failed: {e}", "", str(e))
            raise compilation_error
    
    def invoke_choreo_compiler(self, dsl_file_path: str, kernel_name: str = "unknown") -> str:
        """
        Invoke the Choreo compiler to compile DSL to shared library.

        Args:
            dsl_file_path: Path to the Choreo DSL file to compile
            kernel_name: Name of the kernel for debug artifacts

        Returns:
            Path to the compiled shared library

        Raises:
            CompilationError: If compilation fails
        """
        logger.info(f"Invoking Choreo compiler for {dsl_file_path}")
        
        if not os.path.exists(dsl_file_path):
            raise CompilationError(f"DSL file not found: {dsl_file_path}", "", "")
        
        # Generate output paths
        base_name = os.path.splitext(dsl_file_path)[0]
        object_path = f"{base_name}.o"
        shared_lib_path = f"{base_name}.so"

        # Step 1: Compile to object file using choreo -c -fpic
        compile_cmd = [
            'choreo',
            '-c',           # Compile only, don't link
            '-fpic',        # Generate position-independent code for shared library
            dsl_file_path,
            '-o', object_path
        ]
        
        try:
            logger.debug(f"Running Choreo compiler: {' '.join(compile_cmd)}")

            # Step 1: Run Choreo compiler to generate object file
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=self.compilation_timeout,
                check=False
            )

            if result.returncode != 0:
                # Read DSL content for error context
                try:
                    with open(dsl_file_path, 'r') as f:
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

            # Save intermediate object file to debug directory
            debug_manager = get_debug_manager()
            debug_manager.save_intermediate_object(kernel_name, object_path)
            logger.info(f"Intermediate object file saved to debug directory: {kernel_name}.o")

            # Step 2: Link object file into shared library using gcc/clang
            # Note: We use regular gcc for linking, not topscc, because the object file
            # from choreo is already compiled and just needs to be linked into a shared library
            link_cmd = [
                'gcc',  # or 'clang' depending on system
                '-shared',
                '-fPIC',
                object_path,
                '-o', shared_lib_path
            ]

            logger.debug(f"Linking shared library: {' '.join(link_cmd)}")
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
                    compiler_output=""
                )

            # Keep object file for host wrapper compilation
            # Don't clean up object file - it's needed for GCU execution
            logger.debug(f"Preserving object file for host wrapper: {object_path}")

            logger.info(f"Choreo compilation and linking successful: {shared_lib_path}")
            return shared_lib_path
            
        except subprocess.TimeoutExpired:
            raise CompilationError(
                f"Choreo compilation timed out after {self.compilation_timeout} seconds",
                dsl_code="",
                compiler_output=""
            )
        except FileNotFoundError:
            raise CompilationError(
                "Choreo compiler not found in PATH. Please ensure Choreo is installed and available.",
                dsl_code="",
                compiler_output=""
            )
    
    def load_compiled_artifact(self, artifact_path: str) -> ExecutableKernel:
        """
        Load compiled Choreo artifact using ctypes.
        
        Args:
            artifact_path: Path to compiled shared library
            
        Returns:
            ExecutableKernel ready for execution
            
        Raises:
            RuntimeError: If artifact loading fails
        """
        logger.debug(f"Loading Choreo compiled artifact: {artifact_path}")
        
        try:
            return ExecutableKernel.load_from_file(artifact_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Choreo compiled artifact {artifact_path}: {e}")
    
    def cache_compilation_result(self, graph_hash: str, artifact: CompiledArtifact) -> bool:
        """
        Cache compilation result for future use.
        
        Args:
            graph_hash: Hash of the computation graph
            artifact: Compiled artifact to cache
            
        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            self._cache.put(graph_hash, artifact)
            logger.debug(f"Cached compilation result for graph {graph_hash}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache compilation result: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get compilation cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'entries': len(self._cache._metadata),
            'total_size_bytes': sum(
                entry.get('size_bytes', 0) 
                for entry in self._cache._metadata.values()
            ),
            'cache_dir': str(self._cache.cache_dir)
        }
    
    def clear_cache(self):
        """Clear compilation cache."""
        self._cache.clear()
        logger.info("Compilation cache cleared")
    
    def _generate_graph_hash(self, graph_module: torch.fx.GraphModule) -> str:
        """
        Generate hash for FX Graph for caching purposes.
        
        Args:
            graph_module: PyTorch FX Graph module
            
        Returns:
            16-character hash string
        """
        # Create a string representation of the graph structure
        graph_str = ""
        
        for node in graph_module.graph.nodes:
            # Include node operation and target information
            node_info = f"{node.op}:{node.target}:{node.name}"
            
            # Include input shapes if available
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                tensor_meta = node.meta['tensor_meta']
                if hasattr(tensor_meta, 'shape'):
                    node_info += f":{tensor_meta.shape}"
            
            graph_str += node_info + ";"
        
        # Generate hash
        hash_obj = hashlib.md5(graph_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]


# Alias for backward compatibility
JITCompiler = ChoreoJITCompiler
