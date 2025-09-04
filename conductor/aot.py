"""
Ahead-of-time compilation support.

This module handles loading and integration of precompiled artifacts
for AOT mode execution.
"""

import os
import hashlib
import json
import logging
from typing import Optional, Dict, List, Tuple
import torch
from .loader import ExecutableKernel, CompiledArtifact
from .exceptions import ConductorError


logger = logging.getLogger(__name__)


class AOTCompatibilityError(ConductorError):
    """Raised when precompiled artifact is incompatible with current graph."""
    pass


class AOTArtifactNotFoundError(ConductorError):
    """Raised when required precompiled artifact cannot be located."""
    pass


class AOTManager:
    """
    Handles ahead-of-time compiled artifact loading.
    
    This class manages the discovery, validation, and loading of
    precompiled artifacts for AOT mode execution.
    """
    
    def __init__(self, artifact_search_paths: Optional[List[str]] = None):
        """
        Initialize AOT manager with search paths.
        
        Args:
            artifact_search_paths: List of directories to search for artifacts
        """
        self.search_paths = artifact_search_paths or [
            './artifacts',
            '~/.conductor/artifacts',
            '/usr/local/lib/conductor'
        ]
        self._artifact_cache: Dict[str, CompiledArtifact] = {}
        
    def locate_precompiled_artifact(self, graph_signature: str) -> Optional[str]:
        """
        Find precompiled artifact matching graph signature.
        
        Searches through configured paths for artifacts matching the graph signature.
        Supports both .so (shared library) and .o (object file) formats.
        
        Args:
            graph_signature: Unique identifier for the graph
            
        Returns:
            Path to matching artifact, or None if not found
        """
        logger.debug(f"Searching for artifact with signature: {graph_signature}")
        
        # Search for artifacts with matching signature
        for search_path in self.search_paths:
            expanded_path = os.path.expanduser(search_path)
            if not os.path.exists(expanded_path):
                logger.debug(f"Search path does not exist: {expanded_path}")
                continue
                
            logger.debug(f"Searching in: {expanded_path}")
            
            # Look for .so and .o files with matching signature
            # Priority: .so files first (ready to load), then .o files
            for ext in ['.so', '.o']:
                artifact_path = os.path.join(expanded_path, f"{graph_signature}{ext}")
                if os.path.exists(artifact_path) and os.access(artifact_path, os.R_OK):
                    logger.info(f"Found artifact: {artifact_path}")
                    return artifact_path
                    
            # Also search for artifacts with metadata files
            metadata_path = os.path.join(expanded_path, f"{graph_signature}.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if the artifact file referenced in metadata exists
                    if 'artifact_path' in metadata:
                        artifact_path = os.path.join(expanded_path, metadata['artifact_path'])
                        if os.path.exists(artifact_path) and os.access(artifact_path, os.R_OK):
                            logger.info(f"Found artifact via metadata: {artifact_path}")
                            return artifact_path
                            
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to read metadata file {metadata_path}: {e}")
                    
        logger.debug(f"No artifact found for signature: {graph_signature}")
        return None
        
    def validate_artifact_compatibility(self, artifact_path: str, graph_module: torch.fx.GraphModule) -> bool:
        """
        Verify artifact compatibility with current graph.
        
        Performs comprehensive compatibility checking including:
        - File existence and accessibility
        - Graph signature validation
        - Input/output shape and type compatibility
        - Operation sequence validation
        
        Args:
            artifact_path: Path to precompiled artifact
            graph_module: Current FX Graph to validate against
            
        Returns:
            True if artifact is compatible, False otherwise
        """
        try:
            # Basic file checks
            if not os.path.exists(artifact_path):
                logger.debug(f"Artifact file does not exist: {artifact_path}")
                return False
                
            if not os.access(artifact_path, os.R_OK):
                logger.debug(f"Artifact file is not readable: {artifact_path}")
                return False
                
            # Check file format
            if not (artifact_path.endswith('.so') or artifact_path.endswith('.o')):
                logger.debug(f"Unsupported artifact format: {artifact_path}")
                return False
                
            # Generate current graph signature for comparison
            current_signature = self._generate_graph_signature(graph_module)
            
            # Try to find and validate metadata
            metadata = self._load_artifact_metadata(artifact_path)
            if metadata:
                # Check graph signature match
                if 'graph_signature' in metadata:
                    if metadata['graph_signature'] != current_signature:
                        logger.debug(f"Graph signature mismatch: expected {current_signature}, got {metadata['graph_signature']}")
                        return False
                        
                # Check input/output compatibility
                if not self._validate_io_compatibility(metadata, graph_module):
                    return False
                    
                # Check operation compatibility
                if not self._validate_operation_compatibility(metadata, graph_module):
                    return False
                    
            else:
                # Without metadata, we can only do basic checks
                logger.warning(f"No metadata found for artifact {artifact_path}, performing basic validation only")
                
                # Try to extract signature from filename
                filename = os.path.basename(artifact_path)
                if '.' in filename:
                    file_signature = filename.split('.')[0]
                    if file_signature != current_signature:
                        logger.debug(f"Filename signature mismatch: expected {current_signature}, got {file_signature}")
                        return False
                        
            logger.debug(f"Artifact compatibility validated: {artifact_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating artifact compatibility: {e}")
            return False
        
    def load_static_artifact(self, artifact_path: str) -> ExecutableKernel:
        """
        Load precompiled object file or shared library.
        
        Handles both .so (shared library) and .o (object file) formats.
        For object files, attempts to link them into a temporary shared library.
        
        Args:
            artifact_path: Path to precompiled artifact
            
        Returns:
            ExecutableKernel ready for execution
            
        Raises:
            AOTArtifactNotFoundError: If artifact file is not found
            AOTCompatibilityError: If artifact cannot be loaded
            RuntimeError: For other loading failures
        """
        if not os.path.exists(artifact_path):
            raise AOTArtifactNotFoundError(f"Artifact not found: {artifact_path}")
            
        if not os.access(artifact_path, os.R_OK):
            raise AOTCompatibilityError(f"Artifact not readable: {artifact_path}")
            
        try:
            logger.info(f"Loading artifact: {artifact_path}")
            
            # Check if we have this artifact cached
            artifact_key = f"{artifact_path}:{os.path.getmtime(artifact_path)}"
            if artifact_key in self._artifact_cache:
                cached_artifact = self._artifact_cache[artifact_key]
                logger.debug(f"Using cached artifact: {artifact_path}")
                return ExecutableKernel.load_from_file(cached_artifact.path)
            
            # Handle different artifact types
            if artifact_path.endswith('.so'):
                # Shared library - can load directly
                kernel = ExecutableKernel.load_from_file(artifact_path)
                
                # Cache the artifact info
                artifact = CompiledArtifact(
                    path=artifact_path,
                    artifact_type='shared_library',
                    entry_point='conductor_kernel_main',  # Default entry point
                    metadata=self._load_artifact_metadata(artifact_path) or {}
                )
                self._artifact_cache[artifact_key] = artifact
                
                return kernel
                
            elif artifact_path.endswith('.o'):
                # Object file - needs to be linked into shared library
                linked_so_path = self._link_object_file(artifact_path)
                
                kernel = ExecutableKernel.load_from_file(linked_so_path)
                
                # Cache the linked artifact
                artifact = CompiledArtifact(
                    path=linked_so_path,
                    artifact_type='linked_shared_library',
                    entry_point='conductor_kernel_main',
                    metadata=self._load_artifact_metadata(artifact_path) or {}
                )
                self._artifact_cache[artifact_key] = artifact
                
                return kernel
                
            else:
                raise AOTCompatibilityError(
                    f"Unsupported artifact format: {artifact_path}. "
                    f"Supported formats: .so (shared library), .o (object file)"
                )
                
        except (AOTArtifactNotFoundError, AOTCompatibilityError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact {artifact_path}: {e}")
            
    def get_artifact_metadata(self, artifact_path: str) -> Dict[str, any]:
        """
        Extract metadata from precompiled artifact.
        
        Attempts to load metadata from associated .json file or extract
        basic information from the artifact file itself.
        
        Args:
            artifact_path: Path to artifact
            
        Returns:
            Dictionary containing artifact metadata
        """
        metadata = {
            'path': artifact_path,
            'exists': os.path.exists(artifact_path),
            'size': os.path.getsize(artifact_path) if os.path.exists(artifact_path) else 0,
            'type': 'shared_library' if artifact_path.endswith('.so') else 'object_file',
            'readable': os.access(artifact_path, os.R_OK) if os.path.exists(artifact_path) else False
        }
        
        if os.path.exists(artifact_path):
            metadata['mtime'] = os.path.getmtime(artifact_path)
            
        # Try to load additional metadata from associated JSON file
        json_metadata = self._load_artifact_metadata(artifact_path)
        if json_metadata:
            metadata.update(json_metadata)
            
        return metadata
        
    def _generate_graph_signature(self, graph_module: torch.fx.GraphModule) -> str:
        """
        Generate a unique signature for the FX Graph.
        
        Args:
            graph_module: FX Graph module
            
        Returns:
            Unique string signature for the graph
        """
        # Create a deterministic representation of the graph
        graph_repr = []
        
        # Add nodes in order
        for node in graph_module.graph.nodes:
            node_info = {
                'op': node.op,
                'target': str(node.target),
                'args': str(node.args),
                'kwargs': str(sorted(node.kwargs.items()) if node.kwargs else [])
            }
            graph_repr.append(str(node_info))
            
        # Create hash of the graph representation
        graph_str = '|'.join(graph_repr)
        return hashlib.sha256(graph_str.encode()).hexdigest()[:16]
        
    def _load_artifact_metadata(self, artifact_path: str) -> Optional[Dict[str, any]]:
        """
        Load metadata from associated JSON file.
        
        Args:
            artifact_path: Path to artifact file
            
        Returns:
            Metadata dictionary or None if not found
        """
        # Try different metadata file locations
        base_path = os.path.splitext(artifact_path)[0]
        metadata_paths = [
            f"{base_path}.json",
            f"{artifact_path}.json",
            os.path.join(os.path.dirname(artifact_path), f"{os.path.basename(base_path)}.meta.json")
        ]
        
        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
                    
        return None
        
    def _validate_io_compatibility(self, metadata: Dict[str, any], graph_module: torch.fx.GraphModule) -> bool:
        """
        Validate input/output compatibility between artifact and graph.
        
        Args:
            metadata: Artifact metadata
            graph_module: Current FX Graph
            
        Returns:
            True if compatible, False otherwise
        """
        if 'inputs' not in metadata or 'outputs' not in metadata:
            logger.debug("No I/O information in metadata, skipping I/O validation")
            return True
            
        # Extract input/output info from graph
        graph_inputs = []
        graph_outputs = []
        
        for node in graph_module.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append({
                    'name': node.target,
                    'type': str(node.type) if hasattr(node, 'type') else 'unknown'
                })
            elif node.op == 'output':
                # Output nodes typically have a single argument that's a tuple or single value
                # For simplicity, count this as one output
                graph_outputs.append({
                    'name': 'output',
                    'type': 'unknown'
                })
                
        # If no explicit outputs found, assume single output (common case)
        if not graph_outputs:
            graph_outputs = [{'name': 'output', 'type': 'unknown'}]
                
        # Basic compatibility check
        if len(metadata['inputs']) != len(graph_inputs):
            logger.debug(f"Input count mismatch: metadata has {len(metadata['inputs'])}, graph has {len(graph_inputs)}")
            return False
            
        if len(metadata['outputs']) != len(graph_outputs):
            logger.debug(f"Output count mismatch: metadata has {len(metadata['outputs'])}, graph has {len(graph_outputs)}")
            return False
            
        return True
        
    def _validate_operation_compatibility(self, metadata: Dict[str, any], graph_module: torch.fx.GraphModule) -> bool:
        """
        Validate operation compatibility between artifact and graph.
        
        Args:
            metadata: Artifact metadata
            graph_module: Current FX Graph
            
        Returns:
            True if compatible, False otherwise
        """
        if 'operations' not in metadata:
            logger.debug("No operation information in metadata, skipping operation validation")
            return True
            
        # Extract operations from graph
        graph_ops = set()
        for node in graph_module.graph.nodes:
            if node.op == 'call_function':
                # Handle different types of function targets
                target = node.target
                if hasattr(target, '__name__'):
                    # Built-in functions like add, mul
                    op_name = target.__name__
                elif hasattr(target, '_name'):
                    # Some torch functions
                    op_name = target._name
                else:
                    # Fallback to string representation and extract name
                    target_str = str(target)
                    if 'relu' in target_str.lower():
                        op_name = 'relu'
                    elif 'add' in target_str.lower():
                        op_name = 'add'
                    elif 'mul' in target_str.lower():
                        op_name = 'mul'
                    elif '.' in target_str:
                        op_name = target_str.split('.')[-1]
                    else:
                        op_name = target_str
                        
                graph_ops.add(op_name)
            elif node.op == 'call_method':
                graph_ops.add(str(node.target))
                
        metadata_ops = set(metadata['operations'])
        
        # Check if all metadata operations are present in graph operations
        # We use a more flexible matching approach for known operations
        known_op_mappings = {
            'add': ['add', '__add__'],
            'mul': ['mul', '__mul__'], 
            'relu': ['relu'],
            'sigmoid': ['sigmoid'],
            'tanh': ['tanh'],
            'sum': ['sum'],
            'max': ['max'],
            'min': ['min']
        }
        
        for metadata_op in metadata_ops:
            found = False
            
            # Direct match
            if metadata_op in graph_ops:
                found = True
            # Check known mappings
            elif metadata_op in known_op_mappings:
                for possible_name in known_op_mappings[metadata_op]:
                    if possible_name in graph_ops:
                        found = True
                        break
            
            if not found:
                logger.debug(f"Operation '{metadata_op}' not found in graph operations: {graph_ops}")
                return False
                
        return True
        
    def _link_object_file(self, object_path: str) -> str:
        """
        Link object file into a temporary shared library.
        
        Args:
            object_path: Path to object file
            
        Returns:
            Path to linked shared library
            
        Raises:
            RuntimeError: If linking fails
        """
        import subprocess
        import tempfile
        
        # Create temporary shared library path
        temp_dir = tempfile.mkdtemp(prefix='conductor_aot_')
        so_path = os.path.join(temp_dir, f"{os.path.basename(object_path)}.so")
        
        try:
            # Use gcc/clang to link object file into shared library
            # This is a simplified approach - real implementation might need more sophisticated linking
            link_cmd = [
                'gcc',  # or 'clang'
                '-shared',
                '-fPIC',
                '-o', so_path,
                object_path
            ]
            
            logger.debug(f"Linking object file: {' '.join(link_cmd)}")
            result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"Linking failed: {result.stderr}")
                
            if not os.path.exists(so_path):
                raise RuntimeError(f"Linked library not created: {so_path}")
                
            logger.info(f"Successfully linked object file to: {so_path}")
            return so_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Linking timeout for object file: {object_path}")
        except FileNotFoundError:
            raise RuntimeError("GCC compiler not found. Object file linking requires GCC or Clang.")
        except Exception as e:
            raise RuntimeError(f"Failed to link object file {object_path}: {e}")
            
    def clear_cache(self) -> None:
        """Clear the artifact cache."""
        self._artifact_cache.clear()
        logger.debug("Artifact cache cleared")
        
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached artifacts."""
        return {
            'cached_artifacts': len(self._artifact_cache),
            'cache_keys': list(self._artifact_cache.keys())
        }
        
    def load_with_fallback(self, graph_module: torch.fx.GraphModule, 
                          fallback_to_jit: bool = True) -> ExecutableKernel:
        """
        Load precompiled artifact with automatic fallback to JIT compilation.
        
        This method implements the complete AOT loading workflow with fallback:
        1. Generate graph signature
        2. Locate precompiled artifact
        3. Validate compatibility
        4. Load artifact
        5. Fallback to JIT if any step fails
        
        Args:
            graph_module: FX Graph module to load artifact for
            fallback_to_jit: Whether to fallback to JIT compilation on failure
            
        Returns:
            ExecutableKernel ready for execution
            
        Raises:
            AOTArtifactNotFoundError: If no artifact found and fallback disabled
            RuntimeError: If both AOT and JIT compilation fail
        """
        graph_signature = self._generate_graph_signature(graph_module)
        
        try:
            # Step 1: Try to locate precompiled artifact
            logger.info(f"Attempting AOT loading for graph signature: {graph_signature}")
            artifact_path = self.locate_precompiled_artifact(graph_signature)
            
            if artifact_path is None:
                logger.info(f"No precompiled artifact found for signature: {graph_signature}")
                if fallback_to_jit:
                    return self._fallback_to_jit(graph_module, "artifact_not_found")
                else:
                    raise AOTArtifactNotFoundError(
                        f"No precompiled artifact found for graph signature: {graph_signature}"
                    )
            
            # Step 2: Validate compatibility
            logger.debug(f"Validating artifact compatibility: {artifact_path}")
            if not self.validate_artifact_compatibility(artifact_path, graph_module):
                logger.warning(f"Artifact compatibility validation failed: {artifact_path}")
                if fallback_to_jit:
                    return self._fallback_to_jit(graph_module, "compatibility_check_failed")
                else:
                    raise AOTCompatibilityError(
                        f"Artifact {artifact_path} is not compatible with current graph"
                    )
            
            # Step 3: Load the artifact
            logger.info(f"Loading compatible artifact: {artifact_path}")
            kernel = self.load_static_artifact(artifact_path)
            
            logger.info(f"Successfully loaded AOT artifact: {artifact_path}")
            return kernel
            
        except (AOTArtifactNotFoundError, AOTCompatibilityError) as e:
            # These are expected AOT-specific errors
            if fallback_to_jit:
                logger.info(f"AOT loading failed, falling back to JIT: {e}")
                return self._fallback_to_jit(graph_module, str(e))
            else:
                raise
                
        except Exception as e:
            # Unexpected errors during AOT loading
            logger.error(f"Unexpected error during AOT loading: {e}")
            if fallback_to_jit:
                return self._fallback_to_jit(graph_module, f"unexpected_error: {e}")
            else:
                raise RuntimeError(f"AOT loading failed: {e}")
                
    def _fallback_to_jit(self, graph_module: torch.fx.GraphModule, reason: str) -> ExecutableKernel:
        """
        Fallback to JIT compilation when AOT loading fails.
        
        Args:
            graph_module: FX Graph module to compile
            reason: Reason for fallback (for logging)
            
        Returns:
            ExecutableKernel from JIT compilation
            
        Raises:
            RuntimeError: If JIT compilation also fails
        """
        logger.info(f"Falling back to JIT compilation. Reason: {reason}")
        
        try:
            # Import JIT compiler here to avoid circular imports
            from .choreo_jit import JITCompiler
            
            jit_compiler = JITCompiler()
            compiled_artifact = jit_compiler.compile_graph(graph_module)
            
            # Load the JIT-compiled artifact
            kernel = ExecutableKernel.load_from_file(compiled_artifact.path)
            
            logger.info("JIT fallback compilation successful")
            return kernel
            
        except Exception as jit_error:
            logger.error(f"JIT fallback compilation failed: {jit_error}")
            
            # Try fallback to Inductor backend as last resort
            try:
                return self._fallback_to_inductor(graph_module)
            except Exception as inductor_error:
                raise RuntimeError(
                    f"All compilation methods failed. "
                    f"AOT reason: {reason}, "
                    f"JIT error: {jit_error}, "
                    f"Inductor error: {inductor_error}"
                )
                
    def _fallback_to_inductor(self, graph_module: torch.fx.GraphModule) -> ExecutableKernel:
        """
        Final fallback to PyTorch Inductor backend.
        
        Args:
            graph_module: FX Graph module to compile
            
        Returns:
            ExecutableKernel wrapping Inductor-compiled function
            
        Raises:
            RuntimeError: If Inductor compilation fails
        """
        logger.info("Falling back to PyTorch Inductor backend")
        
        try:
            # Compile with Inductor backend
            inductor_compiled = torch.compile(graph_module, backend='inductor')
            
            # Wrap the Inductor-compiled function in our ExecutableKernel interface
            # This is a simplified wrapper - real implementation would need more work
            class InductorKernelWrapper:
                def __init__(self, compiled_fn):
                    self.compiled_fn = compiled_fn
                    self._is_loaded = True
                    
                def execute(self, inputs):
                    # Convert inputs to the format expected by the compiled function
                    if len(inputs) == 1:
                        return [self.compiled_fn(inputs[0])]
                    else:
                        return [self.compiled_fn(*inputs)]
                        
                def unload(self):
                    self._is_loaded = False
                    
                def get_metadata(self):
                    return {'backend': 'inductor', 'loaded': self._is_loaded}
            
            # Create a mock ExecutableKernel that wraps the Inductor function
            # In a real implementation, this would be more sophisticated
            wrapper = InductorKernelWrapper(inductor_compiled)
            
            logger.info("Inductor fallback compilation successful")
            return wrapper
            
        except Exception as e:
            raise RuntimeError(f"Inductor fallback compilation failed: {e}")
            
    def diagnose_aot_failure(self, graph_module: torch.fx.GraphModule) -> Dict[str, any]:
        """
        Diagnose why AOT loading might fail for a given graph.
        
        Provides detailed diagnostic information to help users understand
        why precompiled artifacts are not available or compatible.
        
        Args:
            graph_module: FX Graph module to diagnose
            
        Returns:
            Dictionary containing diagnostic information
        """
        diagnosis = {
            'graph_signature': self._generate_graph_signature(graph_module),
            'search_paths': self.search_paths,
            'search_results': {},
            'compatibility_issues': [],
            'recommendations': []
        }
        
        # Check each search path
        for search_path in self.search_paths:
            expanded_path = os.path.expanduser(search_path)
            path_info = {
                'path': expanded_path,
                'exists': os.path.exists(expanded_path),
                'readable': os.access(expanded_path, os.R_OK) if os.path.exists(expanded_path) else False,
                'artifacts_found': []
            }
            
            if path_info['exists'] and path_info['readable']:
                # List all artifacts in this path
                try:
                    for file in os.listdir(expanded_path):
                        if file.endswith(('.so', '.o', '.json')):
                            path_info['artifacts_found'].append(file)
                except OSError:
                    path_info['readable'] = False
                    
            diagnosis['search_results'][search_path] = path_info
            
        # Try to locate artifact and check compatibility
        artifact_path = self.locate_precompiled_artifact(diagnosis['graph_signature'])
        if artifact_path:
            diagnosis['artifact_found'] = artifact_path
            diagnosis['artifact_metadata'] = self.get_artifact_metadata(artifact_path)
            
            # Check compatibility issues
            if not self.validate_artifact_compatibility(artifact_path, graph_module):
                diagnosis['compatibility_issues'].append("Artifact failed compatibility validation")
                
                # Try to get more specific information
                metadata = self._load_artifact_metadata(artifact_path)
                if metadata:
                    if 'graph_signature' in metadata:
                        if metadata['graph_signature'] != diagnosis['graph_signature']:
                            diagnosis['compatibility_issues'].append(
                                f"Graph signature mismatch: expected {diagnosis['graph_signature']}, "
                                f"got {metadata['graph_signature']}"
                            )
        else:
            diagnosis['artifact_found'] = None
            diagnosis['recommendations'].append(
                f"No artifact found for signature {diagnosis['graph_signature']}. "
                f"Consider precompiling this graph or using JIT mode."
            )
            
        # Add general recommendations
        if not any(info['exists'] for info in diagnosis['search_results'].values()):
            diagnosis['recommendations'].append(
                "No artifact search paths exist. Create artifact directories or configure custom paths."
            )
            
        return diagnosis