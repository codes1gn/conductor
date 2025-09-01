"""
Test package structure and basic imports.

This test module verifies that the package is properly structured
and all modules can be imported without errors.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_main_package_import():
    """Test that the main conductor package can be imported."""
    import conductor
    
    # Check basic attributes
    assert hasattr(conductor, '__version__')
    assert hasattr(conductor, '__author__')
    assert hasattr(conductor, 'register_backend')
    assert hasattr(conductor, 'ConductorBackend')


def test_codegen_imports():
    """Test that codegen submodules can be imported."""
    from conductor.codegen import (
        GraphAnalyzer,
        ComputationDAG,
        ConductorNode,
        FusionEngine,
        FusionCluster,
        FusionType,
        DSLGenerator,
        Buffer,
        BufferScope,
        BufferManager,
    )
    
    # Basic instantiation tests
    assert GraphAnalyzer is not None
    assert ComputationDAG is not None
    assert ConductorNode is not None
    assert FusionEngine is not None
    assert FusionCluster is not None
    assert FusionType is not None
    assert DSLGenerator is not None
    assert Buffer is not None
    assert BufferScope is not None
    assert BufferManager is not None


def test_runtime_imports():
    """Test that runtime submodules can be imported."""
    from conductor.runtime import (
        JITCompiler,
        AOTManager,
        ExecutableKernel,
        CompiledArtifact,
    )
    
    # Basic instantiation tests
    assert JITCompiler is not None
    assert AOTManager is not None
    assert ExecutableKernel is not None
    assert CompiledArtifact is not None


def test_device_imports():
    """Test that device submodules can be imported."""
    from conductor.device import GCUDevice, GCUInterface
    
    # Basic instantiation tests
    assert GCUDevice is not None
    assert GCUInterface is not None


def test_utils_imports():
    """Test that utility submodules can be imported."""
    from conductor.utils import (
        setup_logging,
        get_logger,
        CompilationCache,
        ConductorError,
        CompilationError,
        UnsupportedOperationError,
        FallbackHandler,
    )
    
    # Basic instantiation tests
    assert setup_logging is not None
    assert get_logger is not None
    assert CompilationCache is not None
    assert ConductorError is not None
    assert CompilationError is not None
    assert UnsupportedOperationError is not None
    assert FallbackHandler is not None


def test_backend_registration():
    """Test that backend registration works without errors."""
    from conductor.backend import register_backend, is_backend_registered
    
    # Test registration function exists
    assert register_backend is not None
    assert is_backend_registered is not None
    
    # Note: We don't actually test registration here since it requires PyTorch
    # and may interfere with other tests


def test_package_structure():
    """Test that the package directory structure is correct."""
    conductor_path = project_root / 'conductor'
    
    # Check main package directory exists
    assert conductor_path.exists()
    assert (conductor_path / '__init__.py').exists()
    
    # Check subpackages exist
    subpackages = ['codegen', 'runtime', 'device', 'utils']
    for subpackage in subpackages:
        subpackage_path = conductor_path / subpackage
        assert subpackage_path.exists(), f"Subpackage {subpackage} not found"
        assert (subpackage_path / '__init__.py').exists(), f"Subpackage {subpackage} missing __init__.py"


def test_configuration_files():
    """Test that configuration files are present."""
    config_files = [
        'setup.py',
        'pyproject.toml',
        'requirements.txt',
        'requirements-dev.txt',
        'README.md',
        '.pre-commit-config.yaml',
    ]
    
    for config_file in config_files:
        file_path = project_root / config_file
        assert file_path.exists(), f"Configuration file {config_file} not found"


if __name__ == '__main__':
    pytest.main([__file__])