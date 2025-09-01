import os
import zipfile

# -----------------------------
# 完整增强版 Markdown 内容 (optimized according to Kiro norms: more detailed, accurate format)
# -----------------------------
docs_content = {
    'product.md': """# Conductor: PyTorch Backend Integration Product

## Product Purpose
Conductor is a backend integration for PyTorch that enables seamless use of self-developed Conductor compiler for custom 'gcu' hardware acceleration, allowing ML models to run efficiently on specialized hardware.

## Target Users
- Machine learning engineers and developers using PyTorch for model training and inference.
- Hardware teams working with 'gcu' devices seeking PyTorch compatibility.
- Researchers and organizations looking for custom backend extensions in AI workflows.

## Key Features
- Minimalist DAG + Buffer AST for efficient graph representation.
- Humanized coding style with simple APIs and readable code.
- JIT Mode support: Dynamic conversion from FX Graph to Conductor DSL, compilation, and loading.
- AOT Mode support: Loading and execution of precompiled artifacts.
- Operation fusion for performance optimization.
- Easy installation and integration as a standalone package.

## Business Goals
- Facilitate adoption of 'gcu' hardware in the PyTorch ecosystem.
- Reduce development time for custom backends through spec-driven approach.
- Promote open-source contributions by providing extensible, minimalist design.

## Project Vision
Conductor will integrate PyTorch with the Conductor compiler to achieve 'gcu' hardware acceleration:
- Using a minimalist DAG + Buffer AST.
- Maintaining a humanized coding style.
- Supporting JIT Mode: FX Graph -> Conductor DSL -> compile -> load.
- Supporting AOT Mode: load precompiled artifacts.

## FX Graph -> Conductor DSL
- Represent FX Graph using ConductorNode DAG.
- Enable FusionCluster to combine compatible nodes into a single DSL function.
- Manage buffer scopes: local, shared, global.
- Automatically handle temporary variables for inputs and outputs.
- Ensure DSL output follows DAG topological order.

## Repo Strategy
- Develop as an independent Python package named 'conductor'.
- Installation via 'pip install conductor'.
- Require users to have standard PyTorch installed beforehand.
- Offer API or monkey patch for backend registration.
- Use minimalist setup with setup.py / pyproject.toml and minimal dependencies.

## Key Objectives
1. Implement FX -> Conductor DSL conversion including FusionCluster.
2. Integrate JIT/AOT pipelines with examples in conductor-samples.
3. Keep core implementation Python-only, with C++ limited to device registration.
4. Validate FX -> DSL correctness using FileCheck tests.

## Success Metrics
- Functional completeness: Full support for FX Graph execution on 'gcu'.
- Performance: JIT/AOT execution times comparable to native PyTorch backends.
- Ease of use: Simple API aligning with minimalist and humanized design principles.
- Adoption: Measured by package downloads, GitHub stars, and user feedback.""",

    'tech.md': """# Technical Guidelines for Conductor Integration

## Technology Stack
- Python: >=3.8
- PyTorch: 2.0+
- Package: Independent Python package 'conductor'.
- Device Registration: Python API primary, optional C++ extension using pybind11.
- DSL Codegen: Pure Python conversion from FX Graph to Conductor DSL (.co files).
- Compiler Integration: Subprocess calls to Conductor CLI for compilation.
- Artifact Loading: ctypes for .so files, linker mechanisms for .o files in AOT mode.
- Test Framework: pytest for unit/integration tests, LLVM FileCheck for codegen validation.

## Approved Frameworks and Libraries
- Core: PyTorch, os, subprocess, ctypes.
- Math/Utility: numpy (optional for testing).
- Build: setuptools for setup.py, tomli for pyproject.toml.
- Avoid heavy or unnecessary dependencies to maintain minimalism.

## Technical Constraints
- No runtime internet access required.
- Full compatibility with torch.compile API.
- Fallback mechanism to Inductor for unsupported operations or devices.
- Code must be platform-independent where possible.
- Security: No execution of untrusted code in compilation pipeline.

## FX -> Conductor DSL
- ConductorNode DAG for graph representation.
- FusionCluster heuristics: Fuse elementwise operations and elementwise+reduction patterns.
- Buffer scopes: 'local' for temporaries, 'shared' for inter-kernel communication, 'global' for persistent data.
- DAG topological traversal for code generation order.
- Automatic temporary variable management to prevent naming conflicts and optimize memory.

## Humanized Coding Style
- Naming: Descriptive snake_case for functions/variables, CamelCase for classes.
- Docstrings: Comprehensive for all public APIs, including parameters, returns, examples.
- Modularity: Single-responsibility modules, e.g., separate codegen from backend logic.
- Minimalist: Minimize code complexity, avoid over-abstraction.
- Best Practices: Include type hints, error handling with custom exceptions, logging for debug.""",

    'structure.md': """# Project Structure

conductor/
├── .kiro/
│   ├── steering/
│   │   ├── product.md
│   │   ├── tech.md
│   │   └── structure.md
│   └── specs/
│       ├── design.md
│       ├── requirements.md
│       └── tasks.md
├── conductor/
│   ├── __init__.py
│   ├── backend.py  # Handles backend registration and JIT/AOT pipelines
│   ├── codegen.py  # FX Graph to Conductor DSL conversion logic
│   └── utils.py    # Utilities for buffers, nodes, and fusion
├── conductor/device/
│   ├── gcu.cpp     # C++ code for device registration (optional)
│   └── setup.py    # Build script for C++ extension
├── conductor-samples/
│   ├── add.co      # Example DSL for addition operation
│   ├── mul.co      # Example DSL for multiplication
│   └── ...         # Additional sample DSL files
├── tests/
│   ├── test_jit.py      # Tests for JIT mode
│   ├── test_aot.py      # Tests for AOT mode
│   ├── test_fusion.py   # Tests for fusion clusters
│   └── filecheck/       # Directory for FileCheck test files
├── setup.py
├── pyproject.toml
└── README.md           # Project overview, installation, usage examples

## Naming Conventions
- Files and directories: lowercase_with_underscores
- Classes: CamelCase
- Functions and variables: snake_case
- Constants: UPPER_CASE

## Architecture
- Layered: User API -> Backend -> Codegen -> Compiler Interface
- Extensible points: Custom fusion heuristics via subclassing, additional buffer scopes.
- Version control: Use Git for repo, with .gitignore for build artifacts.""",

    'design.md': """# Design Notes

## Core Concepts

### Buffer
- name: str (variable name)
- scope: 'local' | 'shared' | 'global'
- producer: ConductorNode (node that produces this buffer)
- Inputs: Can be global, shared, or local scopes
- Temporary variables: Default to 'local' scope
- Connections: Used to link across FusionClusters or DSL functions for data flow

### ConductorNode
- op: str (operator name or DSL function)
- inputs: List[Buffer]
- outputs: List[Buffer]
- next_nodes: List[ConductorNode] (for DAG traversal)
- fused: bool (indicates if part of a FusionCluster)
- Purpose: Represents individual nodes in the FX Graph DAG, supports independent DSL generation or fusion

### FusionCluster
- nodes: List[ConductorNode]
- inputs: List[Buffer] (external inputs to the cluster)
- outputs: List[Buffer] (external outputs from the cluster)
- dsl_name: str (name of the generated DSL function)
- Purpose: Fuses sequences of elementwise or elementwise+reduction operators into a single DSL function for performance
- Internal: Shares temporary buffers, replaces original node outputs with cluster outputs

## Conversion Flow (FX -> DSL)
1. Parse fx.GraphModule to build DAG with ConductorNodes and Buffers.
2. Apply fusion heuristics to identify and create FusionClusters (e.g., chain of add/mul ops).
3. Generate DSL code for each cluster or unfused node.
4. Update DAG to connect clusters appropriately.
5. Mode handling: JIT performs on-the-fly compilation; AOT loads existing artifacts.

## Sequence Diagram for JIT Mode
```mermaid
sequenceDiagram
    participant User
    participant PyTorch
    participant Conductor
    participant Compiler
    participant GCU
    User->>PyTorch: torch.compile(model, backend='gcu')
    PyTorch->>Conductor: Provide FX Graph
    Conductor->>Conductor: Build DAG, Fuse, Generate DSL (.co)
    Conductor->>Compiler: Call CLI to compile .co to .so
    Compiler->>Conductor: Return compiled .so
    Conductor->>Conductor: Load .so via ctypes
    PyTorch->>User: Return compiled module
    User->>PyTorch: Execute model
    PyTorch->>Conductor: Run on GCU
    Conductor->>GCU: Execute kernel
```

## Design Principles
- Minimalist: Simple DAG + Buffer AST to reduce complexity.
- Modes: Full support for JIT and AOT workflows.
- Performance: Fusion to minimize kernel launches.
- Flexibility: Buffer scopes for memory optimization.
- Humanized: Clean, intuitive code structure.
- Extensibility: Hooks for custom heuristics, optimizations, variable reuse.

## Implementation Considerations
- Caching: Cache compiled .so for repeated JIT calls.
- Error Handling: Graceful fallback, detailed error messages for compilation failures.
- Testing: End-to-end tests for graph conversion, fusion correctness.
- Scalability: Handle large graphs with efficient traversal algorithms.""",

    'requirements.md': """# Requirements

## Functional Requirements (Using EARS Notation)
WHEN the conductor package is installed and imported,
THE SYSTEM SHALL register the 'gcu' backend for use with torch.compile.

WHEN a PyTorch model is compiled with backend='gcu' in JIT mode,
THE SYSTEM SHALL convert the FX Graph to Conductor DSL, compile it to .so, load it, and enable execution on GCU hardware.

WHEN a PyTorch model is compiled with backend='gcu' in AOT mode,
THE SYSTEM SHALL load precompiled .so or .o artifacts and integrate them for execution on GCU.

WHEN an unsupported operation is encountered in the FX Graph,
THE SYSTEM SHALL fallback to the Inductor backend without error.

WHEN generating Conductor DSL,
THE SYSTEM SHALL support DAG representation, FusionCluster for compatible ops, and buffer scopes (local/shared/global).

WHEN running tests,
THE SYSTEM SHALL validate DSL generation correctness using FileCheck and ensure JIT/AOT pipelines work via unit/integration tests.

WHEN installing the package,
THE SYSTEM SHALL allow pip install conductor as an independent repo with minimal dependencies.

## User Stories and Acceptance Criteria
As a PyTorch developer, I want to use 'gcu' backend seamlessly,
So that my models accelerate on custom hardware.
Acceptance Criteria: torch.compile succeeds, model runs faster on GCU, verified by benchmarks.

As a tester, I want automated tests for fusion,
So that optimizations are correct.
Acceptance Criteria: Test cases pass for elementwise and reduction fusions.

## Non-Functional Requirements
- Platform: Python >=3.8, PyTorch 2.0+.
- Design: Minimalist - simple, readable, maintainable code.
- Style: Humanized - clear naming, docstrings, modular structure.
- Extensibility: Allow future enhancements like additional fusion heuristics or optimizations.
- Compatibility: Fully align with standard torch.compile API and PyTorch updates.""",

    'tasks.md': """# Tasks

## Task 1: Repo Setup
- id: T001
- description: Initialize the independent Python repo for Conductor, including setup for pip installation, virtualenv support, and minimal dependencies.
- dependencies: []
- sub-tasks:
  - Create setup.py and pyproject.toml with required metadata.
  - Define dependencies (e.g., torch).
  - Test installation in a fresh virtualenv.
- outputs: Fully installable 'conductor' package via pip, ready for import and use.

## Task 2: Device Registration
- id: T002
- description: Implement API for registering GCU device, with optional C++ extension for low-level integration.
- dependencies: [T001]
- sub-tasks:
  - Develop Python registration function.
  - Build C++ extension if needed using setup.py.
  - Test registration in torch.compile.
- outputs: 'gcu' backend registered and recognizable by PyTorch.

## Task 3: FX -> DSL Conversion
- id: T003
- description: Develop classes for ConductorNode, Buffer, FusionCluster; build DAG, apply fusion heuristics, and generate DSL.
- dependencies: [T002]
- sub-tasks:
  - Implement Buffer and ConductorNode classes.
  - Add fusion logic for elementwise/reduction.
  - Handle buffer scopes and temp variables.
  - Generate .co files from DAG.
- outputs: DSL files (.co) correctly generated from FX.GraphModule examples.

## Task 4: JIT Integration
- id: T004
- description: Build JIT pipeline to convert FX -> DSL -> compile -> load, with caching for repeated compilations.
- dependencies: [T003]
- sub-tasks:
  - Integrate subprocess for compiler calls.
  - Use ctypes for loading .so.
  - Add caching mechanism.
  - Test with conductor-samples.
- outputs: Models successfully run on GCU via JIT mode.

## Task 5: AOT Integration
- id: T005
- description: Build AOT pipeline to load and integrate precompiled artifacts with torch.compile.
- dependencies: [T003]
- sub-tasks:
  - Implement artifact loading logic.
  - Handle .so and .o formats.
  - Integrate fallback to Inductor.
  - Test loading and execution.
- outputs: Models successfully run on GCU via AOT mode.

## Task 6: Testing
- id: T006
- description: Create unit, integration, and FileCheck tests to validate buffer scopes, fusion, and overall pipelines.
- dependencies: [T004, T005]
- sub-tasks:
  - Write pytest tests for JIT/AOT.
  - Set up FileCheck for DSL output verification.
  - Add coverage for edge cases like unsupported ops.
- outputs: All tests pass, confirming correctness of JIT/AOT and codegen.

## Task 7: Documentation
- id: T007
- description: Complete and refine steering (product, tech, structure) and spec (design, requirements, tasks) documents.
- dependencies: [T006]
- sub-tasks:
  - Update docs with details and diagrams.
  - Ensure alignment with Kiro norms.
  - Add examples in README.md.
- outputs: Comprehensive docs ready for kiro.dev development and collaboration."""
}

# -----------------------------
# 创建目录并写入文件
# -----------------------------
output_dir = 'conductor_kiro_docs'
steering_dir = os.path.join(output_dir, 'steering')
specs_dir = os.path.join(output_dir, 'specs')
os.makedirs(steering_dir, exist_ok=True)
os.makedirs(specs_dir, exist_ok=True)

# 写入 steering docs
for fname in ['product.md', 'tech.md', 'structure.md']:
    with open(os.path.join(steering_dir, fname), 'w', encoding='utf-8') as f:
        f.write(docs_content[fname])

# 写入 spec docs
for fname in ['design.md', 'requirements.md', 'tasks.md']:
    with open(os.path.join(specs_dir, fname), 'w', encoding='utf-8') as f:
        f.write(docs_content[fname])

# 打包成 zip
zip_filename = 'conductor_kiro_docs.zip'
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arc_name = os.path.relpath(full_path, output_dir)
            zipf.write(full_path, arcname=arc_name)

print(f'生成完成: {zip_filename}')
