GLOBAL RULES for AUGGIE/AUGMENT CODE IDE

0. pre-task reading
- Before start, understand user request and read through:
  - steering docs in .kiro/steering/{product,structure,tech}.md  
  - specs docs in .kiro/specs/conductor-mvp-dev/{requirements,design,tasks}.md  

1. Task Understanding & Decomposition
- Decompose the task into smaller, sequential steps where appropriate.
- If the task is ambiguous or underspecified, identify the uncertainties and attempt to clarify them according to steering/spec docs, do not stop to ask user.

2. Contextual Reading Before Action
- Before performing each step, analyze what information or source materials (e.g., code files, documentation, test cases) must be read to proceed effectively.
- Always prioritize understanding over execution; do not rush into coding without sufficient context comprehension.

3. Behaviors in Coding
- When modifying or adding functionality in the codebase:
  - First, search for existing modules, functions, or components that might already implement similar behavior.
  - Carefully analyze whether the existing implementation satisfies the current requirements.
  - Prefer reusing or refactoring existing code instead of creating new modules.
- About Coding Style
  - avoid mock or simulated implementation, this can confuse the later development and health check of project
  - ALWAYS use humanized coding style, ALWAYS prefer compact/concise implementation.

4. Performance & Development Standards
- All code, comments, and docstrings must be written in clear, fluent English.
- Design principles must favor simplicity and maintainability; avoid introducing unnecessary architectural complexity.
- WHEN init new project, Follow an incremental development approach:
  - BEGIN with a minimal viable implementation (MVP).

5. Shell Command Result Verification & Error Detection
- **MANDATORY**: Always carefully read the ENTIRE output of shell commands, including return codes, error messages, and warnings
- **NEVER** assume success based on partial output or wishful thinking
- **ERROR DETECTION PROTOCOL**: If ANY of the following appear in command output, treat as FAILURE and debug:
  - Non-zero return codes
  - Error messages (containing "error", "failed", "exception", "traceback")
  - Warning messages that indicate problems
  - Empty or truncated output when content was expected
  - Import errors or module not found messages
  - Stack traces or Python exceptions
- **DEBUGGING REQUIREMENT**: When a command fails, you MUST:
  - Analyze the specific error message
  - Identify the root cause
  - Attempt to fix the issue
  - Re-run the command to verify the fix
  - Only mark tasks complete after verified success
- **NO FALSE COMPLETION**: Never mark tasks as "PASSED" or "COMPLETE" based on wishful thinking or partial success
- **REQUIRE CONCRETE EVIDENCE**: Demand concrete evidence of full functionality before marking anything as complete

6. Code Submission & Test-Fix Behavior
- ALWAYS cleaning up smoke test script or temporal files, UNLESS user ask to keep the document.
- If the user requests code submission:
  - Always attempt to fix failing tests before committing code.
  - USE humanized simple commit message, ALWAYS conclude what progresses in feature/development, not side-branching works like tests/scripts
