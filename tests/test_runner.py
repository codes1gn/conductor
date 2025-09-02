"""
Comprehensive test runner for Conductor test suite.

This module provides utilities for running different categories of tests
and generating comprehensive reports.
"""

import pytest
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


class ConductorTestRunner:
    """Test runner for Conductor test suite."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize test runner."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, Any]:
        """Run unit tests."""
        print("Running unit tests...")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit"),
            "-m", "unit",
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
            
        if coverage:
            args.extend([
                "--cov=conductor",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit"
            ])
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': 'unit',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("Running integration tests...")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': 'integration',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def run_filecheck_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run FileCheck DSL validation tests."""
        print("Running FileCheck tests...")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir / "filecheck"),
            "-m", "filecheck",
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': 'filecheck',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("Running performance tests...")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-m", "performance",
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': 'performance',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, Any]:
        """Run all test categories."""
        print("Running complete test suite...")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
            
        if coverage:
            args.extend([
                "--cov=conductor",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/complete",
                "--cov-report=xml",
                "--cov-fail-under=90"
            ])
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': 'all',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def run_specific_tests(self, test_pattern: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests matching a specific pattern."""
        print(f"Running tests matching pattern: {test_pattern}")
        
        args = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-k", test_pattern,
            "--tb=short"
        ]
        
        if verbose:
            args.append("-v")
        
        start_time = time.time()
        result = subprocess.run(args, capture_output=True, text=True, cwd=self.project_root)
        duration = time.time() - start_time
        
        return {
            'category': f'pattern:{test_pattern}',
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive test report."""
        report_lines = [
            "=" * 80,
            "CONDUCTOR TEST SUITE REPORT",
            "=" * 80,
            ""
        ]
        
        total_duration = sum(r['duration'] for r in results)
        successful_tests = sum(1 for r in results if r['success'])
        total_tests = len(results)
        
        # Summary
        report_lines.extend([
            f"Total test categories: {total_tests}",
            f"Successful categories: {successful_tests}",
            f"Failed categories: {total_tests - successful_tests}",
            f"Total duration: {total_duration:.2f} seconds",
            ""
        ])
        
        # Detailed results
        for result in results:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            report_lines.extend([
                f"{status} {result['category'].upper()} ({result['duration']:.2f}s)",
                ""
            ])
            
            if not result['success']:
                report_lines.extend([
                    "STDERR:",
                    result['stderr'][:500] + "..." if len(result['stderr']) > 500 else result['stderr'],
                    ""
                ])
        
        # Coverage information
        if any('cov' in r['stdout'] for r in results):
            report_lines.extend([
                "COVERAGE INFORMATION:",
                "Coverage reports generated in htmlcov/ directory",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            ""
        ])
        
        if successful_tests == total_tests:
            report_lines.append("✓ All tests passed! The codebase is ready for production.")
        else:
            report_lines.extend([
                "✗ Some tests failed. Please review the failures above.",
                "- Check error messages for specific issues",
                "- Ensure all dependencies are installed",
                "- Verify test environment setup"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def check_test_environment(self) -> Dict[str, Any]:
        """Check test environment setup."""
        print("Checking test environment...")
        
        checks = {
            'python_version': sys.version_info >= (3, 8),
            'pytest_available': True,
            'conductor_importable': True,
            'torch_available': True,
            'test_directory_exists': self.test_dir.exists(),
            'coverage_available': True
        }
        
        # Check pytest
        try:
            import pytest
            checks['pytest_version'] = pytest.__version__
        except ImportError:
            checks['pytest_available'] = False
        
        # Check conductor
        try:
            import conductor
            checks['conductor_version'] = conductor.__version__
        except ImportError:
            checks['conductor_importable'] = False
        
        # Check torch
        try:
            import torch
            checks['torch_version'] = torch.__version__
        except ImportError:
            checks['torch_available'] = False
        
        # Check coverage
        try:
            import coverage
            checks['coverage_version'] = coverage.__version__
        except ImportError:
            checks['coverage_available'] = False
        
        return checks
    
    def print_environment_report(self, env_checks: Dict[str, Any]) -> None:
        """Print environment check report."""
        print("\nTest Environment Report:")
        print("-" * 40)
        
        for check, result in env_checks.items():
            if isinstance(result, bool):
                status = "✓" if result else "✗"
                print(f"{status} {check}: {result}")
            else:
                print(f"✓ {check}: {result}")
        
        print()


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conductor Test Runner")
    parser.add_argument("--category", choices=["unit", "integration", "filecheck", "performance", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--pattern", help="Run tests matching pattern")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--check-env", action="store_true", help="Check test environment only")
    
    args = parser.parse_args()
    
    runner = ConductorTestRunner()
    
    # Check environment
    env_checks = runner.check_test_environment()
    runner.print_environment_report(env_checks)
    
    if args.check_env:
        return
    
    # Check for critical environment issues
    critical_issues = [
        not env_checks['python_version'],
        not env_checks['pytest_available'],
        not env_checks['conductor_importable'],
        not env_checks['torch_available'],
        not env_checks['test_directory_exists']
    ]
    
    if any(critical_issues):
        print("❌ Critical environment issues detected. Please fix before running tests.")
        return 1
    
    # Run tests
    results = []
    coverage = not args.no_coverage
    
    if args.pattern:
        result = runner.run_specific_tests(args.pattern, args.verbose)
        results.append(result)
    elif args.category == "unit":
        result = runner.run_unit_tests(args.verbose, coverage)
        results.append(result)
    elif args.category == "integration":
        result = runner.run_integration_tests(args.verbose)
        results.append(result)
    elif args.category == "filecheck":
        result = runner.run_filecheck_tests(args.verbose)
        results.append(result)
    elif args.category == "performance":
        result = runner.run_performance_tests(args.verbose)
        results.append(result)
    elif args.category == "all":
        # Run all categories
        results.append(runner.run_unit_tests(args.verbose, coverage))
        results.append(runner.run_integration_tests(args.verbose))
        results.append(runner.run_filecheck_tests(args.verbose))
        # Skip performance tests by default unless explicitly requested
        if args.category == "performance":
            results.append(runner.run_performance_tests(args.verbose))
    
    # Generate and print report
    report = runner.generate_test_report(results)
    print("\n" + report)
    
    # Return appropriate exit code
    if all(r['success'] for r in results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())