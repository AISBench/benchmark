#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit test runner script
Supports running all unit tests in specified folders and generating coverage reports
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner"""
    
    def __init__(self, test_dir: str, output_dir: str = "test_reports", source_dirs: Optional[List[str]] = None):
        """
        Initialize test runner
        
        Args:
            test_dir: Test folder path
            output_dir: Output directory
            source_dirs: Manually specified source code directory list (optional)
        """
        self.test_dir = Path(test_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Coverage report file paths
        self.coverage_html = self.output_dir / "htmlcov"
        self.coverage_xml = self.output_dir / "coverage.xml"
        self.coverage_json = self.output_dir / "coverage.json"
        
        # Infer corresponding source code directories
        if source_dirs:
            self.source_dirs = source_dirs
            print(f"🎯 Using manually specified source directories: {', '.join(self.source_dirs)}")
        else:
            self.source_dirs = self._infer_source_directories()
    
    def _infer_source_directories(self) -> List[str]:
        """
        Infer corresponding source code directories based on test directory
        
        Returns:
            List of source code directories
        """
        source_dirs = []
        
        # Get test directory path relative to project root
        try:
            relative_test_path = self.test_dir.relative_to(self.project_root)
        except ValueError:
            # If test directory is not under project root, return empty list
            return source_dirs
        
        # Convert test path to source code path
        # Example: tests/UT/cli -> ais_bench/benchmark/cli
        if relative_test_path.parts[0] == 'tests' and len(relative_test_path.parts) >= 2:
            if relative_test_path.parts[1] == 'UT':
                # For UT tests, map to ais_bench/benchmark directory
                if len(relative_test_path.parts) > 2:
                    # Case with subdirectories, e.g. tests/UT/cli -> ais_bench/benchmark/cli
                    sub_path = Path(*relative_test_path.parts[2:])
                    source_path = self.project_root / 'ais_bench' / 'benchmark' / sub_path
                else:
                    # Case with only tests/UT, map to entire ais_bench/benchmark directory
                    source_path = self.project_root / 'ais_bench' / 'benchmark'
                
                if source_path.exists():
                    source_dirs.append(str(source_path))
                    print(f"🎯 Detected test directory: {self.test_dir}")
                    print(f"📁 Corresponding source directory: {source_path}")
                else:
                    print(f"⚠️  Warning: Corresponding source directory does not exist: {source_path}")
        
        # If no matching source directory found, default to entire ais_bench package
        if not source_dirs:
            source_dirs = ['ais_bench']
            print(f"📦 Using default source directory: ais_bench")
        
        return source_dirs
        
    def check_dependencies(self) -> bool:
        """Check if necessary dependencies are installed"""
        required_packages = ['pytest', 'pytest-cov', 'coverage']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Please run the following command to install:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ Dependency check passed")
        return True
    
    def find_test_files(self) -> List[Path]:
        """Find test files"""
        test_files = []
        
        if not self.test_dir.exists():
            print(f"❌ Test directory does not exist: {self.test_dir}")
            return test_files
        
        # Find all test files
        for pattern in ['test_*.py', '*_test.py']:
            test_files.extend(self.test_dir.rglob(pattern))
        
        # Remove duplicates and sort
        test_files = sorted(list(set(test_files)))
        
        print(f"📁 Found {len(test_files)} test files in {self.test_dir}")
        return test_files
    
    def clean_cache(self) -> None:
        """Clean Python cache files to avoid import conflicts"""
        import shutil
        
        print("🧹 Cleaning Python cache files...")
        
        # Delete all __pycache__ directories in tests/UT
        cache_dirs = list(self.test_dir.rglob("__pycache__"))
        for cache_dir in cache_dirs:
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
        
        # Delete all .pyc files in tests/UT
        pyc_files = list(self.test_dir.rglob("*.pyc"))
        for pyc_file in pyc_files:
            try:
                pyc_file.unlink()
            except Exception:
                pass
        
        # Delete all .pyo files in tests/UT
        pyo_files = list(self.test_dir.rglob("*.pyo"))
        for pyo_file in pyo_files:
            try:
                pyo_file.unlink()
            except Exception:
                pass
        
        # Delete pytest cache in tests directory and project root
        pytest_cache_paths = [
            self.test_dir / ".pytest_cache",
            self.project_root / ".pytest_cache",
            self.project_root / "tests" / ".pytest_cache",
        ]
        for cache_path in pytest_cache_paths:
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                except Exception:
                    pass
        
        # Also clean __pycache__ in project root if it exists
        root_cache = self.project_root / "__pycache__"
        if root_cache.exists():
            try:
                shutil.rmtree(root_cache)
            except Exception:
                pass
        
        print("✅ Cache cleaned")
    
    def run_tests(self, 
                  verbose: bool = False,
                  parallel: Optional[object] = None,  # Can be int, 'auto', or None
                  max_failures: Optional[int] = None,
                  specific_tests: Optional[List[str]] = None) -> bool:
        """
        Run tests
        
        Args:
            verbose: Whether to show detailed output
            parallel: Number of parallel processes (None means no parallel, 'auto' means auto-detect, int means specific worker count)
            max_failures: Maximum number of failures
            specific_tests: Specific test files to run
            
        Returns:
            Whether tests were successful
        """
        # Clean cache before running tests to avoid import conflicts
        self.clean_cache()
        
        print(f"\n🚀 Starting test execution...")
        print(f"📂 Test directory: {self.test_dir}")
        print(f"📊 Report directory: {self.output_dir}")
        
        # Build pytest command
        # Add --cache-clear to ensure pytest doesn't use stale cache
        # Add --import-mode=importlib to avoid module import conflicts with same-named test files
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.test_dir),
            '--tb=short',  # Short error traceback
            '--cache-clear',  # Clear pytest cache before running
            '--import-mode=importlib',  # Use importlib to avoid conflicts with same-named modules
        ]
        
        # Add coverage configuration
        if self.source_dirs:
            # Use inferred source directories
            for source_dir in self.source_dirs:
                cmd.extend(['--cov', source_dir])
            print(f"📊 Coverage will measure the following source directories: {', '.join(self.source_dirs)}")
        else:
            # Default to entire ais_bench package
            cmd.append('--cov=ais_bench')
            print("📊 Coverage will measure entire ais_bench package")
        
        # Add coverage data file configuration
        pytest_ini_path = Path(__file__).parent / 'pytest.ini'
        if pytest_ini_path.exists():
            cmd.extend(['--cov-config', str(pytest_ini_path)])
        else:
            # If tests/pytest.ini doesn't exist, use .coveragerc in project root
            coveragerc_path = self.project_root / '.coveragerc'
            if coveragerc_path.exists():
                cmd.extend(['--cov-config', str(coveragerc_path)])
        
        # Add coverage report configuration
        cmd.extend([
            '--cov-report=html:' + str(self.coverage_html),
            '--cov-report=xml:' + str(self.coverage_xml),
            '--cov-report=json:' + str(self.coverage_json),
            '--cov-report=term-missing',
        ])
        
        # Add options
        if verbose:
            cmd.append('-v')
        
        if parallel is not None:
            if parallel == 'auto':
                cmd.extend(['-n', 'auto'])  # Use pytest-xdist for parallel testing with auto-detect
                print("⚡ Running tests in parallel mode (auto-detect workers)")
            elif isinstance(parallel, int) and parallel > 0:
                cmd.extend(['-n', str(parallel)])  # Use pytest-xdist for parallel testing with specified worker count
                print(f"⚡ Running tests in parallel mode with {parallel} workers")
            else:
                # If parallel is 0 or negative, use auto
                cmd.extend(['-n', 'auto'])
                print("⚡ Running tests in parallel mode (auto-detect workers)")
        
        if max_failures:
            cmd.extend(['--maxfail', str(max_failures)])
        
        # If specific test files are specified
        if specific_tests:
            cmd.extend(specific_tests)
        
        # Add other useful options
        cmd.extend([
            '--strict-markers',  # Strict marker checking
            '--disable-warnings',  # Disable warnings
            '--color=yes',  # Colored output
        ])
        
        print(f"🔧 Executing command: {' '.join(cmd)}")
        print("-" * 80)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run tests
            result = subprocess.run(cmd, cwd=self.project_root, check=False)
            
            # Calculate runtime
            end_time = time.time()
            duration = end_time - start_time
            
            print("-" * 80)
            print(f"⏱️  Test runtime: {duration:.2f} seconds")
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                return True
            else:
                print(f"❌ Tests failed (exit code: {result.returncode})")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def generate_summary_report(self) -> None:
        """Generate test summary report"""
        summary_file = self.output_dir / "test_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Test Execution Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test directory: {self.test_dir}\n")
            f.write(f"Source directory: {', '.join(self.source_dirs) if self.source_dirs else 'ais_bench'}\n")
            f.write(f"Report directory: {self.output_dir}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Coverage report files:\n")
            f.write(f"- HTML report: {self.coverage_html}/index.html\n")
            f.write(f"- XML report: {self.coverage_xml}\n")
            f.write(f"- JSON report: {self.coverage_json}\n")
        
        print(f"📄 Test summary saved to: {summary_file}")
    
    def open_coverage_report(self) -> None:
        """Open coverage report"""
        html_file = self.coverage_html / "index.html"
        if html_file.exists():
            print(f"🌐 Coverage HTML report: {html_file}")
            print("You can open it in a browser to view detailed report")
        else:
            print("❌ Coverage HTML report not generated")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run unit tests and generate coverage reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python run_tests.py tests/UT                    # Run all tests in tests/UT directory
        python run_tests.py tests/UT/cli -v             # Run tests in cli directory with verbose output
        python run_tests.py tests/UT -p                 # Run tests in parallel (auto-detect workers)
        python run_tests.py tests/UT -p 4               # Run tests in parallel with 4 workers
        python run_tests.py tests/UT --parallel 8        # Run tests in parallel with 8 workers
        python run_tests.py tests/UT --max-failures 5   # Allow maximum 5 test failures
        python run_tests.py tests/UT --output reports   # Specify output directory
        python run_tests.py tests/UT/cli --source-dirs ais_bench/benchmark/cli  # Manually specify source directories
        """
    )
    
    parser.add_argument(
        'test_dir',
        default='tests/UT',
        help='Test folder path'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='test_reports',
        help='Output directory (default: test_reports)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        type=lambda x: int(x) if x and x.isdigit() else 'auto',
        nargs='?',
        const='auto',
        metavar='WORK_NUM',
        help='Run tests in parallel with specified number of workers (requires pytest-xdist). '
             'If no number is specified, uses auto-detect. Example: -p 4 or --parallel 4'
    )
    
    parser.add_argument(
        '--max-failures',
        type=int,
        help='Maximum number of failures'
    )
    
    parser.add_argument(
        '--specific-tests',
        nargs='+',
        help='Specify particular test files to run'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Only check dependencies, do not run tests'
    )
    
    parser.add_argument(
        '--source-dirs',
        nargs='+',
        help='Manually specify source directories for coverage measurement (overrides auto-inference)'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(args.test_dir, args.output, args.source_dirs)
    
    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependency check completed")
        sys.exit(0)
    
    # Find test files
    test_files = runner.find_test_files()
    if not test_files:
        print("❌ No test files found")
        sys.exit(1)
    
    # Run tests
    success = runner.run_tests(
        verbose=args.verbose,
        parallel=args.parallel,
        max_failures=args.max_failures,
        specific_tests=args.specific_tests
    )
    
    # Generate summary report
    runner.generate_summary_report()
    
    # Display coverage report information
    runner.open_coverage_report()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
