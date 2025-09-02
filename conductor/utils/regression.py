"""
Performance regression detection and monitoring.

This module provides capabilities to detect performance regressions
by comparing current performance against historical baselines.
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from .logging import get_logger
from .profiler import PerformanceMetrics, ConductorProfiler

logger = get_logger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    name: str
    timestamp: str
    metrics: PerformanceMetrics
    environment: Dict[str, str]
    model_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline to dictionary."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'metrics': self.metrics.to_dict(),
            'environment': self.environment,
            'model_info': self.model_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create baseline from dictionary."""
        metrics = PerformanceMetrics(**data['metrics'])
        return cls(
            name=data['name'],
            timestamp=data['timestamp'],
            metrics=metrics,
            environment=data['environment'],
            model_info=data['model_info']
        )


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    is_regression: bool
    severity: str  # 'minor', 'major', 'critical'
    threshold: float
    
    def __str__(self) -> str:
        """String representation of regression result."""
        direction = "↑" if self.change_percent > 0 else "↓"
        status = "REGRESSION" if self.is_regression else "OK"
        return f"{self.metric_name}: {self.current_value:.3f} vs {self.baseline_value:.3f} " \
               f"({direction}{abs(self.change_percent):.1f}%) [{status}]"


class PerformanceRegressionDetector:
    """
    Performance regression detection system.
    
    This class provides comprehensive regression detection by comparing
    current performance metrics against historical baselines.
    """
    
    def __init__(self, baseline_dir: Optional[str] = None):
        """
        Initialize regression detector.
        
        Args:
            baseline_dir: Directory to store performance baselines
        """
        if baseline_dir is None:
            baseline_dir = Path.home() / '.conductor' / 'baselines'
        
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Regression thresholds (percentage change)
        self.thresholds = {
            'compilation_time': {'minor': 10.0, 'major': 25.0, 'critical': 50.0},
            'execution_time': {'minor': 5.0, 'major': 15.0, 'critical': 30.0},
            'memory_peak_mb': {'minor': 15.0, 'major': 30.0, 'critical': 50.0},
            'fusion_ratio': {'minor': -5.0, 'major': -15.0, 'critical': -25.0},  # Negative = worse
            'cache_hit_ratio': {'minor': -10.0, 'major': -20.0, 'critical': -35.0},
            'buffer_reuse_ratio': {'minor': -10.0, 'major': -20.0, 'critical': -35.0}
        }
    
    def create_baseline(
        self, 
        name: str, 
        metrics: PerformanceMetrics, 
        model_info: Dict[str, Any],
        environment: Optional[Dict[str, str]] = None
    ) -> PerformanceBaseline:
        """
        Create a new performance baseline.
        
        Args:
            name: Baseline name/identifier
            metrics: Performance metrics
            model_info: Information about the model
            environment: Environment information
            
        Returns:
            Created performance baseline
        """
        if environment is None:
            environment = self._get_environment_info()
        
        baseline = PerformanceBaseline(
            name=name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            environment=environment,
            model_info=model_info
        )
        
        # Save baseline to disk
        self._save_baseline(baseline)
        
        logger.info(f"Created performance baseline: {name}")
        return baseline
    
    def detect_regressions(
        self, 
        current_metrics: PerformanceMetrics,
        baseline_name: str,
        model_info: Dict[str, Any]
    ) -> List[RegressionResult]:
        """
        Detect performance regressions against a baseline.
        
        Args:
            current_metrics: Current performance metrics
            baseline_name: Name of baseline to compare against
            model_info: Information about current model
            
        Returns:
            List of regression results
        """
        baseline = self._load_baseline(baseline_name)
        if baseline is None:
            logger.warning(f"Baseline {baseline_name} not found")
            return []
        
        # Validate model compatibility
        if not self._models_compatible(baseline.model_info, model_info):
            logger.warning(f"Model incompatible with baseline {baseline_name}")
            return []
        
        results = []
        current_dict = current_metrics.to_dict()
        baseline_dict = baseline.metrics.to_dict()
        
        for metric_name in current_dict:
            if metric_name in baseline_dict and metric_name in self.thresholds:
                result = self._analyze_metric_regression(
                    metric_name,
                    current_dict[metric_name],
                    baseline_dict[metric_name]
                )
                if result:
                    results.append(result)
        
        # Log regression summary
        regressions = [r for r in results if r.is_regression]
        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")
            for regression in regressions:
                logger.warning(f"  {regression}")
        else:
            logger.info("No performance regressions detected")
        
        return results
    
    def get_performance_trend(
        self, 
        baseline_name: str, 
        metric_name: str,
        days: int = 30
    ) -> List[Tuple[str, float]]:
        """
        Get performance trend for a specific metric.
        
        Args:
            baseline_name: Baseline name pattern
            metric_name: Metric to analyze
            days: Number of days to look back
            
        Returns:
            List of (timestamp, value) tuples
        """
        baselines = self._load_recent_baselines(baseline_name, days)
        
        trend_data = []
        for baseline in baselines:
            if hasattr(baseline.metrics, metric_name):
                value = getattr(baseline.metrics, metric_name)
                trend_data.append((baseline.timestamp, value))
        
        # Sort by timestamp
        trend_data.sort(key=lambda x: x[0])
        
        return trend_data
    
    def generate_regression_report(
        self, 
        results: List[RegressionResult],
        baseline_name: str
    ) -> str:
        """
        Generate comprehensive regression report.
        
        Args:
            results: Regression analysis results
            baseline_name: Baseline name
            
        Returns:
            Formatted regression report
        """
        report_lines = [
            f"Performance Regression Report",
            f"Baseline: {baseline_name}",
            f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # Summary statistics
        total_metrics = len(results)
        regressions = [r for r in results if r.is_regression]
        critical_regressions = [r for r in regressions if r.severity == 'critical']
        major_regressions = [r for r in regressions if r.severity == 'major']
        minor_regressions = [r for r in regressions if r.severity == 'minor']
        
        report_lines.extend([
            "Summary:",
            f"  Total Metrics Analyzed: {total_metrics}",
            f"  Regressions Detected: {len(regressions)}",
            f"    Critical: {len(critical_regressions)}",
            f"    Major: {len(major_regressions)}",
            f"    Minor: {len(minor_regressions)}",
            ""
        ])
        
        # Detailed results
        if regressions:
            report_lines.extend([
                "Regression Details:",
                "-" * 40
            ])
            
            for regression in sorted(regressions, key=lambda r: r.change_percent, reverse=True):
                report_lines.append(f"  {regression}")
            
            report_lines.append("")
        
        # Improvements
        improvements = [r for r in results if not r.is_regression and r.change_percent < -1.0]
        if improvements:
            report_lines.extend([
                "Performance Improvements:",
                "-" * 40
            ])
            
            for improvement in sorted(improvements, key=lambda r: r.change_percent):
                report_lines.append(f"  {improvement}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "Recommendations:",
            "-" * 20
        ])
        
        if critical_regressions:
            report_lines.append("  🚨 CRITICAL: Immediate investigation required")
        elif major_regressions:
            report_lines.append("  ⚠️  MAJOR: Performance review recommended")
        elif minor_regressions:
            report_lines.append("  ℹ️  MINOR: Monitor for trends")
        else:
            report_lines.append("  ✅ No significant regressions detected")
        
        return "\n".join(report_lines)
    
    def _analyze_metric_regression(
        self, 
        metric_name: str, 
        current_value: float, 
        baseline_value: float
    ) -> Optional[RegressionResult]:
        """Analyze regression for a specific metric."""
        if baseline_value == 0:
            return None
        
        change_percent = ((current_value - baseline_value) / baseline_value) * 100
        thresholds = self.thresholds.get(metric_name, {})
        
        # Determine if this is a regression and its severity
        is_regression = False
        severity = 'none'
        threshold = 0.0
        
        # For metrics where higher is worse (time, memory)
        if metric_name in ['compilation_time', 'execution_time', 'memory_peak_mb', 'memory_allocated_mb']:
            if change_percent >= thresholds.get('critical', 50.0):
                is_regression = True
                severity = 'critical'
                threshold = thresholds['critical']
            elif change_percent >= thresholds.get('major', 25.0):
                is_regression = True
                severity = 'major'
                threshold = thresholds['major']
            elif change_percent >= thresholds.get('minor', 10.0):
                is_regression = True
                severity = 'minor'
                threshold = thresholds['minor']
        
        # For metrics where lower is worse (ratios)
        elif metric_name in ['fusion_ratio', 'cache_hit_ratio', 'buffer_reuse_ratio']:
            if change_percent <= thresholds.get('critical', -25.0):
                is_regression = True
                severity = 'critical'
                threshold = abs(thresholds['critical'])
            elif change_percent <= thresholds.get('major', -15.0):
                is_regression = True
                severity = 'major'
                threshold = abs(thresholds['major'])
            elif change_percent <= thresholds.get('minor', -5.0):
                is_regression = True
                severity = 'minor'
                threshold = abs(thresholds['minor'])
        
        return RegressionResult(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            change_percent=change_percent,
            is_regression=is_regression,
            severity=severity,
            threshold=threshold
        )
    
    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to disk."""
        filename = f"{baseline.name}_{baseline.timestamp.replace(':', '-')}.json"
        filepath = self.baseline_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(baseline.to_dict(), f, indent=2)
    
    def _load_baseline(self, name: str) -> Optional[PerformanceBaseline]:
        """Load most recent baseline with given name."""
        pattern = f"{name}_*.json"
        matching_files = list(self.baseline_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Get most recent file
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            return PerformanceBaseline.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load baseline {latest_file}: {e}")
            return None
    
    def _load_recent_baselines(self, name_pattern: str, days: int) -> List[PerformanceBaseline]:
        """Load recent baselines matching pattern."""
        pattern = f"{name_pattern}_*.json"
        matching_files = list(self.baseline_dir.glob(pattern))
        
        # Filter by age
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_files = [f for f in matching_files if f.stat().st_mtime > cutoff_time]
        
        baselines = []
        for filepath in recent_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                baseline = PerformanceBaseline.from_dict(data)
                baselines.append(baseline)
            except Exception as e:
                logger.warning(f"Failed to load baseline {filepath}: {e}")
        
        return baselines
    
    def _models_compatible(self, baseline_info: Dict[str, Any], current_info: Dict[str, Any]) -> bool:
        """Check if models are compatible for comparison."""
        # Check model type
        if baseline_info.get('type') != current_info.get('type'):
            return False
        
        # Check input shape compatibility
        baseline_shape = baseline_info.get('input_shape')
        current_shape = current_info.get('input_shape')
        
        if baseline_shape and current_shape:
            # Allow some flexibility in batch size
            if len(baseline_shape) != len(current_shape):
                return False
            
            # Compare non-batch dimensions
            for i in range(1, len(baseline_shape)):
                if baseline_shape[i] != current_shape[i]:
                    return False
        
        return True
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get current environment information."""
        import platform
        import sys
        
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "unknown"
        
        try:
            import conductor
            conductor_version = conductor.__version__
        except ImportError:
            conductor_version = "unknown"
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'torch_version': torch_version,
            'conductor_version': conductor_version,
            'hostname': platform.node()
        }


class ContinuousPerformanceMonitor:
    """
    Continuous performance monitoring system.
    
    This class provides ongoing performance monitoring and alerting
    for production deployments.
    """
    
    def __init__(self, detector: PerformanceRegressionDetector):
        """
        Initialize continuous monitor.
        
        Args:
            detector: Regression detector instance
        """
        self.detector = detector
        self.monitoring_active = False
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: callable):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def monitor_compilation(
        self, 
        model_name: str,
        profiler: ConductorProfiler,
        model_info: Dict[str, Any]
    ):
        """
        Monitor a compilation for performance issues.
        
        Args:
            model_name: Name/identifier for the model
            profiler: Profiler with compilation metrics
            model_info: Model information
        """
        if not self.monitoring_active:
            return
        
        metrics = profiler.metrics
        
        # Check against baseline
        baseline_name = f"{model_name}_baseline"
        regressions = self.detector.detect_regressions(metrics, baseline_name, model_info)
        
        # Trigger alerts for significant regressions
        critical_regressions = [r for r in regressions if r.severity == 'critical']
        major_regressions = [r for r in regressions if r.severity == 'major']
        
        if critical_regressions or major_regressions:
            self._trigger_alerts(model_name, regressions)
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.monitoring_active = True
        logger.info("Continuous performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logger.info("Continuous performance monitoring stopped")
    
    def _trigger_alerts(self, model_name: str, regressions: List[RegressionResult]):
        """Trigger performance alerts."""
        alert_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'regressions': [asdict(r) for r in regressions]
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


def create_performance_monitoring_system(baseline_dir: Optional[str] = None) -> Tuple[PerformanceRegressionDetector, ContinuousPerformanceMonitor]:
    """
    Create a complete performance monitoring system.
    
    Args:
        baseline_dir: Directory for storing baselines
        
    Returns:
        Tuple of (detector, monitor)
    """
    detector = PerformanceRegressionDetector(baseline_dir)
    monitor = ContinuousPerformanceMonitor(detector)
    
    return detector, monitor