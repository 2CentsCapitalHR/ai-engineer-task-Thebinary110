"""
Comprehensive Logging System
Handles metrics, performance tracking, and error logging.
"""

import logging
import logging.handlers
import time
import json
import functools
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime
import threading
from collections import defaultdict, deque
import sys

class PerformanceTracker:
    """Track performance metrics for various operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Record timing for an operation."""
        with self.lock:
            entry = {
                "duration": duration,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            self.metrics[operation].append(entry)
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            if operation:
                if operation not in self.metrics:
                    return {}
                
                durations = [entry["duration"] for entry in self.metrics[operation]]
                return self._calculate_stats(operation, durations)
            else:
                # Return stats for all operations
                stats = {}
                for op, entries in self.metrics.items():
                    durations = [entry["duration"] for entry in entries]
                    stats[op] = self._calculate_stats(op, durations)
                return stats
    
    def _calculate_stats(self, operation: str, durations: list) -> Dict[str, Any]:
        """Calculate statistics for a list of durations."""
        if not durations:
            return {"operation": operation, "count": 0}
        
        durations.sort()
        count = len(durations)
        
        return {
            "operation": operation,
            "count": count,
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / count,
            "median": durations[count // 2],
            "p95": durations[int(count * 0.95)] if count > 0 else 0,
            "p99": durations[int(count * 0.99)] if count > 0 else 0
        }

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class MetricsCollector:
    """Collect and aggregate application metrics."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.lock = threading.Lock()
    
    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            key = self._build_key(metric, tags)
            self.counters[key] += value
    
    def set_gauge(self, metric: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        with self.lock:
            key = self._build_key(metric, tags)
            self.gauges[key] = value
    
    def _build_key(self, metric: str, tags: Dict[str, str] = None) -> str:
        """Build metric key with tags."""
        if not tags:
            return metric
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}[{tag_str}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timestamp": time.time()
            }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()

class EnhancedLogger:
    """Enhanced logger with performance tracking and metrics."""
    
    def __init__(self, name: str, log_dir: str = "adgm_corporate_agent/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_tracker = PerformanceTracker()
        self.metrics_collector = MetricsCollector()
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        
        # JSON handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.json",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        
        # Error handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra fields."""
        self.logger.info(message, extra=kwargs)
        self.metrics_collector.increment("log.info")
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
        self.metrics_collector.increment("log.debug")
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
        self.metrics_collector.increment("log.warning")
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
        self.metrics_collector.increment("log.error")
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
        self.metrics_collector.increment("log.critical")
    
    def time_operation(self, operation: str):
        """Decorator to time operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.performance_tracker.record_timing(
                        operation, 
                        duration,
                        {"success": True, "function": func.__name__}
                    )
                    
                    self.info(
                        f"Operation '{operation}' completed",
                        duration=duration,
                        function=func.__name__,
                        operation=operation
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.performance_tracker.record_timing(
                        operation, 
                        duration,
                        {"success": False, "error": str(e), "function": func.__name__}
                    )
                    
                    self.error(
                        f"Operation '{operation}' failed",
                        duration=duration,
                        function=func.__name__,
                        operation=operation,
                        error=str(e)
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_document_processing(self, 
                               document_name: str, 
                               document_type: str,
                               processing_time: float,
                               issues_found: int,
                               success: bool = True):
        """Log document processing details."""
        self.info(
            f"Document processed: {document_name}",
            document_name=document_name,
            document_type=document_type,
            processing_time=processing_time,
            issues_found=issues_found,
            success=success
        )
        
        # Update metrics
        self.metrics_collector.increment("documents.processed")
        self.metrics_collector.increment(f"documents.type.{document_type.lower().replace(' ', '_')}")
        self.metrics_collector.set_gauge("documents.avg_processing_time", processing_time)
        
        if issues_found > 0:
            self.metrics_collector.increment("documents.with_issues")
    
    def log_rag_query(self, 
                     query: str, 
                     results_count: int, 
                     retrieval_time: float,
                     confidence_scores: list = None):
        """Log RAG query details."""
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        self.info(
            f"RAG query executed",
            query_length=len(query),
            results_count=results_count,
            retrieval_time=retrieval_time,
            avg_confidence=avg_confidence
        )
        
        # Update metrics
        self.metrics_collector.increment("rag.queries")
        self.metrics_collector.set_gauge("rag.avg_results", results_count)
        self.metrics_collector.set_gauge("rag.avg_retrieval_time", retrieval_time)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "performance_stats": self.performance_tracker.get_stats(),
            "metrics": self.metrics_collector.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"metrics_{timestamp}.json"
        
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(filepath)

# Global logger registry
_loggers = {}

def setup_logger(name: str, log_dir: str = "adgm_corporate_agent/logs") -> EnhancedLogger:
    """Setup or get existing logger."""
    if name not in _loggers:
        _loggers[name] = EnhancedLogger(name, log_dir)
    return _loggers[name]

def get_logger(name: str) -> EnhancedLogger:
    """Get existing logger."""
    return _loggers.get(name, setup_logger(name))