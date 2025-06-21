"""
Performance Profiler for Optimized Inference Pipeline
Tracks and analyzes performance improvements across the inference pipeline
"""

import time
import torch
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ProfilePoint:
    """Individual profiling measurement"""
    name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    device: str
    metadata: Dict[str, Any]

@dataclass
class InferenceProfile:
    """Complete inference profiling session"""
    session_id: str
    model_type: str
    device: str
    total_sequences: int
    total_frames: int
    start_time: float
    end_time: float
    total_duration: float
    profile_points: List[ProfilePoint]
    device_stats: Dict[str, Any]
    optimization_summary: Dict[str, Any]

class PerformanceProfiler:
    """
    Comprehensive performance profiler for inference pipeline optimization
    """
    
    def __init__(self, enable_memory_tracking: bool = True, 
                 enable_device_tracking: bool = True):
        """
        Initialize performance profiler
        
        Args:
            enable_memory_tracking: Track memory usage
            enable_device_tracking: Track device-specific metrics
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_device_tracking = enable_device_tracking
        
        # Current session data
        self.current_session = None
        self.profile_points = []
        self.active_contexts = {}
        
        # Historical data
        self.completed_sessions = []
        
        # Device monitoring
        self.device_monitor = DeviceMonitor() if enable_device_tracking else None
        
        logger.info("Performance profiler initialized")
    
    def start_session(self, session_id: str, model_type: str, device: str) -> str:
        """
        Start a new profiling session
        
        Args:
            session_id: Unique session identifier
            model_type: Type of model being profiled
            device: Device being used
            
        Returns:
            Session ID
        """
        if self.current_session is not None:
            logger.warning("Ending previous session before starting new one")
            self.end_session()
        
        self.current_session = {
            "session_id": session_id,
            "model_type": model_type,
            "device": device,
            "start_time": time.perf_counter(),
            "total_sequences": 0,
            "total_frames": 0
        }
        
        self.profile_points = []
        self.active_contexts = {}
        
        if self.device_monitor:
            self.device_monitor.start_monitoring()
        
        logger.info(f"Started profiling session: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[InferenceProfile]:
        """
        End current profiling session and return results
        
        Returns:
            Complete inference profile or None if no active session
        """
        if self.current_session is None:
            logger.warning("No active session to end")
            return None
        
        end_time = time.perf_counter()
        total_duration = end_time - self.current_session["start_time"]
        
        # Get device statistics
        device_stats = {}
        if self.device_monitor:
            device_stats = self.device_monitor.stop_monitoring()
        
        # Create inference profile
        profile = InferenceProfile(
            session_id=self.current_session["session_id"],
            model_type=self.current_session["model_type"],
            device=self.current_session["device"],
            total_sequences=self.current_session["total_sequences"],
            total_frames=self.current_session["total_frames"],
            start_time=self.current_session["start_time"],
            end_time=end_time,
            total_duration=total_duration,
            profile_points=self.profile_points.copy(),
            device_stats=device_stats,
            optimization_summary=self._generate_optimization_summary()
        )
        
        # Store completed session
        self.completed_sessions.append(profile)
        
        # Clear current session
        self.current_session = None
        self.profile_points = []
        self.active_contexts = {}
        
        logger.info(f"Ended profiling session: {profile.session_id}")
        logger.info(f"Total duration: {total_duration:.3f}s, Sequences: {profile.total_sequences}")
        
        return profile
    
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling individual operations
        
        Args:
            operation_name: Name of the operation
            metadata: Additional metadata to store
        """
        if self.current_session is None:
            # If no session, just yield without profiling
            yield
            return
        
        # Start profiling
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()
        device = self.current_session["device"]
        
        try:
            yield
        finally:
            # End profiling
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()
            duration = end_time - start_time
            memory_delta = memory_after - memory_before
            
            # Create profile point
            point = ProfilePoint(
                name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                device=device,
                metadata=metadata or {}
            )
            
            self.profile_points.append(point)
    
    def record_sequence_processing(self, num_sequences: int, num_frames: int):
        """Record that sequences have been processed"""
        if self.current_session:
            self.current_session["total_sequences"] += num_sequences
            self.current_session["total_frames"] += num_frames
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not self.enable_memory_tracking:
            return 0.0
        
        try:
            # Try GPU memory first
            device_str = self.current_session.get("device", "cpu") if self.current_session else "cpu"
            
            if device_str.startswith("cuda"):
                return torch.cuda.memory_allocated() / (1024**2)
            elif device_str == "mps":
                # MPS doesn't have direct memory tracking, use process memory
                return psutil.Process().memory_info().rss / (1024**2)
            else:
                # CPU - use process memory
                return psutil.Process().memory_info().rss / (1024**2)
        except Exception:
            return 0.0
    
    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate summary of optimizations applied"""
        if not self.profile_points:
            return {}
        
        # Calculate operation statistics
        operation_stats = {}
        for point in self.profile_points:
            if point.name not in operation_stats:
                operation_stats[point.name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "total_memory": 0.0,
                    "avg_time": 0.0,
                    "avg_memory": 0.0
                }
            
            stats = operation_stats[point.name]
            stats["count"] += 1
            stats["total_time"] += point.duration
            stats["total_memory"] += point.memory_delta
        
        # Calculate averages
        for stats in operation_stats.values():
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
                stats["avg_memory"] = stats["total_memory"] / stats["count"]
        
        # Overall statistics
        total_operations = len(self.profile_points)
        total_operation_time = sum(p.duration for p in self.profile_points)
        
        return {
            "total_operations": total_operations,
            "total_operation_time": total_operation_time,
            "operation_stats": operation_stats,
            "avg_operation_time": total_operation_time / max(1, total_operations),
            "memory_efficiency": self._calculate_memory_efficiency()
        }
    
    def _calculate_memory_efficiency(self) -> Dict[str, float]:
        """Calculate memory efficiency metrics"""
        if not self.profile_points:
            return {}
        
        memory_deltas = [p.memory_delta for p in self.profile_points]
        
        return {
            "max_memory_increase": max(memory_deltas, default=0.0),
            "avg_memory_delta": np.mean(memory_deltas),
            "memory_variance": np.var(memory_deltas),
            "operations_with_memory_increase": sum(1 for d in memory_deltas if d > 0)
        }
    
    def compare_sessions(self, session1_id: str, session2_id: str) -> Dict[str, Any]:
        """
        Compare two profiling sessions
        
        Args:
            session1_id: First session ID
            session2_id: Second session ID
            
        Returns:
            Comparison results
        """
        session1 = self._find_session(session1_id)
        session2 = self._find_session(session2_id)
        
        if not session1 or not session2:
            return {"error": "One or both sessions not found"}
        
        # Calculate performance improvements
        time_improvement = (session1.total_duration - session2.total_duration) / session1.total_duration * 100
        
        # Throughput comparison (sequences per second)
        throughput1 = session1.total_sequences / session1.total_duration
        throughput2 = session2.total_sequences / session2.total_duration
        throughput_improvement = (throughput2 - throughput1) / throughput1 * 100
        
        # Memory efficiency comparison
        mem_stats1 = session1.optimization_summary.get("memory_efficiency", {})
        mem_stats2 = session2.optimization_summary.get("memory_efficiency", {})
        
        return {
            "session1": {
                "id": session1_id,
                "duration": session1.total_duration,
                "throughput": throughput1,
                "sequences": session1.total_sequences
            },
            "session2": {
                "id": session2_id,
                "duration": session2.total_duration,
                "throughput": throughput2,
                "sequences": session2.total_sequences
            },
            "improvements": {
                "time_reduction_percent": time_improvement,
                "throughput_increase_percent": throughput_improvement,
                "memory_efficiency": {
                    "max_memory_delta": mem_stats2.get("max_memory_increase", 0) - mem_stats1.get("max_memory_increase", 0),
                    "avg_memory_delta": mem_stats2.get("avg_memory_delta", 0) - mem_stats1.get("avg_memory_delta", 0)
                }
            }
        }
    
    def _find_session(self, session_id: str) -> Optional[InferenceProfile]:
        """Find session by ID"""
        for session in self.completed_sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Performance report
        """
        report = {
            "profiler_info": {
                "memory_tracking": self.enable_memory_tracking,
                "device_tracking": self.enable_device_tracking,
                "total_sessions": len(self.completed_sessions)
            },
            "sessions": []
        }
        
        for session in self.completed_sessions:
            session_data = asdict(session)
            # Convert profile points to dictionaries
            session_data["profile_points"] = [asdict(point) for point in session.profile_points]
            report["sessions"].append(session_data)
        
        # Add summary statistics
        if self.completed_sessions:
            durations = [s.total_duration for s in self.completed_sessions]
            throughputs = [s.total_sequences / s.total_duration for s in self.completed_sessions]
            
            report["summary"] = {
                "avg_duration": np.mean(durations),
                "avg_throughput": np.mean(throughputs),
                "best_throughput": max(throughputs),
                "total_sequences_processed": sum(s.total_sequences for s in self.completed_sessions),
                "total_frames_processed": sum(s.total_frames for s in self.completed_sessions)
            }
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report saved to {save_path}")
        
        return report


class DeviceMonitor:
    """Monitor device-specific performance metrics"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {}
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start device monitoring"""
        self.monitoring = True
        self.stats = {
            "gpu_utilization": [],
            "memory_usage": [],
            "temperatures": [],
            "timestamps": []
        }
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate summary statistics
        summary = {}
        for key, values in self.stats.items():
            if values and key != "timestamps":
                summary[f"{key}_avg"] = np.mean(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_min"] = np.min(values)
        
        return {
            "raw_stats": self.stats,
            "summary": summary
        }
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # GPU utilization (NVIDIA only for now)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.stats["gpu_utilization"].append(util.gpu)
                    
                    # Get memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_percent = (mem_info.used / mem_info.total) * 100
                    self.stats["memory_usage"].append(memory_percent)
                    
                    # Get temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, 0)  # NVML_TEMPERATURE_GPU = 0
                    self.stats["temperatures"].append(temp)
                    
                except (ImportError, Exception):
                    # pynvml not available or monitoring failed, skip
                    pass
                
                self.stats["timestamps"].append(timestamp)
                
                # Wait before next measurement
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Device monitoring error: {e}")
                break


# Global profiler instance
_global_profiler = None

def get_profiler(enable_memory_tracking: bool = True, 
                enable_device_tracking: bool = True) -> PerformanceProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enable_memory_tracking, enable_device_tracking)
    return _global_profiler

def profile_inference_session(session_id: str, model_type: str, device: str):
    """Context manager for profiling an entire inference session"""
    profiler = get_profiler()
    profiler.start_session(session_id, model_type, device)
    try:
        yield profiler
    finally:
        return profiler.end_session()
