"""
Logging utilities for capturing and managing terminal output
"""

import sys
import time
import threading
from typing import List, Dict, Any


# Store captured logs
captured_logs: List[Dict[str, Any]] = []
log_lock = threading.Lock()


class LogCapture:
    """
    Custom stream wrapper to capture stdout/stderr for API access
    while preserving original terminal output
    """

    def __init__(self, original_stream, level: str):
        self.original_stream = original_stream
        self.level = level

    def write(self, message: str) -> None:
        """Write message to original stream and capture for API"""
        # Write to original stream and ensure it's flushed immediately
        self.original_stream.write(message)
        self.original_stream.flush()

        # Capture for API if it's not just whitespace
        if message.strip():
            # Filter out polling requests to reduce spam
            if "GET /api/logs" in message or "GET /api/system/metrics" in message:
                return

            # Get current thread info for debugging
            thread_name = threading.current_thread().name
            timestamp = time.strftime("%H:%M:%S")

            # Add thread info if not main thread
            if thread_name != "MainThread":
                message_with_thread = f"[{thread_name}] {message.strip()}"
            else:
                message_with_thread = message.strip()

            log_entry = {
                "timestamp": timestamp,
                "level": self.level,
                "message": message_with_thread,
            }

            # Lock for thread safety
            with log_lock:
                captured_logs.append(log_entry)
                # Keep only last 500 logs to prevent memory issues
                if len(captured_logs) > 500:
                    captured_logs.pop(0)

    def flush(self) -> None:
        """Flush the original stream"""
        self.original_stream.flush()

    def isatty(self) -> bool:
        """Check if the original stream is a terminal"""
        return getattr(self.original_stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        """Delegate any other attributes to the original stream"""
        return getattr(self.original_stream, name)


def setup_log_capture() -> None:
    """Initialize log capture by replacing stdout and stderr"""
    sys.stdout = LogCapture(sys.stdout, "info")
    sys.stderr = LogCapture(sys.stderr, "error")


def get_logs(since: int = 0) -> Dict[str, Any]:
    """
    Get captured logs since a given index

    Args:
        since: Starting index for log retrieval

    Returns:
        Dict containing logs and total count
    """
    with log_lock:
        if since >= len(captured_logs):
            return {"logs": [], "total": len(captured_logs)}
        return {"logs": captured_logs[since:], "total": len(captured_logs)}
