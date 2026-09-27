"""
Custom exceptions for the detector app.
Used for consistent error handling in views and inference pipeline.
"""


class DetectorError(Exception):
    """Base exception for all detector-related errors."""
    pass


class PredictionError(DetectorError):
    """Base class for user-facing prediction errors during inference."""
    pass


class InvalidImageError(PredictionError):
    """Raised when image validation fails (format, size, corrupted, etc.)"""
    pass


class ModelNotAvailableError(PredictionError):
    """Raised when required trained model files cannot be found."""
    pass


class BackgroundFilterError(DetectorError):
    """Raised when background filtering (Rembg) fails."""
    pass


class SessionError(DetectorError):
    """Raised when session creation or retrieval fails."""
    pass


class ScanHistoryError(DetectorError):
    """Raised when ScanHistory database operations fail."""
    pass
