"""
Unit tests for detector.exceptions module.
Tests exception hierarchy, imports, and inheritance.
"""

import sys
from pathlib import Path

# Test that all exceptions can be imported
try:
    from detector.exceptions import (
        DetectorError,
        PredictionError,
        InvalidImageError,
        ModelNotAvailableError,
        BackgroundFilterError,
        SessionError,
        ScanHistoryError,
    )
    print("✓ All exceptions imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


# Test that inference.py imports from exceptions (not defining duplicates)
try:
    from detector.inference import PredictionError as InferencePredictionError
    from detector.exceptions import PredictionError as ExceptionsPredictionError
    
    if InferencePredictionError is ExceptionsPredictionError:
        print("✓ inference.py imports PredictionError from exceptions.py (no duplicates)")
    else:
        print("✗ PredictionError defined in multiple places!")
        sys.exit(1)
except ImportError as e:
    print(f"✗ Failed to verify inference imports: {e}")
    sys.exit(1)


# Test exception hierarchy
print("\n--- Testing Exception Hierarchy ---")

# Test 1: PredictionError inherits from DetectorError
if issubclass(PredictionError, DetectorError):
    print("✓ PredictionError inherits from DetectorError")
else:
    print("✗ PredictionError should inherit from DetectorError")
    sys.exit(1)

# Test 2: InvalidImageError inherits from PredictionError
if issubclass(InvalidImageError, PredictionError):
    print("✓ InvalidImageError inherits from PredictionError")
else:
    print("✗ InvalidImageError should inherit from PredictionError")
    sys.exit(1)

# Test 3: ModelNotAvailableError inherits from PredictionError
if issubclass(ModelNotAvailableError, PredictionError):
    print("✓ ModelNotAvailableError inherits from PredictionError")
else:
    print("✗ ModelNotAvailableError should inherit from PredictionError")
    sys.exit(1)

# Test 4: BackgroundFilterError inherits from DetectorError
if issubclass(BackgroundFilterError, DetectorError):
    print("✓ BackgroundFilterError inherits from DetectorError")
else:
    print("✗ BackgroundFilterError should inherit from DetectorError")
    sys.exit(1)

# Test 5: SessionError inherits from DetectorError
if issubclass(SessionError, DetectorError):
    print("✓ SessionError inherits from DetectorError")
else:
    print("✗ SessionError should inherit from DetectorError")
    sys.exit(1)

# Test 6: ScanHistoryError inherits from DetectorError
if issubclass(ScanHistoryError, DetectorError):
    print("✓ ScanHistoryError inherits from DetectorError")
else:
    print("✗ ScanHistoryError should inherit from DetectorError")
    sys.exit(1)


# Test raising and catching exceptions
print("\n--- Testing Exception Raising and Catching ---")

# Test 7: Raise and catch InvalidImageError as PredictionError
try:
    raise InvalidImageError("Test image error")
except PredictionError as e:
    print(f"✓ InvalidImageError caught as PredictionError: {e}")
except Exception as e:
    print(f"✗ Failed to catch InvalidImageError: {e}")
    sys.exit(1)

# Test 8: Raise and catch ModelNotAvailableError as PredictionError
try:
    raise ModelNotAvailableError("Model files missing")
except PredictionError as e:
    print(f"✓ ModelNotAvailableError caught as PredictionError: {e}")
except Exception as e:
    print(f"✗ Failed to catch ModelNotAvailableError: {e}")
    sys.exit(1)

# Test 9: Raise and catch BackgroundFilterError as DetectorError
try:
    raise BackgroundFilterError("Rembg processing failed")
except DetectorError as e:
    print(f"✓ BackgroundFilterError caught as DetectorError: {e}")
except Exception as e:
    print(f"✗ Failed to catch BackgroundFilterError: {e}")
    sys.exit(1)

# Test 10: Raise and catch SessionError as DetectorError
try:
    raise SessionError("Session creation failed")
except DetectorError as e:
    print(f"✓ SessionError caught as DetectorError: {e}")
except Exception as e:
    print(f"✗ Failed to catch SessionError: {e}")
    sys.exit(1)

# Test 11: Raise and catch ScanHistoryError as DetectorError
try:
    raise ScanHistoryError("Database operation failed")
except DetectorError as e:
    print(f"✓ ScanHistoryError caught as DetectorError: {e}")
except Exception as e:
    print(f"✗ Failed to catch ScanHistoryError: {e}")
    sys.exit(1)

# Test 12: Catch InvalidImageError with generic Exception
try:
    raise InvalidImageError("Generic exception test")
except Exception as e:
    print(f"✓ InvalidImageError caught as Exception: {e}")


# Test that inference.py has no duplicate exception definitions
print("\n--- Checking for Duplicates in inference.py ---")

import inspect
from detector import inference

source = inspect.getsource(inference)
duplicate_keywords = [
    "class PredictionError",
    "class InvalidImageError",
    "class ModelNotAvailableError",
]

has_duplicates = False
for keyword in duplicate_keywords:
    if keyword in source:
        print(f"✗ Found duplicate definition: {keyword} in inference.py")
        has_duplicates = True

if not has_duplicates:
    print("✓ No duplicate exception definitions in inference.py")
else:
    sys.exit(1)


# Test exception message preservation
print("\n--- Testing Exception Messages ---")

test_messages = [
    ("InvalidImageError", InvalidImageError("Image too large")),
    ("ModelNotAvailableError", ModelNotAvailableError("dinov2_banana_disease.pth not found")),
    ("BackgroundFilterError", BackgroundFilterError("Rembg model load failed")),
    ("SessionError", SessionError("Invalid session ID")),
    ("ScanHistoryError", ScanHistoryError("Database connection error")),
]

for name, exc in test_messages:
    if str(exc):
        print(f"✓ {name} preserves message: '{exc}'")
    else:
        print(f"✗ {name} message not preserved")
        sys.exit(1)


print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Exceptions module is working correctly!")
print("="*60)
