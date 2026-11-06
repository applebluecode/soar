"""Device boundary: arms and calibration (the only layer importing pysoarm)."""

from soar.hardware.arm import Arm, arm_info, make_arm
from soar.hardware.calibration import (
    CalibrationResult,
    calibrate_with_rollback,
    calibration_path,
    default_output_path,
    save_calibration,
    sha256_of,
)

__all__ = [
    "Arm",
    "CalibrationResult",
    "arm_info",
    "calibrate_with_rollback",
    "calibration_path",
    "default_output_path",
    "make_arm",
    "save_calibration",
    "sha256_of",
]
