"""Utility helpers shared across the application."""

import math


def clamp(value, lower, upper):
    """Clamp numeric value between lower and upper bounds."""
    return max(lower, min(upper, value))


def ang_norm(angle):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))
