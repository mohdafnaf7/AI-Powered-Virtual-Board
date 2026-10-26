"""
smoothing.py
-------------
Two well-established signal-processing techniques that turn raw, jittery
fingertip coordinates into strokes that actually feel like a marker on a
whiteboard rather than a shaky mouse cursor:

1. OneEuroFilter — an adaptive low-pass filter (Casiez et al., 2012) that
   smooths out camera/tracking jitter when the fingertip is nearly still
   (fine detail, writing) while automatically loosening up and tracking
   closely when the fingertip moves fast (broad strokes). A fixed-alpha
   moving average can't do both at once: tuned for steady hands it lags
   badly on fast motion, tuned for fast motion it stays jittery when still.

2. catmull_rom_segment — given 4 consecutive tracked points, generates a
   smooth curve through the middle two (using the outer two as tangent
   guides) instead of a straight line. This is what removes the
   "polygon"/faceted look you get from connecting raw sample points
   directly, especially on fast curves and corners.
"""

import math


class OneEuroFilter:
    """One Euro Filter for a single scalar signal (use two instances for
    x and y). min_cutoff controls steady-state smoothing (lower = smoother
    but more lag); beta controls how much the filter loosens up as speed
    increases (higher = snappier fast strokes)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, timestamp):
        if self.t_prev is None:
            self.t_prev = timestamp
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(timestamp - self.t_prev, 1e-6)

        # Derivative (speed) estimate, itself lightly low-pass filtered.
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Cutoff frequency adapts to speed: faster motion -> higher cutoff
        # -> less smoothing lag. Slower motion -> lower cutoff -> steadier.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat


def catmull_rom_point(p0, p1, p2, p3, t):
    """Evaluate a Catmull-Rom spline at parameter t in [0, 1], moving from
    p1 (t=0) to p2 (t=1), using p0/p3 as tangent-defining neighbors."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def catmull_rom_segment(p0, p1, p2, p3, steps=8):
    """Return `steps` interpolated points tracing the smooth curve from
    p1 up to and including p2 (excludes p1 itself, since callers already
    have that as their current pen position)."""
    return [catmull_rom_point(p0, p1, p2, p3, i / steps) for i in range(1, steps + 1)]