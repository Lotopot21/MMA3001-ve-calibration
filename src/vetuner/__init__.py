"""Offline volumetric efficiency table calibration from engine sensor logs.

Estimates a corrected VE table and the resulting injector pulse widths from
logged air-fuel ratio data, and provides a forward engine model for generating
synthetic logs with known ground truth.
"""