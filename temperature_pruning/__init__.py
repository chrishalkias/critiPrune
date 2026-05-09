"""Empirical test of the diluted Curie-Weiss prediction T_c(p) = J_0 p.

Sub-module that loads trained FCNetwork checkpoints from the existing
``unstructured_pruning`` suite, applies a Gaussian-weight-noise temperature
knob and a random Bernoulli pruning mask, and checks whether the critical
pruning fraction p_c is linear in the temperature sigma.
"""
