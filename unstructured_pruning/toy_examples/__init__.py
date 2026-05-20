"""Toy experiments for the unstructured-pruning theory.

Each module here sets up a minimal, analytically tractable network whose
pruning behaviour can be compared directly against a closed-form theoretical
prediction from the paper. The first such module
(``binary_classification``) implements the Appendix D scalar binary task
and checks the prediction :math:`A(s) = \\Phi(c\\sqrt{s/(1-s)})` of
equation (D17).
"""
