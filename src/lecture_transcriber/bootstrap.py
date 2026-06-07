"""Application composition root.

Wiring of concrete implementations lives here. The composition root is the only
place that is allowed to import concrete adapters; services and domain code
receive their dependencies through constructor parameters and must not import
this module.
"""
