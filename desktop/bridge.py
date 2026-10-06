"""Allowlisted native bridge implementation pending.

Expose file/directory dialogs and necessary window actions only.
Never expose eval, arbitrary imports, shell execution or unrestricted reads.
Selected paths must still be validated by the backend.
"""
