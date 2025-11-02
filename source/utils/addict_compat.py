"""Minimal fallback implementation of addict.Dict used by PointTransformerV3."""

from __future__ import annotations


class Dict(dict):
    """A dict subclass that exposes keys as attributes and auto-wraps mappings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure nested mappings become Dict instances immediately
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Dict):
                super().__setitem__(key, Dict(value))

    # Attribute access -----------------------------------------------------
    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Dict):
            value = Dict(value)
            super().__setitem__(item, value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    # Mapping overrides ----------------------------------------------------
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Dict):
            value = Dict(value)
        super().__setitem__(key, value)

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    def copy(self):
        return Dict(super().copy())

