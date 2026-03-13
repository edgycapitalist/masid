"""Architecture registry — maps keys to concrete implementations."""

from __future__ import annotations

from masid.architectures import BaseArchitecture
from masid.architectures.iamd import IAMDArchitecture
from masid.architectures.irm import IRMArchitecture
from masid.architectures.jro import JROArchitecture

_REGISTRY: dict[str, type[BaseArchitecture]] = {
    "irm": IRMArchitecture,
    "jro": JROArchitecture,
    "iamd": IAMDArchitecture,
}


def get_architecture(key: str, **kwargs: object) -> BaseArchitecture:
    """Instantiate an architecture by its short key.

    Parameters
    ----------
    key : str
        One of ``"irm"``, ``"jro"``, ``"iamd"``.
    **kwargs
        Passed to the architecture constructor (e.g. ``domain`` for IAMD).
    """
    if key not in _REGISTRY:
        raise ValueError(f"Unknown architecture {key!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**kwargs)
