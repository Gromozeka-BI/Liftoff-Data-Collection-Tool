"""Experimental PnP solver prototype inspired by the CTU gate localization code."""

__all__ = ["PnPResult", "PnPSolver2"]


def __getattr__(name: str):
    if name in __all__:
        from .pnp_solver_2 import PnPResult, PnPSolver2

        return {"PnPResult": PnPResult, "PnPSolver2": PnPSolver2}[name]
    raise AttributeError(name)
