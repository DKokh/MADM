"""MADM -- multi-assay antibody developability models.

One pipeline from a table of VH/VL sequences to trained models and predictions,
across a panel of developability assays.

    from madm import Config, Ensemble

    ens = Ensemble("runs/SMP_feat+ablang_MT_cluster_ab12cd34/bundle.pt")
    ens.streams          # what inputs this model needs

`Config`, `madm.table`, `madm.metrics` and `madm.features` are pure numpy/pandas and
import without torch. `FvModel` and `Ensemble` are resolved on first use, so
reading a config, validating a table or aligning features to the IMGT frame does
not require a deep-learning install.
"""
__version__ = "0.1.0"

from .config import Config          # noqa: F401

__all__ = ["Config", "FvModel", "Ensemble", "__version__"]


def __getattr__(name):              # PEP 562: defer the torch-dependent imports
    if name == "FvModel":
        from .model import FvModel
        return FvModel
    if name == "Ensemble":
        from .predict import Ensemble
        return Ensemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
