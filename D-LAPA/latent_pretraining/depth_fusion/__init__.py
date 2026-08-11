__all__ = ["DepthFusionConfig", "DepthFusionPolicy"]


def __getattr__(name):
    if name in __all__:
        from latent_pretraining.depth_fusion.model import DepthFusionConfig, DepthFusionPolicy

        exports = {
            "DepthFusionConfig": DepthFusionConfig,
            "DepthFusionPolicy": DepthFusionPolicy,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
