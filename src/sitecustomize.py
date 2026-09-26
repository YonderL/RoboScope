"""Process-wide hooks for this experiment.

DataLoader workers and LIBERO AsyncVectorEnv workers are new interpreters.
A cache set only in the training process is invisible to them.
"""

import os

cache = os.environ.get("ROBOSCOPE_IMAGE_CACHE")
if cache:
    from roboscope.data.smolvla_image_cache import install

    install(cache)

# get_assets_path() ignores LIBERO_CONFIG_PATH and falls back to the package tree.
# That tree has no scene XMLs. Sync eval ran inside the trainer, which assigns the
# cache itself; async eval workers have to do it at import time.
_config_dir = os.environ.get("LIBERO_CONFIG_PATH")
if _config_dir:
    _config_file = os.path.join(_config_dir, "config.yaml")
    _assets = None
    if os.path.isfile(_config_file):
        for _line in open(_config_file):
            if _line.startswith("assets:"):
                _assets = _line.split(":", 1)[1].strip()
                break
    if _assets and os.path.isdir(_assets):
        import libero.libero as _libero_api

        _libero_api._assets_path_cache = _assets
