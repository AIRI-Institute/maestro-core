import logging
from functools import lru_cache

from .entrypoints_config import EntrypointsConfig
from .entrypoints import create_entrypoint


logger = logging.getLogger(__name__)


class EntrypointsAccessor:
    def __init__(self, cfg: EntrypointsConfig):
        self.cfg = cfg

    def _warmup_if_needed(self):
        if not self.cfg.warmup:
            return
        assert self.cfg.default_entrypoint_key in self.cfg.entrypoints
        for ec_key in self.cfg.entrypoints.keys():
            self.get(ec_key)

    @lru_cache(None)
    def get_default(self):
        default_key = self.cfg.default_entrypoint_key
        entrypoint = self.get(default_key)
        return entrypoint

    @lru_cache(None)
    def get(self, entrypoint_key: str):
        ep = self.cfg.entrypoints.get(entrypoint_key)
        if ep is None:
            default_key = self.cfg.default_entrypoint_key
            logger.warning(f"Not found entrypoint for key={entrypoint_key}, fallback to {default_key}")
            ep = self.cfg.entrypoints[default_key]
        entrypoint = create_entrypoint(ep.name, ep.args)
        return entrypoint

    def __getitem__(self, key):
        return self.get(key)

    def clear(self):
        self.get.cache_clear()
