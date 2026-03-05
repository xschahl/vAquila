"""Façade de compatibilité pour les implémentations de commandes CLI."""

from vaquila.commands.cache import cmd_list_models, cmd_rm_model
from vaquila.commands.run import cmd_run
from vaquila.commands.system import cmd_doctor, cmd_infer, cmd_ps, cmd_rebalance, cmd_stop

__all__ = [
    "cmd_list_models",
    "cmd_rm_model",
    "cmd_run",
    "cmd_ps",
    "cmd_stop",
    "cmd_rebalance",
    "cmd_doctor",
    "cmd_infer",
]
