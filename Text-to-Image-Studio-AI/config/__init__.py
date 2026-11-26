import sys


def _hide_outdated_torchao(min_version: str = "0.16.0") -> None:
    """Colab ships an old optional 'torchao' that newer peft/diffusers reject with an ImportError when a LoRA
    adapter is attached. This project never uses torchao, so an outdated copy is hidden from them.
    Runs whenever any part of the project is imported (the web app, the tests, notebooks)."""
    try:
        from importlib import metadata
        from packaging.version import Version
        if "torchao" not in sys.modules and Version(metadata.version("torchao")) < Version(min_version):
            sys.modules["torchao"] = None
    except Exception:
        pass


_hide_outdated_torchao()
