"""Run a Tinker script against a server that requires Modal proxy auth.

Every tinker.ServiceClient the script creates sends MODAL_PROXY_TOKEN_ID and
MODAL_PROXY_TOKEN_SECRET as Modal-Key and Modal-Secret headers, so scripts that
build their own client, such as tinker-cookbook's, work unchanged:

    python with_proxy_auth.py -m tinker_cookbook.recipes.chat_sl.train <args>
    python with_proxy_auth.py path/to/script.py <args>
"""

import inspect
import os
import runpy
import sys

import tinker


def proxy_headers():
    token_id = os.environ.get("MODAL_PROXY_TOKEN_ID")
    token_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not token_secret:
        return {}
    return {"Modal-Key": token_id, "Modal-Secret": token_secret}


def install():
    headers = proxy_headers()
    if not headers:
        sys.exit("Set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET.")
    init = tinker.ServiceClient.__init__
    # Tinker 0.36 renamed default_headers and ignores the old name.
    parameters = inspect.signature(init).parameters
    name = (
        "default_headers_overrides"
        if "default_headers_overrides" in parameters
        else "default_headers"
    )

    def init_with_proxy_auth(self, *args, **kwargs):
        kwargs[name] = {**headers, **(kwargs.get(name) or {})}
        init(self, *args, **kwargs)

    tinker.ServiceClient.__init__ = init_with_proxy_auth


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    install()
    if sys.argv[1] == "-m":
        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    else:
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
