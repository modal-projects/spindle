# Upstreaming a config change

For a new model config or performance tuning of an existing one, open a PR
directly. For bigger changes, discuss first in Slack (https://modal.com/slack).
For bugs or missing models/features, open an issue
(https://github.com/modal-projects/spindle/issues).

## Steps

1. Reproduce and fix on a local subclass:

   ```bash
   spindle config init --preset <name> > my_model.py
   # edit my_model.py
   spindle config validate my_model.py
   spindle deploy my_model.py
   ```

   Run a short real training/sampling check against it.

2. Port the change into `src/spindle/configs/<name>.py` in a fork.

3. Run `uv run --no-sync pytest -q` and `spindle config validate` on the
   edited config.

4. Open the PR with evidence: config name, what changed, before/after
   behavior or throughput, hardware used.
