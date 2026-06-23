"""flash_osiris -- convert FLASH MHD output into initialized OSIRIS PIC input decks.

Two modules:
  * ``yt_plugin``  -- the yt plugin (symlink to ~/.config/yt/my_plugins.py); the SINGLE
                      place that knows how OSIRIS ion populations are defined from the
                      FLASH materials (target / chamber / ...), not from ion mass.
  * ``generator``  -- reads a run.yaml, loads FLASH via the plugin, and renders the
                      OSIRIS deck + python-init script + interpolation slices.

Importing this package is side-effect free; ``generator`` calls ``yt.enable_plugins()``
when it is imported/run, so import it only inside the osiris generation environment.
"""

__all__ = ["generator", "yt_plugin"]
