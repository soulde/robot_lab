"""Training-specific Viser startup controls."""

from isaaclab_visualizers.viser.viser_visualizer import ViserVisualizer


class _GuiProxy:
    def __init__(self, gui):
        self.gui = gui

    def __getattr__(self, name):
        return getattr(self.gui, name)

    def add_button(self, label, *args, **kwargs):
        if label == "Pause Rendering":
            label = "Resume Rendering"
            kwargs["color"] = "orange"
        return self.gui.add_button(label, *args, **kwargs)


class _ServerProxy:
    def __init__(self, server):
        self.server = server
        self.gui = _GuiProxy(server.gui)

    def __getattr__(self, name):
        return getattr(self.server, name)


class StartPausedViserVisualizer(ViserVisualizer):
    """Start the web server and controls with scene updates paused."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._paused_rendering = True

    def _setup_isaaclab_sidebar(self, server):
        super()._setup_isaaclab_sidebar(_ServerProxy(server))
        print("[VISER] Rendering starts paused; training continues. Click Resume Rendering to display it.", flush=True)
