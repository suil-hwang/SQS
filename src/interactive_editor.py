# interactive_editor.py
import argparse

from ui.controller import SQSController
from ui.model import SQSModel
from ui.panel import EditorPanel
from ui.view import SceneView
from ui.visual_config import VisualizationConfig


def main():
    parser = argparse.ArgumentParser(description="SQS Interactive Editor")
    parser.parse_args()

    model = SQSModel()

    config = VisualizationConfig()

    view = SceneView(config)
    controller = SQSController(model, view)
    view.initialize_polyscope()
    controller.rebuild_view()
    view.show(EditorPanel(controller).render)

if __name__ == "__main__":
    main()
