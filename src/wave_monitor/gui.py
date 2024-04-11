import sys

from PySide6.QtWidgets import QApplication

from wave_monitor import MonitorWindow, config_log


def start():
    config_log()
    app = QApplication(sys.argv)
    _ = MonitorWindow()
    sys.exit(app.exec())
