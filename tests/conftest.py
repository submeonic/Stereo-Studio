"""Run GUI tests headlessly, including when an individual test module is selected."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
