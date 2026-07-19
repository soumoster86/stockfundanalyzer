"""Tab renderers for the main Streamlit app."""

from ui.tabs.report import render_report
from ui.tabs.ranking import render_ranking
from ui.tabs.compare import render_compare
from ui.tabs.sector import render_sector
from ui.tabs.train import render_train

__all__ = [
    "render_report",
    "render_ranking",
    "render_compare",
    "render_sector",
    "render_train",
]
