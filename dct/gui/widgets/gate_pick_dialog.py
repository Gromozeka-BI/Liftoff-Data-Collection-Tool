"""Small dialog for picking a gate when adding a gate event in replay editor."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout,
)


class GatePickDialog(QDialog):
    """Shows a list of gates from the track; user picks one (or None).

    gates: list of dicts with 'id' and optionally 'name'/'label'
    Returns selected gate_id (int) or -1 if "No gate / any gate" selected.
    """

    def __init__(self, gates: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select gate")
        self.setMinimumWidth(280)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select gate ID (optional):"))

        self._list = QListWidget()
        # First item: no specific gate
        item0 = QListWidgetItem("— No specific gate (gate_id = -1)")
        item0.setData(Qt.ItemDataRole.UserRole, -1)
        self._list.addItem(item0)

        for g in gates:
            gid = g.get("id", -1)
            name = g.get("name") or g.get("label") or f"Gate {gid}"
            sf = "  ★ S/F" if g.get("is_start_finish") else ""
            item = QListWidgetItem(f"{name}  (id={gid}){sf}")
            item.setData(Qt.ItemDataRole.UserRole, gid)
            self._list.addItem(item)

        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(self.accept)
        lay.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected_gate_id(self) -> int:
        item = self._list.currentItem()
        if item is None:
            return -1
        return item.data(Qt.ItemDataRole.UserRole)
