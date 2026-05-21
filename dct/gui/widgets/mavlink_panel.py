"""Shared Mavlink sidebar block used by Setup (Record) and Replay pages."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MAV_SRC_CAMKF = "camkf"
MAV_SRC_KF = "kf"

_MAV_POINT_COL_W = 22
_MAV_FIELD_MIN_W = {"lat": 48, "lon": 48, "alt": 46}
_MAV_FIELD_FLOOR = {"lat": 40, "lon": 40, "alt": 40}
_MAV_COL_SHARE = (0.34, 0.34, 0.32)
_MAV_ROW_SPACING = 4
_MAV_ROW_GAP = 4
_MAV_GROUP_HPAD = 12
_MAV_SPIN_PORT_W = 58
_MAV_SPIN_ID_W = 44
# Row layout (not grid) — normal spinbox height without vertical overlap.
_ANCHOR_SPIN_H = 28


def _make_anchor_spinbox(
    field: str,
    decimals: int,
    value: float,
    min_val: float,
    max_val: float,
) -> QDoubleSpinBox:
    spn = QDoubleSpinBox()
    spn.setRange(min_val, max_val)
    spn.setDecimals(decimals)
    spn.setSingleStep(0.000001 if field != "alt" else 0.5)
    spn.setValue(value)
    spn.setFixedHeight(_ANCHOR_SPIN_H)
    spn.setFixedWidth(_MAV_FIELD_MIN_W[field])
    spn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return spn


def _add_anchor_header_row(layout: QVBoxLayout) -> None:
    row = QHBoxLayout()
    row.setSpacing(_MAV_ROW_SPACING)
    corner = QLabel("")
    corner.setFixedWidth(_MAV_POINT_COL_W)
    row.addWidget(corner)
    for key, title in (("lat", "Lat"), ("lon", "Lon"), ("alt", "Alt")):
        lbl = QLabel(title)
        lbl.setFixedWidth(_MAV_FIELD_MIN_W[key])
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(lbl)
    row.addStretch()
    layout.addLayout(row)


def mavlink_panel_min_width() -> int:
    """Minimum width for Point + Lat + Lon + Alt (no horizontal clip)."""
    return (
        _MAV_POINT_COL_W
        + sum(_MAV_FIELD_MIN_W.values())
        + 3 * _MAV_ROW_SPACING
        + _MAV_GROUP_HPAD
    )


def _column_widths_for_grid(grid_inner_w: int) -> dict[str, int]:
    """Lat / Lon / Alt — roughly equal; spare width goes to trailing stretch."""
    avail = grid_inner_w - _MAV_POINT_COL_W - 3 * _MAV_ROW_SPACING
    avail = max(avail, sum(_MAV_FIELD_FLOOR.values()))
    lat = max(_MAV_FIELD_FLOOR["lat"], int(avail * _MAV_COL_SHARE[0]))
    lon = max(_MAV_FIELD_FLOOR["lon"], int(avail * _MAV_COL_SHARE[1]))
    alt = max(_MAV_FIELD_FLOOR["alt"], int(avail * _MAV_COL_SHARE[2]))
    total = lat + lon + alt
    if total > avail:
        scale = avail / total
        lat = max(_MAV_FIELD_FLOOR["lat"], int(lat * scale))
        lon = max(_MAV_FIELD_FLOOR["lon"], int(lon * scale))
        alt = max(_MAV_FIELD_FLOOR["alt"], avail - lat - lon)
    return {"lat": lat, "lon": lon, "alt": alt}


def _update_anchor_header_widths(
    anchors_wrap: QWidget,
    widths: dict[str, int],
) -> None:
    lay = anchors_wrap.layout()
    if lay is None or lay.count() == 0:
        return
    row_lay = lay.itemAt(0).layout()
    if row_lay is None:
        return
    for idx, field in enumerate(("lat", "lon", "alt")):
        item = row_lay.itemAt(idx + 1)
        w = item.widget() if item else None
        if w is not None:
            w.setFixedWidth(widths[field])


def _set_anchor_spin_widths(
    panel: MavlinkPanel,
    widths: dict[str, int],
) -> None:
    for fields in panel.spn_anchors.values():
        for field, spn in fields.items():
            spn.setFixedWidth(widths[field])
    _update_anchor_header_widths(panel.anchors_wrap, widths)


def fit_mavlink_panel_to_width(panel: MavlinkPanel, box_width: int) -> None:
    """Fit Mavlink group to sidebar content width (same as Session config blocks)."""
    box_width = max(160, int(box_width))
    panel.box.setMinimumWidth(0)
    panel.box.setMaximumWidth(box_width)
    panel.box.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Minimum,
    )
    grid_inner = max(120, box_width - _MAV_GROUP_HPAD)
    widths = _column_widths_for_grid(grid_inner)
    _set_anchor_spin_widths(panel, widths)
    panel.spn_port.setFixedWidth(_MAV_SPIN_PORT_W)
    panel.spn_sysid.setFixedWidth(_MAV_SPIN_ID_W)
    panel.spn_compid.setFixedWidth(_MAV_SPIN_ID_W)


def mavlink_anchors_min_height() -> int:
    """Minimum height for the Lat/Lon/Alt block (header + 3 data rows)."""
    rows = 4
    return rows * _ANCHOR_SPIN_H + (rows - 1) * _MAV_ROW_GAP + 6


def finalize_mavlink_panel(panel: MavlinkPanel) -> None:
    """Keep Mavlink from being vertically squeezed (Setup scroll in Record)."""
    panel.box.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Minimum,
    )
    panel.anchors_wrap.setMinimumHeight(mavlink_anchors_min_height())
    panel.anchors_wrap.adjustSize()
    panel.box.adjustSize()
    panel.box.setMinimumHeight(panel.box.minimumSizeHint().height())


def apply_mavlink_panel_layout(
    panel: MavlinkPanel,
    content_width: int | None = None,
) -> None:
    """Same post-build sizing for Record and Replay."""
    finalize_mavlink_panel(panel)
    if content_width is not None and content_width > 0:
        fit_mavlink_panel_to_width(panel, content_width)


@dataclass
class MavlinkPanel:
    box: QGroupBox
    anchors_wrap: QWidget
    combo_source: QComboBox
    chk_enabled: QCheckBox
    txt_host: QLineEdit
    spn_port: QSpinBox
    spn_sysid: QSpinBox
    spn_compid: QSpinBox
    spn_anchors: dict[str, dict[str, QDoubleSpinBox]]
    lbl_bounds: QLabel


def build_mavlink_panel(
    mav_settings: dict,
    *,
    bounds_default: str,
    on_changed: Callable[[], None] | None = None,
) -> MavlinkPanel:
    """Build the standard Mavlink group (same layout in Record and Replay)."""
    mav_box = QGroupBox("Mavlink")
    mav_lay = QVBoxLayout(mav_box)
    mav_lay.setSpacing(4)

    source_row = QHBoxLayout()
    source_row.setSpacing(4)
    source_row.addWidget(QLabel("Source"))
    combo_source = QComboBox()
    combo_source.addItem("CamKF", userData=MAV_SRC_CAMKF)
    combo_source.addItem("KF", userData=MAV_SRC_KF)
    saved_source = str(mav_settings.get("source", MAV_SRC_CAMKF))
    idx = combo_source.findData(saved_source)
    if idx >= 0:
        combo_source.setCurrentIndex(idx)
    source_row.addWidget(combo_source, stretch=1)
    mav_lay.addLayout(source_row)

    chk_enabled = QCheckBox("Enable UDP telemetry")
    chk_enabled.setChecked(bool(mav_settings.get("enabled", False)))
    mav_lay.addWidget(chk_enabled)

    endpoint_row = QHBoxLayout()
    endpoint_row.setSpacing(4)
    endpoint_row.addWidget(QLabel("Host"))
    txt_host = QLineEdit(str(mav_settings.get("host", "127.0.0.1")))
    endpoint_row.addWidget(txt_host, stretch=1)
    endpoint_row.addWidget(QLabel("Port"))
    spn_port = QSpinBox()
    spn_port.setRange(1, 65535)
    spn_port.setValue(int(mav_settings.get("port", 14550)))
    spn_port.setFixedWidth(_MAV_SPIN_PORT_W)
    endpoint_row.addWidget(spn_port)
    mav_lay.addLayout(endpoint_row)

    ids_row = QHBoxLayout()
    ids_row.setSpacing(4)
    ids_row.addWidget(QLabel("Sys"))
    spn_sysid = QSpinBox()
    spn_sysid.setRange(1, 255)
    spn_sysid.setValue(int(mav_settings.get("system_id", 1)))
    spn_sysid.setFixedWidth(_MAV_SPIN_ID_W)
    ids_row.addWidget(spn_sysid)
    ids_row.addWidget(QLabel("Comp"))
    spn_compid = QSpinBox()
    spn_compid.setRange(1, 255)
    spn_compid.setValue(int(mav_settings.get("component_id", 1)))
    spn_compid.setFixedWidth(_MAV_SPIN_ID_W)
    ids_row.addWidget(spn_compid)
    ids_row.addStretch()
    mav_lay.addLayout(ids_row)

    anchors = mav_settings.get("anchors", {})
    spn_anchors: dict[str, dict[str, QDoubleSpinBox]] = {}
    anchors_wrap = QWidget()
    anchors_lay = QVBoxLayout(anchors_wrap)
    anchors_lay.setContentsMargins(0, 0, 0, 0)
    anchors_lay.setSpacing(_MAV_ROW_GAP)
    _add_anchor_header_row(anchors_lay)
    for key, title in [("origin", "0.0"), ("x", "X.0"), ("z", "0.Z")]:
        saved = anchors.get(key, {}) if isinstance(anchors, dict) else {}
        spn_anchors[key] = {}
        row = QHBoxLayout()
        row.setSpacing(_MAV_ROW_SPACING)
        pt_lbl = QLabel(title)
        pt_lbl.setFixedWidth(_MAV_POINT_COL_W)
        row.addWidget(pt_lbl)
        for field, decimals, default, min_val, max_val in [
            ("lat", 7, 0.0, -90.0, 90.0),
            ("lon", 7, 0.0, -180.0, 180.0),
            ("alt", 2, 0.0, -1000.0, 10000.0),
        ]:
            spn = _make_anchor_spinbox(
                field,
                decimals,
                float(saved.get(field, default)),
                min_val,
                max_val,
            )
            row.addWidget(spn)
            spn_anchors[key][field] = spn
        row.addStretch()
        anchors_lay.addLayout(row)
    mav_lay.addWidget(anchors_wrap)

    lbl_bounds = QLabel(bounds_default)
    lbl_bounds.setProperty("role", "dim")
    lbl_bounds.setWordWrap(True)
    mav_lay.addWidget(lbl_bounds)

    panel = MavlinkPanel(
        box=mav_box,
        anchors_wrap=anchors_wrap,
        combo_source=combo_source,
        chk_enabled=chk_enabled,
        txt_host=txt_host,
        spn_port=spn_port,
        spn_sysid=spn_sysid,
        spn_compid=spn_compid,
        spn_anchors=spn_anchors,
        lbl_bounds=lbl_bounds,
    )
    finalize_mavlink_panel(panel)

    if on_changed is not None:
        combo_source.currentIndexChanged.connect(on_changed)
        chk_enabled.stateChanged.connect(on_changed)
        txt_host.editingFinished.connect(on_changed)
        spn_port.valueChanged.connect(on_changed)
        spn_sysid.valueChanged.connect(on_changed)
        spn_compid.valueChanged.connect(on_changed)
        for fields in spn_anchors.values():
            for spn in fields.values():
                spn.valueChanged.connect(on_changed)

    return panel


def read_mavlink_settings(panel: MavlinkPanel) -> dict:
    anchors: dict[str, dict[str, float]] = {}
    for key, fields in panel.spn_anchors.items():
        anchors[key] = {field: float(spn.value()) for field, spn in fields.items()}
    return {
        "source": panel.combo_source.currentData() or MAV_SRC_CAMKF,
        "enabled": bool(panel.chk_enabled.isChecked()),
        "host": panel.txt_host.text().strip() or "127.0.0.1",
        "port": int(panel.spn_port.value()),
        "system_id": int(panel.spn_sysid.value()),
        "component_id": int(panel.spn_compid.value()),
        "anchors": anchors,
    }
