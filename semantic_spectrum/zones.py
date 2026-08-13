"""
Zone definitions for the Phase 2 zone-aware blending pipeline.

Two modes are supported at runtime:
  standard      -- bilateral zones merged (hands, arms, legs).
                   No extra argument needed; suitable for symmetric motions.
  side_specific -- zones split by laterality (left_hand, right_hand, etc.).
                   Pass --zone_mode side_specific for asymmetric style transfer.

Joint index reference (HumanML3D / SMPL-H, 22 joints):
  0 Pelvis   1 L_Hip      2 R_Hip     3 Spine1
  4 L_Knee   5 R_Knee     6 Spine2    7 L_Ankle
  8 R_Ankle  9 Spine3    10 L_Toe    11 R_Toe
 12 Neck     13 L_Collar  14 R_Collar 15 Head
 16 L_Shoulder 17 R_Shoulder 18 L_Elbow 19 R_Elbow
 20 L_Wrist   21 R_Wrist
"""

from __future__ import annotations

_STANDARD: dict[str, list[int]] = {
    "head":  [12, 15],               # Neck, Head
    "torso": [0, 3, 6, 9],           # Pelvis, Spine1-3
    "arms":  [13, 14, 16, 17, 18, 19],  # Collars, Shoulders, Elbows
    "hands": [20, 21],               # Wrists
    "legs":  [1, 2, 4, 5, 7, 8],    # Hips, Knees, Ankles
    "feet":  [10, 11],               # Toes
}

_SIDE_SPECIFIC: dict[str, list[int]] = {
    "head":        [12, 15],
    "torso":       [0, 3, 6, 9],
    "left_arm":    [13, 16, 18],     # L_Collar, L_Shoulder, L_Elbow
    "right_arm":   [14, 17, 19],     # R_Collar, R_Shoulder, R_Elbow
    "left_hand":   [20],             # L_Wrist
    "right_hand":  [21],             # R_Wrist
    "left_leg":    [1, 4, 7],        # L_Hip, L_Knee, L_Ankle
    "right_leg":   [2, 5, 8],        # R_Hip, R_Knee, R_Ankle
    "feet":        [10, 11],
}

_MODES: dict[str, dict[str, list[int]]] = {
    "standard":     _STANDARD,
    "side_specific": _SIDE_SPECIFIC,
}


class ZoneConfig:
    """Immutable zone definition for a given mode."""

    def __init__(self, mode: str = "standard"):
        if mode not in _MODES:
            raise ValueError(f"Unknown zone mode '{mode}'. Choose from: {list(_MODES)}")
        self.mode  = mode
        self.zones: dict[str, list[int]] = dict(_MODES[mode])

    # ------------------------------------------------------------------
    @staticmethod
    def get_zones(mode: str = "standard") -> dict[str, list[int]]:
        """Return zone-to-joint-index mapping without instantiating."""
        if mode not in _MODES:
            raise ValueError(f"Unknown zone mode '{mode}'.")
        return dict(_MODES[mode])

    def zone_names(self) -> list[str]:
        return list(self.zones.keys())

    def joint_indices(self, zone: str) -> list[int]:
        if zone not in self.zones:
            raise KeyError(f"Zone '{zone}' not in mode '{self.mode}'.")
        return self.zones[zone]

    def __repr__(self) -> str:
        return f"ZoneConfig(mode={self.mode!r}, zones={list(self.zones)})"
