# rstar_utils.py
import os, json
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.spacing import SpacingIndicator as Spacing

class RStarMetrics:
    """
    Fixed-R* metrics for 2D (F_st, S_st), both maximization in physical space.
    Looks for files in this priority (if `design` is set, e.g., 'DB'):
      1) ./R_star_{design}_normalized.csv    or  ./R_star_{design}_raw.csv
         ./normalization_minmax_{design}.json
      2) ./rstar/{design}/R_star_normalized.csv (or R_star_raw.csv)
         ./rstar/{design}/normalization_minmax.json
      3) ./R_star_normalized.csv (or R_star_raw.csv)
         ./normalization_minmax.json
    """
    def __init__(self, base_dir=".", design=None, files=None):
        self.base_dir = base_dir
        self.design = design

        # Resolve paths
        self.norm_path, self.raw_path, self.minmax_path = None, None, None
        if files:  # explicit override
            self.norm_path = os.path.join(base_dir, files.get("norm")) if files.get("norm") else None
            self.raw_path  = os.path.join(base_dir, files.get("raw"))  if files.get("raw")  else None
            self.minmax_path = os.path.join(base_dir, files.get("minmax")) if files.get("minmax") else None
        else:
            candidates = []
            if design:
                candidates += [
                    (f"R_star_{design}_normalized.csv", f"R_star_{design}_raw.csv", f"normalization_minmax_{design}.json"),
                    (os.path.join("rstar", design, "R_star_normalized.csv"),
                     os.path.join("rstar", design, "R_star_raw.csv"),
                     os.path.join("rstar", design, "normalization_minmax.json")),
                ]
            candidates += [("R_star_normalized.csv", "R_star_raw.csv", "normalization_minmax.json")]

            for norm, raw, mm in candidates:
                norm = os.path.join(base_dir, norm)
                raw  = os.path.join(base_dir, raw)
                mm   = os.path.join(base_dir, mm)
                if os.path.exists(mm) and (os.path.exists(norm) or os.path.exists(raw)):
                    self.norm_path, self.raw_path, self.minmax_path = norm, raw, mm
                    break

        if self.minmax_path is None:
            raise FileNotFoundError("Could not locate normalization_minmax*.json + R_star files.")

        # Load min/max (physical F_st, S_st)
        with open(self.minmax_path, "r") as f:
            mm = json.load(f)
        self.ref_min = np.array(mm["ref_min"], float)
        self.ref_max = np.array(mm["ref_max"], float)

        # Load R* in normalized MAX space or raw then normalize
        if self.norm_path and os.path.exists(self.norm_path):
            Rn_max = pd.read_csv(self.norm_path).to_numpy(float)  # in [0,1], MAX orientation
        else:
            R_raw = pd.read_csv(self.raw_path).to_numpy(float)    # physical units
            span = np.maximum(self.ref_max - self.ref_min, 1e-12)
            Rn_max = np.clip((R_raw - self.ref_min) / span, 0.0, 1.0)

        # Indicators expect MINIMIZATION; flip: min = 1 - max
        self.R_min = 1.0 - Rn_max
        self.hv = HV(ref_point=np.array([1.0, 1.0]))  # (1,1) is the worst in MIN space
        self.igd_plus = IGDPlus(self.R_min)
        self.spacing = Spacing()

    @staticmethod
    def I_to_J(front_I):
        """Your code minimizes I; physical is J = 1/I - 1 (F_st,S_st)."""
        I = np.maximum(np.asarray(front_I, float), 1e-12)
        return (1.0 / I) - 1.0

    def J_to_min(self, QJ):
        """Normalize physical (max) to [0,1], then flip to MIN space."""
        span = np.maximum(self.ref_max - self.ref_min, 1e-12)
        Qmax = np.clip((np.asarray(QJ, float) - self.ref_min) / span, 0.0, 1.0)
        return 1.0 - Qmax

    def compute_metrics_from_I(self, front_I, generation=None, gen_time=None, survival_stats=None):
        if front_I is None or len(front_I) == 0:
            return {"generation": generation, "generation_time": gen_time,
                    "hypervolume": float("nan"), "igd_plus": float("nan"),
                    "spacing": float("nan"), "crossover_survival": survival_stats}
        QJ   = self.I_to_J(front_I)
        Qmin = self.J_to_min(QJ)
        hv_val = float(self.hv.do(Qmin)) if len(Qmin) else float("nan")
        igd_val = float(self.igd_plus.do(Qmin)) if len(Qmin) else float("nan")
        try:
            spc_val = float(self.spacing.do(Qmin)) if Qmin.shape[0] >= 2 else float("nan")
        except Exception:
            spc_val = float("nan")
        return {"generation": generation, "generation_time": gen_time,
                "hypervolume": hv_val, "igd_plus": igd_val, "spacing": spc_val,
                "crossover_survival": survival_stats}
