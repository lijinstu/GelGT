import numpy as np
import torch

class RegressionTargetTransform:

    def __init__(self, mode, mean, std, y_min, y_max, clamp_min, clamp_max, eps=1e-06):
        self.mode = mode
        self.mean = float(mean)
        self.std = float(std) if float(std) > eps else 1.0
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.clamp_min = None if clamp_min is None else float(clamp_min)
        self.clamp_max = None if clamp_max is None else float(clamp_max)
        self.eps = float(eps)

    @classmethod
    def from_labels(cls, labels, mode='none', clamp_mode='percentile', clamp_percentiles=(2.0, 98.0), eps=1e-06):
        labels = np.asarray(labels, dtype=np.float64)
        finite = labels[np.isfinite(labels)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=np.float64)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if mode == 'log1p':
            z = np.log1p(np.maximum(finite, 0.0))
        elif mode == 'asinh':
            z = np.arcsinh(finite)
        elif mode == 'logit':
            (lo, hi) = (min(0.0, y_min), max(1.0, y_max))
            scaled = (np.clip(finite, lo, hi) - lo) / max(hi - lo, eps)
            scaled = np.clip(scaled, eps, 1.0 - eps)
            z = np.log(scaled / (1.0 - scaled))
            (y_min, y_max) = (lo, hi)
        elif mode == 'standardize':
            z = finite
        elif mode == 'none':
            z = finite
        else:
            raise ValueError(f'Unsupported regression transform: {mode}')
        mean = float(np.mean(z))
        std = float(np.std(z))
        if clamp_mode == 'percentile':
            (clamp_min, clamp_max) = np.percentile(finite, clamp_percentiles)
        elif clamp_mode == 'minmax':
            (clamp_min, clamp_max) = (y_min, y_max)
        elif clamp_mode == 'none':
            (clamp_min, clamp_max) = (None, None)
        else:
            raise ValueError(f'Unsupported regression clamp: {clamp_mode}')
        return cls(mode, mean, std, y_min, y_max, clamp_min, clamp_max, eps)

    def _tensor(self, value, like):
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def forward(self, y):
        if self.mode == 'log1p':
            z = torch.log1p(torch.clamp(y, min=0.0))
        elif self.mode == 'asinh':
            z = torch.asinh(y)
        elif self.mode == 'logit':
            lo = self._tensor(self.y_min, y)
            hi = self._tensor(self.y_max, y)
            scaled = (torch.clamp(y, min=self.y_min, max=self.y_max) - lo) / torch.clamp(hi - lo, min=self.eps)
            scaled = torch.clamp(scaled, min=self.eps, max=1.0 - self.eps)
            z = torch.logit(scaled)
        else:
            z = y
        if self.mode in {'standardize', 'log1p', 'asinh', 'logit'}:
            return (z - self._tensor(self.mean, y)) / self._tensor(self.std, y)
        return z

    def inverse(self, z):
        if self.mode in {'standardize', 'log1p', 'asinh', 'logit'}:
            raw = z * self._tensor(self.std, z) + self._tensor(self.mean, z)
        else:
            raw = z
        if self.mode == 'log1p':
            y = torch.expm1(raw)
        elif self.mode == 'asinh':
            y = torch.sinh(raw)
        elif self.mode == 'logit':
            y = self._tensor(self.y_min, z) + torch.sigmoid(raw) * self._tensor(self.y_max - self.y_min, z)
        else:
            y = raw
        return self.clamp(y)

    def clamp(self, y):
        if self.clamp_min is None or self.clamp_max is None:
            return y
        return torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

    def jacobian_weight(self, y):
        pass
        if self.mode == 'log1p':
            w = torch.clamp(y + 1.0, min=self.eps)
        elif self.mode == 'asinh':
            w = torch.sqrt(y * y + 1.0)
        elif self.mode == 'logit':
            span = max(self.y_max - self.y_min, self.eps)
            scaled = (torch.clamp(y, min=self.y_min, max=self.y_max) - self.y_min) / span
            scaled = torch.clamp(scaled, min=self.eps, max=1.0 - self.eps)
            w = scaled * (1.0 - scaled) * span
        else:
            w = torch.ones_like(y)
        return w / torch.clamp(w.mean(), min=self.eps)

def _mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))

def apply_regression_calibration(val_pred, test_pred, val_target, method='none', center=None, clamp_min=None, clamp_max=None):
    val_pred = np.asarray(val_pred, dtype=np.float64)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    val_target = np.asarray(val_target, dtype=np.float64)
    skip_clamp = False
    if method == 'none':
        (out_val, out_test) = (val_pred, test_pred)
    elif method == 'linear':
        if val_pred.size < 2 or float(np.std(val_pred)) < 1e-12:
            (out_val, out_test) = (val_pred, test_pred)
        else:
            (a, b) = np.polyfit(val_pred, val_target, 1)
            out_val = a * val_pred + b
            out_test = a * test_pred + b
    elif method == 'median_blend':
        if center is None:
            center = float(np.median(val_target)) if val_target.size else 0.0
        best_alpha = 1.0
        best_mae = _mae(val_pred, val_target)
        for alpha in np.linspace(0.0, 1.0, 21):
            cand = alpha * val_pred + (1.0 - alpha) * center
            mae = _mae(cand, val_target)
            if mae <= best_mae:
                best_mae = mae
                best_alpha = float(alpha)
        out_val = best_alpha * val_pred + (1.0 - best_alpha) * center
        out_test = best_alpha * test_pred + (1.0 - best_alpha) * center
    elif method == 'mae':
        if val_pred.size < 2 or float(np.std(val_pred)) < 1e-12:
            (out_val, out_test) = (val_pred, test_pred)
        else:
            (best_a, best_b) = (1.0, 0.0)
            best_mae = _mae(val_pred, val_target)
            for a in np.linspace(0.5, 1.5, 201):
                b = float(np.median(val_target - a * val_pred))
                m = _mae(a * val_pred + b, val_target)
                if m < best_mae:
                    (best_mae, best_a, best_b) = (m, float(a), b)
            out_val = best_a * val_pred + best_b
            out_test = best_a * test_pred + best_b
        skip_clamp = True
    elif method == 'shrink':
        if center is None:
            center = float(np.median(val_target)) if val_target.size else 0.0
        if val_pred.size < 2 or float(np.std(val_pred)) < 1e-12:
            (out_val, out_test) = (val_pred, test_pred)
        else:
            best_s = 1.0
            best_mae = _mae(val_pred, val_target)
            for s in np.arange(0.6, 1.61, 0.02):
                m = _mae(center + float(s) * (val_pred - center), val_target)
                if m < best_mae:
                    (best_mae, best_s) = (m, float(s))
            out_val = center + best_s * (val_pred - center)
            out_test = center + best_s * (test_pred - center)
        skip_clamp = True
    elif method == 'logit_temp':
        if val_pred.size < 2:
            (out_val, out_test) = (val_pred, test_pred)
        else:

            def _apply(p, T):
                q = np.clip(p, 1e-06, 1.0 - 1e-06)
                z = np.log(q / (1.0 - q)) * T
                return 1.0 / (1.0 + np.exp(-z))
            (best_T, best_mae) = (1.0, _mae(val_pred, val_target))
            for T in np.arange(0.05, 2.01, 0.05):
                m = _mae(_apply(val_pred, float(T)), val_target)
                if m < best_mae:
                    (best_mae, best_T) = (m, float(T))
            out_val = _apply(val_pred, best_T)
            out_test = _apply(test_pred, best_T)
    elif method == 'power':
        if val_pred.size < 2 or float(np.max(val_pred)) <= 0:
            (out_val, out_test) = (val_pred, test_pred)
        else:
            vp = np.maximum(np.asarray(val_pred, dtype=np.float64), 0.0)
            tp = np.maximum(np.asarray(test_pred, dtype=np.float64), 0.0)
            (best_a, best_b) = (1.0, 1.0)
            best_mae = _mae(val_pred, val_target)
            for b in np.arange(0.5, 2.01, 0.02):
                base = vp ** float(b)
                for a in np.arange(0.05, 3.01, 0.05):
                    m = _mae(float(a) * base, val_target)
                    if m < best_mae:
                        (best_mae, best_a, best_b) = (m, float(a), float(b))
            out_val = best_a * vp ** best_b
            out_test = best_a * tp ** best_b
        skip_clamp = True
    else:
        raise ValueError(f'Unsupported regression calibration: {method}')
    if not skip_clamp and clamp_min is not None and (clamp_max is not None):
        out_val = np.clip(out_val, clamp_min, clamp_max)
        out_test = np.clip(out_test, clamp_min, clamp_max)
    return (out_val, out_test)
