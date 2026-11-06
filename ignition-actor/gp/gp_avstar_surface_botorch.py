#!/usr/bin/env python3
"""GP advisor for the critical-amplitude surface A*_v(R, gamma).

Reads the coordinator's threshold estimates, fits (R, gamma) -> log(A*_v), and writes
predictions, diagnostics, and new sample locations. R is the stimulus extent: the disk
radius in the radial problem and the segment length in the wave-segment problem.

Each input uses linear or log coordinates, chosen by analytic leave-one-out predictive log
probability once enough thresholds exist. Diagnostics use analytic leave-one-out residuals
with the fitted hyperparameters held fixed, split into interior and boundary points.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path


def first_nonempty(raw: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = raw.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes", "y"}


def require_botorch():
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    return torch, SingleTaskGP, Normalize, Standardize, ExactMarginalLogLikelihood, fit_gpytorch_mll


def target_to_model_space(value: float, transform: str) -> float:
    if transform != "log":
        raise ValueError(f"unknown target transform: {transform}")
    if value <= 0.0:
        raise ValueError("log target transform requires positive A*_v")
    return math.log(value)


def noise_to_model_space(value: float, half_width: float, transform: str, floor: float) -> float:
    """Observation variance in log space from a bracket half-width."""
    if transform != "log":
        raise ValueError(f"unknown target transform: {transform}")
    sigma = max(half_width / max(value, 1.0e-300), floor)
    return sigma * sigma


def model_to_physical(mean, variance, transform: str):
    """Posterior median, 95% interval, and standard deviation of a log-space posterior."""
    if transform != "log":
        raise ValueError(f"unknown target transform: {transform}")
    std = variance.clamp_min(0.0).sqrt()
    return mean.exp(), (mean - 1.96 * std).exp(), (mean + 1.96 * std).exp(), std


# An axis swept over more than this factor is treated as multiplicative.
AUTO_LOG_SPAN_RATIO = 4.0

# Fitted lengthscales are clamped before they scale the duplicate guard: one
# longer than the domain means the axis is nearly redundant but must not grow
# the guard without bound, and a collapsed one must not shrink it to nothing.
LENGTHSCALE_CLAMP = (0.05, 1.0)

# Fewer interior points than this and the interior leave-one-out figure is not
# reported, so an accuracy stop cannot fire on a handful of residuals.
MIN_INTERIOR_LOO_POINTS = 4

# Below this many thresholds there is not enough evidence to choose the input
# coordinates from the data, and the span heuristic is used instead.
MIN_TRANSFORM_SELECTION_POINTS = 8

# A learned input warp costs two parameters per axis; below this many
# thresholds fitting them is noise rather than signal.
MIN_WARP_SELECTION_POINTS = 16


@dataclass(frozen=True)
class SurfaceRow:
    sample: int | None
    R: float
    gamma: float
    Av_star_mid: float
    Av_star_lo: float
    Av_star_hi: float
    half_width: float
    source: str


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def within_closed_interval(value: float, lower: float, upper: float) -> bool:
    """Closed-bound check robust to one-ulp text/linspace roundoff."""
    tolerance = 32.0 * math.ulp(max(1.0, abs(lower), abs(upper)))
    return lower - tolerance <= value <= upper + tolerance


def read_surface_rows(path: Path) -> list[SurfaceRow]:
    """Read valid thresholds from a server or advisor CSV."""
    rows: list[SurfaceRow] = []
    with path.expanduser().open(newline="") as stream:
        for raw in csv.DictReader(stream):
            status = str(raw.get("status", "")).strip().lower()
            if status in {
                "failed",
                "cancelled",
                "upper_bound_not_igniting",
                "discarded_upper_bound_not_igniting",
                "no_threshold",
            }:
                continue
            converged = raw.get("converged")
            if converged is not None and str(converged).strip() and not parse_bool(converged):
                continue

            R_text = first_nonempty(raw, "R", "L")
            gamma_text = first_nonempty(raw, "gamma")
            mid_text = first_nonempty(
                raw,
                "Av_star_mid",
                "critical_Av",
                "A_v_star",
                "Avstar",
                "Av_star",
                "Astar_v",
            )
            if R_text is None or gamma_text is None or mid_text is None:
                continue

            R = float(R_text)
            gamma = float(gamma_text)
            mid = float(mid_text)
            lo = float(first_nonempty(raw, "Av_star_lo", "Av_lo") or mid)
            hi = float(first_nonempty(raw, "Av_star_hi", "Av_hi") or mid)
            bracket_width = first_nonempty(raw, "bracket_width")
            half_width_text = first_nonempty(raw, "half_width")
            if half_width_text is not None:
                half_width = abs(float(half_width_text))
            elif bracket_width is not None:
                half_width = 0.5 * abs(float(bracket_width))
            else:
                half_width = 0.5 * abs(hi - lo)
            if not all(math.isfinite(v) for v in (R, gamma, mid, lo, hi, half_width)):
                continue
            if gamma <= 0.0 or mid <= 0.0:
                continue

            sample_text = first_nonempty(raw, "sample", "sample_id") or ""
            sample = int(sample_text) if str(sample_text).strip() else None
            rows.append(
                SurfaceRow(
                    sample=sample,
                    R=R,
                    gamma=gamma,
                    Av_star_mid=mid,
                    Av_star_lo=min(lo, hi),
                    Av_star_hi=max(lo, hi),
                    half_width=half_width,
                    source=first_nonempty(raw, "source", "run_dir") or str(path),
                )
            )
    rows.sort(key=lambda row: (row.gamma, row.R))
    return rows


def read_locations(path: Path | None) -> list[tuple[float, float]]:
    """Read attempted or pending (R, gamma) locations."""
    if path is None:
        return []
    locations: list[tuple[float, float]] = []
    with path.expanduser().open(newline="") as stream:
        for raw in csv.DictReader(stream):
            R_text = first_nonempty(raw, "R", "L")
            gamma_text = first_nonempty(raw, "gamma")
            if R_text is None or gamma_text is None:
                continue
            try:
                location = (float(R_text), float(gamma_text))
            except ValueError:
                continue
            if math.isfinite(location[0]) and _finite_positive(location[1]):
                locations.append(location)
    return locations


def write_surface_rows(rows: list[SurfaceRow], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_id",
                "R",
                "gamma",
                "Av_star_mid",
                "Av_star_lo",
                "Av_star_hi",
                "half_width",
                "source",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    "" if row.sample is None else row.sample,
                    row.R,
                    row.gamma,
                    row.Av_star_mid,
                    row.Av_star_lo,
                    row.Av_star_hi,
                    row.half_width,
                    row.source,
                ]
            )


def axis_transform_choice(lower: float, upper: float, requested: str) -> str:
    """Fallback coordinate for one axis when there is too little data to choose.

    Treats an axis swept over more than ``AUTO_LOG_SPAN_RATIO`` as
    multiplicative.  This is only a cold-start guess; once
    ``MIN_TRANSFORM_SELECTION_POINTS`` thresholds exist,
    :func:`select_input_space` picks the coordinates from the data instead.
    """
    if requested == "linear":
        return "linear"
    if requested == "log":
        if not (lower > 0.0 and upper > 0.0):
            raise SystemExit("a log input transform requires positive bounds")
        return "log"
    if lower > 0.0 and upper > 0.0 and upper / lower >= AUTO_LOG_SPAN_RATIO:
        return "log"
    return "linear"


@dataclass(frozen=True)
class InputSpace:
    """The coordinates the GP, the designs and every distance all live in."""

    transforms: tuple[str, str]
    physical_lower: tuple[float, float]
    physical_upper: tuple[float, float]

    @staticmethod
    def build(
        r_min: float,
        r_max: float,
        gamma_min: float,
        gamma_max: float,
        requested_R: str,
        requested_gamma: str,
    ) -> "InputSpace":
        return InputSpace(
            (
                axis_transform_choice(r_min, r_max, requested_R),
                axis_transform_choice(gamma_min, gamma_max, requested_gamma),
            ),
            (r_min, gamma_min),
            (r_max, gamma_max),
        )

    def forward(self, physical):
        """Physical (R, gamma) -> model space."""
        torch = __import__("torch")
        return torch.stack(
            [
                physical[..., axis].log()
                if self.transforms[axis] == "log"
                else physical[..., axis]
                for axis in (0, 1)
            ],
            dim=-1,
        )

    def inverse(self, model):
        """Model space -> physical (R, gamma)."""
        torch = __import__("torch")
        return torch.stack(
            [
                model[..., axis].exp()
                if self.transforms[axis] == "log"
                else model[..., axis]
                for axis in (0, 1)
            ],
            dim=-1,
        )

    def bounds(self):
        torch = __import__("torch")
        edges = [
            [
                math.log(value) if transform == "log" else value
                for value, transform in zip(corner, self.transforms)
            ]
            for corner in (self.physical_lower, self.physical_upper)
        ]
        return torch.tensor(edges, dtype=torch.double)

    def describe(self) -> str:
        return f"R={self.transforms[0]}, gamma={self.transforms[1]}"


def normalize_locations(X, bounds):
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def minimum_normalized_distance(X, references, bounds, scale=None):
    """Minimum distance from each X to references, in the model unit square.

    ``scale`` optionally divides each axis by its fitted lengthscale, so the
    distance measures redundancy the way the GP sees it rather than in raw
    unit-square geometry.
    """
    torch = __import__("torch")
    if references is None or references.numel() == 0:
        return torch.full(
            (X.shape[0],), math.inf, dtype=X.dtype, device=X.device
        )
    X_unit = normalize_locations(X, bounds)
    reference_unit = normalize_locations(references, bounds)
    if scale is not None:
        X_unit = X_unit / scale
        reference_unit = reference_unit / scale
    return torch.cdist(X_unit, reference_unit).amin(dim=1)


def fitted_lengthscales(model):
    """Per-axis ARD lengthscales in the GP's own normalized input space."""
    torch = __import__("torch")
    kernel = getattr(model.covar_module, "base_kernel", model.covar_module)
    lengthscale = getattr(kernel, "lengthscale", None)
    if lengthscale is None:
        return None
    values = lengthscale.detach().reshape(-1).to(dtype=torch.double)
    if (
        values.numel() != 2
        or not bool(torch.isfinite(values).all())
        or float(values.min()) <= 0.0
    ):
        return None
    return values.clamp(*LENGTHSCALE_CLAMP)


@dataclass(frozen=True)
class DuplicateGuard:
    """How close a suggestion may sit to an attempted or pending location."""

    bounds: object
    scale: object
    radius: float
    floor: float

    def keep_mask(self, candidates, references):
        mask = minimum_normalized_distance(candidates, references, self.bounds) >= self.floor
        if self.radius <= 0.0:
            return mask
        scaled = (
            minimum_normalized_distance(candidates, references, self.bounds, self.scale)
            >= self.radius
        )
        combined = mask & scaled
        # The floor is the hard guarantee; the adaptive radius is a preference
        # and must never empty the pool.
        return combined if bool(combined.any()) else mask

    def filter(self, candidates, references):
        return candidates[self.keep_mask(candidates, references)]

    def describe(self) -> str:
        if self.scale is None:
            return f"unit-square radius={self.radius:g}, floor={self.floor:g}"
        scale = [round(float(v), 4) for v in self.scale]
        return (
            f"radius={self.radius:g} lengthscales (scale={scale}), "
            f"floor={self.floor:g}"
        )


def fit_surface_gp(
    rows: list[SurfaceRow],
    space: InputSpace,
    target_transform: str,
    noise_floor: float,
    warp: bool = False,
):
    """Fit (R, gamma) -> transformed A*_v.

    ``warp`` inserts a learned Kumaraswamy input warp after normalization.  A
    stationary kernel has one lengthscale per axis, which cannot hold both the
    flat large-R plateau and the steep rise as R approaches the radius below
    which nothing ignites; the warp lets the model stretch the input axis
    where the response actually changes.  It costs two parameters per axis, so
    it is worth having only once enough thresholds exist to fit them -- which
    is why the caller selects it by evidence rather than assuming it.
    """
    (
        torch,
        SingleTaskGP,
        Normalize,
        Standardize,
        ExactMarginalLogLikelihood,
        fit_gpytorch_mll,
    ) = require_botorch()

    physical_X = torch.tensor(
        [[row.R, row.gamma] for row in rows], dtype=torch.double
    )
    train_X = space.forward(physical_X)
    train_Y = torch.tensor(
        [
            [target_to_model_space(row.Av_star_mid, target_transform)]
            for row in rows
        ],
        dtype=torch.double,
    )
    train_Yvar = torch.tensor(
        [
            [
                noise_to_model_space(
                    row.Av_star_mid,
                    row.half_width,
                    target_transform,
                    noise_floor,
                )
            ]
            for row in rows
        ],
        dtype=torch.double,
    )
    if warp:
        from botorch.models.transforms.input import ChainedInputTransform, Warp

        input_transform = ChainedInputTransform(
            normalize=Normalize(d=2, bounds=space.bounds()),
            warp=Warp(indices=[0, 1]),
        )
    else:
        input_transform = Normalize(d=2, bounds=space.bounds())
    model = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return torch, model


def loo_log_probability(model) -> float:
    """Mean leave-one-out predictive log-probability of the training targets.

    Rasmussen & Williams eq. 5.10-5.12, with the fitted hyperparameters held
    fixed.  The targets are identical across input-coordinate choices, so this
    is directly comparable between them, and unlike the marginal likelihood it
    scores the model's *predictions* -- including their calibration -- rather
    than its fit.
    """
    torch = __import__("torch")
    X_internal = model.train_inputs[0]
    y = model.train_targets.reshape(-1)
    with torch.no_grad():
        mean = model.mean_module(X_internal).reshape(-1)
        covariance = model.covar_module(X_internal).to_dense()
        covariance = covariance + torch.diag(model.likelihood.noise.reshape(-1))
        chol = torch.linalg.cholesky(covariance)
        alpha = torch.cholesky_solve((y - mean).unsqueeze(-1), chol).squeeze(-1)
        precision = torch.diagonal(torch.cholesky_inverse(chol)).clamp_min(1.0e-300)
        log_probability = (
            0.5 * precision.log()
            - 0.5 * alpha.square() / precision
            - 0.5 * math.log(2.0 * math.pi)
        )
    value = float(log_probability.mean())
    return value if math.isfinite(value) else -math.inf


def next_input_model_selection_n(
    n_training: int,
    r_min: float,
    r_max: float,
    gamma_min: float,
    gamma_max: float,
    requested_R: str,
    requested_gamma: str,
    warp_mode: str = "auto",
) -> int:
    """Next sample count that expands the requested structural model options.

    Zero means that no sample-count eligibility transition remains.  An auto
    axis needs a refresh at the coordinate-selection threshold only when both
    linear and logarithmic coordinates are admissible.  Auto warping has its
    own later threshold.  These are the same constants used by selection,
    rather than a second set of thresholds maintained by the coordinator.

    This describes the *requested* model family, not the winning model.  The
    coordinator therefore retains this value from a model-selection request;
    an intervening fit of explicitly cached choices must not replace it.
    """
    upcoming = []
    if n_training < MIN_TRANSFORM_SELECTION_POINTS and any(
        requested == "auto" and lower > 0.0 and upper > 0.0
        for lower, upper, requested in (
            (r_min, r_max, requested_R),
            (gamma_min, gamma_max, requested_gamma),
        )
    ):
        upcoming.append(MIN_TRANSFORM_SELECTION_POINTS)
    if warp_mode == "auto" and n_training < MIN_WARP_SELECTION_POINTS:
        upcoming.append(MIN_WARP_SELECTION_POINTS)
    return min(upcoming, default=0)


def select_input_space(
    rows: list[SurfaceRow],
    r_min: float,
    r_max: float,
    gamma_min: float,
    gamma_max: float,
    requested_R: str,
    requested_gamma: str,
    target_transform: str,
    noise_floor: float,
    warp_mode: str = "auto",
):
    """Fit every allowed coordinate combination and keep the best-scoring one.

    Which coordinate an axis wants is a property of the response, not of the
    bounds: on the radial surface log(R) is decisively right while gamma is
    better left linear, and no rule reading only the swept ranges can tell
    those two axes apart.  So an axis left on ``auto`` is decided by leave-one-out
    predictive log-probability, re-decided every round as evidence accumulates.
    """

    def options(lower: float, upper: float, requested: str) -> list[str]:
        if requested != "auto":
            return [axis_transform_choice(lower, upper, requested)]
        if lower > 0.0 and upper > 0.0:
            return ["linear", "log"]
        return ["linear"]

    if warp_mode == "on":
        warps = [True]
    elif warp_mode == "off":
        warps = [False]
    else:
        # The warp adds two parameters per axis.  Offer it only once there are
        # enough thresholds that fitting them is not just noise.
        warps = [False, True] if len(rows) >= MIN_WARP_SELECTION_POINTS else [False]

    # The asynchronous coordinator feeds the previously selected structural
    # model back as explicit choices between periodic re-selection rounds.  In
    # that case there is nothing to compare: fitting an unwarped baseline and
    # then the requested warped model would double the cost of every async
    # update and could even ignore an explicit ``--input-warp on`` request when
    # the baseline happened to score slightly higher.
    if (
        requested_R != "auto"
        and requested_gamma != "auto"
        and warp_mode in {"on", "off"}
    ):
        space = InputSpace.build(
            r_min, r_max, gamma_min, gamma_max, requested_R, requested_gamma
        )
        chosen_warp = warp_mode == "on"
        torch, model = fit_surface_gp(
            rows, space, target_transform, noise_floor, chosen_warp
        )
        return torch, model, space, {}, "requested", chosen_warp

    if len(rows) < MIN_TRANSFORM_SELECTION_POINTS:
        space = InputSpace.build(
            r_min, r_max, gamma_min, gamma_max, requested_R, requested_gamma
        )
        torch, model = fit_surface_gp(
            rows, space, target_transform, noise_floor, warps[0]
        )
        return torch, model, space, {}, "span heuristic (too few thresholds)", warps[0]

    # Coordinates first, then the warp: a warp can absorb a bad coordinate choice in-sample,
    # and that absorption does not generalize.
    scores: dict[tuple[str, str, bool], float] = {}
    coordinate_best = None
    for transform_R in options(r_min, r_max, requested_R):
        for transform_gamma in options(gamma_min, gamma_max, requested_gamma):
            candidate = InputSpace(
                (transform_R, transform_gamma),
                (r_min, gamma_min),
                (r_max, gamma_max),
            )
            try:
                torch, model = fit_surface_gp(
                    rows, candidate, target_transform, noise_floor, False
                )
                score = loo_log_probability(model)
            except Exception:
                continue
            scores[(transform_R, transform_gamma, False)] = score
            if coordinate_best is None or score > coordinate_best[0]:
                coordinate_best = (score, torch, model, candidate)
    if coordinate_best is None:
        raise SystemExit("could not fit the surface GP in any input coordinates")

    best = (*coordinate_best, False)
    if True in warps:
        space_fixed = coordinate_best[3]
        try:
            torch, model = fit_surface_gp(
                rows, space_fixed, target_transform, noise_floor, True
            )
            score = loo_log_probability(model)
            scores[(*space_fixed.transforms, True)] = score
            if score > coordinate_best[0]:
                best = (score, torch, model, space_fixed, True)
        except Exception:
            pass
    _, torch, model, space, chosen_warp = best
    reason = (
        "leave-one-out log-probability"
        if len(scores) > 1
        else "requested"
    )
    return torch, model, space, scores, reason, chosen_warp


def posterior_at(model, X, target_transform: str) -> dict:
    torch = __import__("torch")
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1).detach()
        variance = posterior.variance.squeeze(-1).detach().clamp_min(0.0)
    physical_mean, physical_lower, physical_upper, model_std = model_to_physical(
        mean, variance, target_transform
    )
    return {
        "X": X.detach(),
        "model_mean": mean,
        "model_variance": variance,
        "model_std": model_std,
        "physical_mean": physical_mean,
        "physical_lower": physical_lower,
        "physical_upper": physical_upper,
    }


def sobol_locations(
    torch,
    count: int,
    bounds,
    seed: int,
    *,
    include_boundary_anchors: bool = False,
):
    """Scrambled Sobol points, uniform in the model space ``bounds`` describe."""
    count = max(1, count)
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    unit = engine.draw(count).to(dtype=torch.double)
    X = bounds[0] + unit * (bounds[1] - bounds[0])
    if include_boundary_anchors:
        anchors_unit = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.double,
        )
        anchors = bounds[0] + anchors_unit * (bounds[1] - bounds[0])
        X = torch.cat([anchors, X], dim=0)
    return X


def structured_prediction_grid(torch, bounds, r_count: int, gamma_count: int):
    radii = torch.linspace(
        float(bounds[0, 0]), float(bounds[1, 0]), max(2, r_count), dtype=torch.double
    )
    gammas = torch.linspace(
        float(bounds[0, 1]),
        float(bounds[1, 1]),
        max(2, gamma_count),
        dtype=torch.double,
    )
    return torch.cartesian_prod(radii, gammas)


def _relative_loo_summary(relative, rows, indices, prefix: str) -> dict:
    """RMS/max relative leave-one-out error over a subset of the design."""
    torch = __import__("torch")
    if not len(indices):
        return {
            f"{prefix}_n": 0,
            f"{prefix}_rms_relative_error": math.inf,
            f"{prefix}_max_relative_error": math.inf,
            f"{prefix}_worst_R": 0.0,
            f"{prefix}_worst_gamma": math.nan,
        }
    subset = relative[torch.tensor(indices, dtype=torch.long)]
    worst = indices[int(subset.argmax())]
    return {
        f"{prefix}_n": len(indices),
        f"{prefix}_rms_relative_error": float(subset.square().mean().sqrt()),
        f"{prefix}_max_relative_error": float(subset.max()),
        f"{prefix}_worst_R": rows[worst].R,
        f"{prefix}_worst_gamma": rows[worst].gamma,
    }


def analytic_gp_loo(
    model,
    rows: list[SurfaceRow],
    space: InputSpace,
    bounds,
    target_transform: str,
    interior_margin: float,
) -> dict:
    """GP leave-one-out residuals with fitted hyperparameters held fixed.

    For a Gaussian process with observation covariance K and
    alpha=K^{-1}(y-m), the conditional mean excluding observation i is
    y_i-alpha_i/(K^{-1})_ii.  Unlike the old neighbor interpolation metric,
    this conditions on every remaining point in the two-dimensional design.

    Points within ``interior_margin`` of the model-space boundary are reported
    separately: dropping a corner leaves nothing to interpolate from, so its
    residual measures the design, not the model, and would keep an accuracy
    stop from ever firing.
    """
    torch = __import__("torch")
    # In eval mode train_inputs already holds the input-transformed design;
    # transforming it again would normalize the normalized coordinates.
    X_internal = model.train_inputs[0]
    y = model.train_targets.reshape(-1)
    with torch.no_grad():
        mean = model.mean_module(X_internal).reshape(-1)
        covariance = model.covar_module(X_internal).to_dense()
        noise = model.likelihood.noise.reshape(-1)
        covariance = covariance + torch.diag(noise)
        chol = torch.linalg.cholesky(covariance)
        centered = y - mean
        alpha = torch.cholesky_solve(centered.unsqueeze(-1), chol).squeeze(-1)
        inverse = torch.cholesky_inverse(chol)
        inverse_diagonal = torch.diagonal(inverse).clamp_min(1.0e-300)
        loo_internal = y - alpha / inverse_diagonal

        transform = getattr(model, "outcome_transform", None)
        if transform is not None and hasattr(transform, "means"):
            loo_model = (
                loo_internal * transform.stdvs.reshape(-1)[0]
                + transform.means.reshape(-1)[0]
            )
        else:
            loo_model = loo_internal
        loo_physical = loo_model.exp()
    actual = torch.tensor(
        [row.Av_star_mid for row in rows], dtype=torch.double
    )
    relative = (loo_physical - actual).abs() / actual.abs().clamp_min(1.0e-300)

    physical = torch.tensor(
        [[row.R, row.gamma] for row in rows], dtype=torch.double
    )
    unit = normalize_locations(space.forward(physical), bounds)
    edge_distance = torch.minimum(unit, 1.0 - unit).amin(dim=1)
    interior = [
        index
        for index in range(len(rows))
        if float(edge_distance[index]) > interior_margin
    ]
    boundary = [index for index in range(len(rows)) if index not in set(interior)]

    metrics = _relative_loo_summary(
        relative, rows, list(range(len(rows))), "shape_loo"
    )
    if len(interior) >= MIN_INTERIOR_LOO_POINTS:
        metrics.update(_relative_loo_summary(relative, rows, interior, "interior_loo"))
    else:
        metrics.update(_relative_loo_summary(relative, rows, [], "interior_loo"))
    metrics.update(_relative_loo_summary(relative, rows, boundary, "boundary_loo"))
    metrics["interior_loo_margin"] = interior_margin
    return metrics


def boundary_support(bounds, training_model, tolerance: float) -> dict[str, bool]:
    """Whether each model-space edge carries a sample within ``tolerance``."""
    torch = __import__("torch")
    if training_model is None or training_model.numel() == 0:
        unit = torch.empty((0, 2), dtype=torch.double)
    else:
        unit = normalize_locations(training_model, bounds)
    supported = lambda column, target: bool(  # noqa: E731
        unit.numel() and bool(((unit[:, column] - target).abs() <= tolerance).any())
    )
    return {
        "lower_boundary_supported": supported(0, 0.0),
        "upper_boundary_supported": supported(0, 1.0),
        "gamma_lower_boundary_supported": supported(1, 0.0),
        "gamma_upper_boundary_supported": supported(1, 1.0),
    }


def surface_metrics(
    model,
    rows: list[SurfaceRow],
    space: InputSpace,
    bounds,
    target_transform: str,
    integration_count: int,
    seed: int,
    boundary_tolerance: float,
    interior_margin: float,
    lengthscales,
    training_locations,
    selection_reason: str,
    chosen_warp: bool,
) -> dict[str, float | int | bool | str]:
    torch = __import__("torch")
    X = sobol_locations(torch, integration_count, bounds, seed + 101)
    result = posterior_at(model, X, target_transform)
    physical_half_width = 0.5 * (
        result["physical_upper"] - result["physical_lower"]
    ).clamp_min(0.0)
    relative_half_width = physical_half_width / result[
        "physical_mean"
    ].abs().clamp_min(1.0e-300)
    metrics: dict[str, float | int | bool | str] = {
        "R_min": float(space.physical_lower[0]),
        "R_max": float(space.physical_upper[0]),
        "gamma_min": float(space.physical_lower[1]),
        "gamma_max": float(space.physical_upper[1]),
        "n_grid": int(X.shape[0]),
        "n_training": len(rows),
        "target_transform": target_transform,
        "input_transform_R": space.transforms[0],
        "input_transform_gamma": space.transforms[1],
        "input_transform_selected_by": selection_reason,
        "input_warp": "on" if chosen_warp else "off",
        "loo_log_probability": loo_log_probability(model),
        "lengthscale_R": math.nan if lengthscales is None else float(lengthscales[0]),
        "lengthscale_gamma": math.nan
        if lengthscales is None
        else float(lengthscales[1]),
        "max_model_std": float(result["model_std"].max()),
        "rms_model_std": float(result["model_variance"].mean().sqrt()),
        "mean_model_variance": float(result["model_variance"].mean()),
        "integrated_model_variance": float(result["model_variance"].mean()),
        "max_physical_half_width95": float(physical_half_width.max()),
        "mean_physical_half_width95": float(physical_half_width.mean()),
        "max_relative_half_width95": float(relative_half_width.max()),
        "rms_relative_half_width95": float(relative_half_width.square().mean().sqrt()),
        "mean_relative_half_width95": float(relative_half_width.mean()),
    }
    metrics.update(
        analytic_gp_loo(
            model, rows, space, bounds, target_transform, interior_margin
        )
    )
    # Only converged thresholds may certify stopping support. Attempted and
    # pending locations remain valid acquisition exclusions, but an unfinished
    # or failed edge point is not evidence for the fitted surface.
    metrics.update(
        boundary_support(bounds, training_locations, boundary_tolerance)
    )
    return metrics


def _variation_weighted_integration_points(
    model,
    torch,
    bounds,
    count: int,
    variation_weight: float,
    seed: int,
):
    """Sobol integration points mixed with points weighted by |grad mean|."""
    count = max(2, count)
    weight = min(1.0, max(0.0, variation_weight))
    variation_count = round(weight * count)
    uniform_count = count - variation_count
    parts = []
    if uniform_count:
        parts.append(sobol_locations(torch, uniform_count, bounds, seed + 211))
    if variation_count:
        pool_count = max(4 * variation_count, 256)
        unit = torch.quasirandom.SobolEngine(
            2, scramble=True, seed=seed + 307
        ).draw(pool_count).to(dtype=torch.double)
        unit.requires_grad_(True)
        X = bounds[0] + unit * (bounds[1] - bounds[0])
        mean = model.posterior(X).mean.reshape(-1)
        gradient = torch.autograd.grad(mean.sum(), unit)[0]
        scores = gradient.square().sum(dim=1).sqrt().detach()
        if not bool(torch.isfinite(scores).all()) or float(scores.sum()) <= 1.0e-14:
            parts.append(sobol_locations(torch, variation_count, bounds, seed + 401))
        else:
            probabilities = scores.clamp_min(1.0e-15)
            state = torch.random.get_rng_state()
            try:
                torch.manual_seed(seed + 409)
                chosen = torch.multinomial(
                    probabilities,
                    min(variation_count, pool_count),
                    replacement=False,
                )
            finally:
                torch.random.set_rng_state(state)
            parts.append(X.detach()[chosen])
    return torch.cat(parts, dim=0)


def posterior_variance_given(model, mc_points, extra_X=None):
    """Posterior variance at ``mc_points`` after conditioning on extra input
    locations, in the model's standardized output space.

    Conditioning needs only the locations: a GP's posterior variance is
    independent of the observed values, so work that is merely dispatched --
    or merely contemplated -- already counts.
    """
    torch = __import__("torch")
    with torch.no_grad():
        X = model.train_inputs[0]
        noise = model.likelihood.noise.reshape(-1)
        if extra_X is not None and extra_X.numel():
            extra = model.transform_inputs(extra_X.reshape(-1, X.shape[-1]))
            X = torch.cat([X, extra], dim=0)
            noise = torch.cat([noise, noise.mean().expand(extra.shape[0])])
        target = model.transform_inputs(mc_points)
        K = model.covar_module(X).to_dense() + torch.diag(noise)
        cross = model.covar_module(X, target).to_dense()
        prior = model.covar_module(target, target).to_dense().diagonal()
        solved = torch.cholesky_solve(cross, torch.linalg.cholesky(K))
        return (prior - (cross * solved).sum(dim=0)).clamp_min(0.0)


def predicted_relative_half_width(model, grid_X, extra_X, target_transform):
    """Max and RMS relative 95% half-width the surrogate would have after
    conditioning on ``extra_X``.

    Used to answer "will the accuracy gate already be satisfied once the work
    now being contemplated reports?"  If so, anything beyond that is dispatched
    only to be cancelled.
    """
    torch = __import__("torch")
    with torch.no_grad():
        mean = model.posterior(grid_X).mean.squeeze(-1).detach()
    variance = posterior_variance_given(model, grid_X, extra_X)
    outcome = getattr(model, "outcome_transform", None)
    if outcome is not None and hasattr(outcome, "stdvs"):
        variance = variance * outcome.stdvs.reshape(-1)[0] ** 2
    physical_mean, lower, upper, _ = model_to_physical(
        mean, variance, target_transform
    )
    half = 0.5 * (upper - lower).clamp_min(0.0)
    relative = half / physical_mean.abs().clamp_min(1.0e-300)
    return float(relative.max()), float(relative.square().mean().sqrt())


def integrated_posterior_variance(model, mc_points, extra_X=None) -> float:
    """Mean posterior variance over ``mc_points``, optionally conditioning on
    additional input locations.

    A GP's posterior variance depends only on where you observe, not on what
    you observe, so a location that is merely in flight can be conditioned on
    before its value is known.  That is what makes the marginal value of the
    j-th point in a batch computable before any of them have run.
    """
    return float(posterior_variance_given(model, mc_points, extra_X).mean())


def candidate_locations(torch, count: int, bounds, seed: int):
    return sobol_locations(
        torch, count, bounds, seed + 503, include_boundary_anchors=True
    )


def suggestion_record(
    model,
    X,
    space: InputSpace,
    target_transform: str,
    acquisition_value: float,
    method: str,
    pending_count: int,
    marginal_gain: float = float("nan"),
    marginal_gain_ratio: float = float("nan"),
) -> dict[str, float | str | int | bool]:
    result = posterior_at(model, X.reshape(1, 2), target_transform)
    physical = space.inverse(X.reshape(1, 2)).reshape(-1)
    mean = float(result["physical_mean"][0])
    lower = float(result["physical_lower"][0])
    upper = float(result["physical_upper"][0])
    return {
        "R": float(physical[0]),
        "gamma": float(physical[1]),
        "posterior_std": float(result["model_std"][0]),
        "relative_half_width95": 0.5 * max(0.0, upper - lower) / max(abs(mean), 1e-300),
        "pred_mean": mean,
        "pred_lower95": lower,
        "pred_upper95": upper,
        "acquisition_value": acquisition_value,
        "marginal_gain": marginal_gain,
        "marginal_gain_ratio": marginal_gain_ratio,
        "requested_method": method,
        "selection_method": method,
        "fallback_used": False,
        "pending_count": pending_count,
    }


def _analytic_nipv_values(
    model,
    candidates,
    integration,
    X_pending=None,
    acquisition_chunk_size: int = 128,
):
    """Evaluate q=1 NIPV by exact Gaussian conditioning, without fantasy models.

    Pending and previously selected locations are conditioned on together with
    the training data.  Their observation noise, and each proposed location's
    noise, is the original training-noise mean, matching BoTorch's fixed-noise
    ``fantasize`` convention.  Scores are in the unstandardized target units,
    not physical-amplitude units when the target is logarithmic.

    A single Cholesky factor and integration cross-solve are shared across all
    candidate chunks.  No candidate-by-candidate or integration-by-integration
    covariance matrix is materialized.  Unsupported models and unstable solves
    raise errors so the caller can visibly fall back to the reference backend.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from gpytorch.settings import min_variance

    if not isinstance(model, SingleTaskGP) or model.num_outputs != 1:
        raise RuntimeError("analytic NIPV requires a single-output SingleTaskGP")
    if not isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        raise RuntimeError("analytic NIPV requires fixed Gaussian observation noise")
    if getattr(model.likelihood, "second_noise_covar", None) is not None:
        raise RuntimeError("analytic NIPV does not support additional learned noise")
    outcome = getattr(model, "outcome_transform", None)
    if outcome is not None and type(outcome) is not Standardize:
        raise RuntimeError("analytic NIPV supports only Standardize outcome transforms")
    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None:
        for transform in input_transform.modules():
            if getattr(transform, "is_one_to_many", False):
                raise RuntimeError("analytic NIPV does not support one-to-many input transforms")
            if (getattr(transform, "transform_on_eval", False)
                    and not getattr(transform, "transform_on_fantasize", True)):
                raise RuntimeError("analytic NIPV requires the same input transforms for fantasies")

    with torch.no_grad():
        model.eval()
        training = model.train_inputs[0]
        noise = model.likelihood.noise
        if training.ndim != 2 or noise.ndim != 1:
            raise RuntimeError("analytic NIPV does not support batched GP models")
        if (candidates.ndim != 2 or integration.ndim != 2
                or integration.shape[0] == 0 or noise.numel() != training.shape[0]):
            raise RuntimeError("invalid analytic NIPV input dimensions")
        if not bool(torch.isfinite(noise).all()) or not bool((noise > 0).all()):
            raise RuntimeError("analytic NIPV requires finite positive training noise")
        if candidates.shape[0] == 0:
            return candidates.new_empty((0,))
        fantasy_noise = noise.mean()
        scale_squared = (outcome.stdvs.reshape(-1)[0].square()
                         if outcome is not None else training.new_tensor(1.0))
        if not bool(torch.isfinite(scale_squared)) or not bool(scale_squared > 0):
            raise RuntimeError("invalid analytic NIPV outcome scale")

        # eval-mode training inputs already contain Normalize/Warp transforms;
        # apply those transforms exactly once to new locations only.
        conditioning = training
        conditioning_noise = noise
        if X_pending is not None and X_pending.numel():
            if X_pending.ndim != 2:
                raise RuntimeError("analytic NIPV pending locations must be a matrix")
            transformed_pending = model.transform_inputs(X_pending)
            conditioning = torch.cat((training, transformed_pending), dim=0)
            conditioning_noise = torch.cat((
                noise, fantasy_noise.expand(transformed_pending.shape[0])))
        target = model.transform_inputs(integration)
        K = model.covar_module(conditioning).to_dense()
        K = K + torch.diag(conditioning_noise)
        if not bool(torch.isfinite(K).all()):
            raise RuntimeError("nonfinite analytic NIPV conditioning covariance")
        # Do not silently add a different jitter/noise model.  The caller uses
        # BoTorch's established numerical safeguards if this factorization fails.
        factor = torch.linalg.cholesky(K)
        cross_target = model.covar_module(conditioning, target).to_dense()
        whitened_target = torch.linalg.solve_triangular(factor, cross_target, upper=False)
        prior_target = model.covar_module(target, diag=True)
        variance_target = prior_target - whitened_target.square().sum(dim=0)
        tolerance = (100.0 * torch.finfo(training.dtype).eps
                     * prior_target.abs().max().clamp_min(1.0))
        if (not bool(torch.isfinite(variance_target).all())
                or bool((variance_target < -tolerance).any())):
            raise RuntimeError("unstable analytic NIPV integration variance")

        values = []
        chunk_size = max(1, acquisition_chunk_size)
        for start in range(0, candidates.shape[0], chunk_size):
            chunk = model.transform_inputs(candidates[start:start + chunk_size])
            cross_candidate = model.covar_module(conditioning, chunk).to_dense()
            whitened_candidate = torch.linalg.solve_triangular(
                factor, cross_candidate, upper=False)
            prior_candidate = model.covar_module(chunk, diag=True)
            variance_candidate = prior_candidate - whitened_candidate.square().sum(dim=0)
            denominator = variance_candidate + fantasy_noise
            if (not bool(torch.isfinite(denominator).all())
                    or not bool((denominator > 0).all())
                    or bool((variance_candidate < -tolerance).any())):
                raise RuntimeError("unstable analytic NIPV candidate variance")
            cross = (model.covar_module(target, chunk).to_dense()
                     - whitened_target.transpose(-1, -2) @ whitened_candidate)
            conditional_variance = variance_target.unsqueeze(-1) - cross.square() / denominator
            if (not bool(torch.isfinite(conditional_variance).all())
                    or bool((conditional_variance < -tolerance).any())):
                raise RuntimeError("unstable analytic NIPV conditioned variance")
            # GPyTorch applies this floor after Standardize is undone; flooring
            # before output scaling would change low-variance acquisition scores.
            unstandardized = conditional_variance * scale_squared
            if not bool(torch.isfinite(unstandardized).all()):
                raise RuntimeError("nonfinite analytic NIPV output-scaled variance")
            values.append(-unstandardized.clamp_min(
                min_variance.value(unstandardized.dtype)).mean(dim=0))
        return torch.cat(values)


def suggest_greedy_nipv(
    model,
    candidates,
    attempted,
    pending,
    guard: DuplicateGuard,
    space: InputSpace,
    bounds,
    batch_size: int,
    integration_count: int,
    acquisition_chunk_size: int,
    variation_weight: float,
    target_transform: str,
    seed: int,
    method: str,
    value_target: float = 0.0,
    stop_max_relative: float = 0.0,
    stop_rms_relative: float = 0.0,
    metric_points=None,
    nipv_backend: str = "botorch",
) -> list[dict]:
    from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

    torch = __import__("torch")
    if nipv_backend not in ("botorch", "analytic"):
        raise ValueError(f"unknown NIPV backend: {nipv_backend}")
    available = guard.filter(candidates, attempted)
    integration = _variation_weighted_integration_points(
        model,
        torch,
        bounds,
        integration_count,
        variation_weight if method == "variation-nipv" else 0.0,
        seed,
    )
    selected_X = []
    suggestions = []
    # Integrated variance before anything new is committed, conditioned on
    # whatever is already in flight.  Each pick maximizes the next reduction
    # for the current fitted model.  Integrated variance reduction is not
    # generally submodular, so we measure every gain instead of assuming a
    # monotonically decreasing sequence or bounding an uncomputed tail.
    conditioned_now = pending if (pending is not None and pending.numel()) else None
    previous_variance = integrated_posterior_variance(model, integration, conditioned_now)
    gains: list[float] = []
    for _ in range(batch_size):
        if available.numel() == 0:
            break
        conditioned = []
        if pending is not None and pending.numel():
            conditioned.append(pending)
        if selected_X:
            conditioned.append(torch.stack(selected_X))
        X_pending = torch.cat(conditioned, dim=0) if conditioned else None
        values = None
        if nipv_backend == "analytic":
            try:
                values = _analytic_nipv_values(
                    model, available, integration, X_pending, acquisition_chunk_size)
            except (RuntimeError, ValueError, NotImplementedError) as error:
                warnings.warn(
                    f"analytic NIPV fallback to BoTorch for this decision: {error}",
                    RuntimeWarning,
                )
                nipv_backend = "botorch"
        if values is None:
            acquisition = qNegIntegratedPosteriorVariance(
                model=model,
                mc_points=integration,
                X_pending=X_pending,
            )
            values = []
            for start in range(0, available.shape[0], max(1, acquisition_chunk_size)):
                stop = min(start + max(1, acquisition_chunk_size), available.shape[0])
                with torch.no_grad():
                    values.append(acquisition(available[start:stop].unsqueeze(-2)).reshape(-1))
            values = torch.cat(values)
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            break
        values[~finite] = -torch.inf
        index = int(values.argmax())
        X = available[index]

        committed = [X.reshape(1, 2)] if not selected_X else [
            torch.stack(selected_X), X.reshape(1, 2)
        ]
        if conditioned_now is not None:
            committed.insert(0, conditioned_now)
        variance_after = integrated_posterior_variance(
            model, integration, torch.cat(committed, dim=0)
        )
        gain = max(0.0, previous_variance - variance_after)
        gains.append(gain)
        previous_variance = variance_after

        selected_X.append(X)
        suggestions.append(
            suggestion_record(
                model,
                X,
                space,
                target_transform,
                float(values[index]),
                method,
                0 if pending is None else int(pending.shape[0]),
                gain,
                float("nan"),
            )
        )
        available = available[guard.keep_mask(available, X.reshape(1, 2))]


    # Commit the prefix capturing ``value_target`` of this fully evaluated
    # batch's variance reduction.  Remaining locations may be reconsidered
    # after pending results arrive.  The existing policy retains at least one
    # suggestion whenever an admissible proposal exists.
    total = sum(gains)
    keep = len(suggestions)
    if value_target > 0.0 and total > 0.0 and len(suggestions) > 1:
        running = 0.0
        for index, gain in enumerate(gains):
            running += gain
            if running / total >= value_target:
                keep = index + 1
                break

    # Second question, independent of the first: will the accuracy gate
    # already be met once this work reports?  With the fitted hyperparameters
    # held fixed, posterior variance depends only on locations.  This is a
    # look-ahead heuristic, not a guarantee: refitting, changed mean predictions
    # and the independent certification conditions can change the later gate.
    if (
        (stop_max_relative > 0.0 or stop_rms_relative > 0.0)
        and metric_points is not None
        and len(suggestions) > 1
    ):
        base = [conditioned_now] if conditioned_now is not None else []
        for index in range(len(suggestions)):
            prefix = torch.cat(base + [torch.stack(selected_X[: index + 1])], dim=0)
            max_rel, rms_rel = predicted_relative_half_width(
                model, metric_points, prefix, target_transform
            )
            satisfied = (
                stop_max_relative <= 0.0 or max_rel <= stop_max_relative
            ) and (stop_rms_relative <= 0.0 or rms_rel <= stop_rms_relative)
            if satisfied:
                keep = min(keep, index + 1)
                break
    if keep < len(suggestions):
        suggestions = suggestions[:keep]
    for index, record in enumerate(suggestions):
        record["marginal_gain_ratio"] = (
            sum(gains[: index + 1]) / total if total > 0.0 else 1.0
        )
    return suggestions


def write_predictions(
    result: dict, space: InputSpace, path: Path, target_transform: str
) -> None:
    half_width = 0.5 * (
        result["physical_upper"] - result["physical_lower"]
    ).clamp_min(0.0)
    relative = half_width / result["physical_mean"].abs().clamp_min(1.0e-300)
    physical_X = space.inverse(result["X"])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "R",
                "gamma",
                "Av_star_mean",
                "Av_star_lower95",
                "Av_star_upper95",
                "Av_star_physical_half_width95",
                "Av_star_relative_half_width95",
                f"model_mean_{target_transform}",
                f"model_std_{target_transform}",
            ]
        )
        for values in zip(
            physical_X[:, 0].tolist(),
            physical_X[:, 1].tolist(),
            result["physical_mean"].tolist(),
            result["physical_lower"].tolist(),
            result["physical_upper"].tolist(),
            half_width.tolist(),
            relative.tolist(),
            result["model_mean"].tolist(),
            result["model_std"].tolist(),
        ):
            writer.writerow(values)


def write_dict_row(values: dict, path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)


def _radical_inverse(index: int, base: int) -> float:
    """One term of a Halton sequence."""
    out, f = 0.0, 1.0 / base
    while index > 0:
        out += f * (index % base)
        index //= base
        f /= base
    return out


def probe_locations(
    torch,
    count: int,
    start_index: int,
    space,
    attempted: list[tuple[float, float]],
    min_distance_normalized: float,
) -> list[tuple[float, float]]:
    """Space-filling probes with physical-coordinate exclusion references.

    ``attempted`` contains physical (R, gamma) locations, including pending
    searches and acquisitions selected in the current proposal. Transform
    those references exactly once into the normalized model coordinates.

    Halton locations do not maximize the acquisition. Only the hard duplicate
    floor is applied, not the model's lengthscale-based spacing preference.
    The deterministic sequence, selected coordinates and duplicate exclusion
    do not provide a theorem of unbiased error estimation.
    """
    out: list[tuple[float, float]] = []
    index = max(1, start_index)
    lo, hi = space.bounds()
    guard_sq = min_distance_normalized * min_distance_normalized
    while len(out) < count and index < start_index + 4096:
        u = torch.tensor(
            [[_radical_inverse(index, 2), _radical_inverse(index, 3)]],
            dtype=torch.double,
        )
        index += 1
        physical = space.inverse(lo + (hi - lo) * u)
        R = float(physical[0, 0])
        gamma = float(physical[0, 1])
        unit = ((space.forward(physical) - lo) / (hi - lo))[0]
        too_close = False
        for aR, agamma in list(attempted) + out:
            a_unit = (
                (space.forward(torch.tensor([[aR, agamma]], dtype=torch.double)) - lo)
                / (hi - lo)
            )[0]
            d = float(((unit - a_unit) ** 2).sum())
            if d < guard_sq:
                too_close = True
                break
        if not too_close:
            out.append((R, gamma))
    return out, index


def write_suggestions(suggestions: list[dict], path: Path) -> None:
    fields = [
        "R",
        "gamma",
        "posterior_std",
        "relative_half_width95",
        "pred_mean",
        "pred_lower95",
        "pred_upper95",
        "acquisition_value",
        "marginal_gain",
        "marginal_gain_ratio",
        "requested_method",
        "selection_method",
        "fallback_used",
        "pending_count",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(suggestions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="gp_surface")
    parser.add_argument("--r-min", type=float, required=True)
    parser.add_argument("--r-max", type=float, required=True)
    parser.add_argument("--gamma-min", type=float, required=True)
    parser.add_argument("--gamma-max", type=float, required=True)
    parser.add_argument("--target-transform", choices=["log"], default="log")
    parser.add_argument(
        "--input-transform-r",
        choices=["auto", "log", "linear"],
        default="auto",
        help="Coordinate the R axis is modelled, sampled and measured in.",
    )
    parser.add_argument(
        "--input-transform-gamma",
        choices=["auto", "log", "linear"],
        default="auto",
        help="Coordinate the gamma axis is modelled, sampled and measured in.",
    )
    parser.add_argument(
        "--input-warp",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "learned Kumaraswamy warp on the GP inputs, letting a stationary "
            "kernel cope with a response far steeper in one part of the "
            "domain than another; auto selects it by leave-one-out "
            "log-probability alongside the coordinates"
        ),
    )
    parser.add_argument("--noise-floor", type=float, default=1.0e-3)
    parser.add_argument("--prediction-r-count", type=int, default=160)
    parser.add_argument("--prediction-gamma-count", type=int, default=100)
    parser.add_argument("--integration-count", type=int, default=1024)
    parser.add_argument("--candidate-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--acquisition",
        choices=["greedy-nipv", "variation-nipv"],
        default="greedy-nipv",
    )
    parser.add_argument(
        "--nipv-backend",
        choices=["botorch", "analytic"],
        default="botorch",
        help="NIPV score backend; analytic uses exact fixed-noise Gaussian conditioning",
    )
    parser.add_argument(
        "--stop-max-relative-half-width",
        type=float,
        default=0.0,
        help=(
            "the run's max relative 95%% half-width stopping tolerance; the "
            "advisor uses it to avoid dispatching work that will be cancelled "
            "because the gate is already met. 0 disables"
        ),
    )
    parser.add_argument(
        "--stop-rms-relative-half-width",
        type=float,
        default=0.0,
        help="as --stop-max-relative-half-width, for the RMS tolerance",
    )
    parser.add_argument(
        "--acquisition-value-target",
        type=float,
        default=0.0,
        help=(
            "commit only the prefix of the batch that captures this fraction "
            "of the variance reduction the full batch would achieve, letting "
            "the model size its own batch; 0 disables, returning all of it"
        ),
    )
    parser.add_argument("--variation-weight", type=float, default=0.5)
    parser.add_argument("--acquisition-chunk-size", type=int, default=128)
    parser.add_argument(
        "--probe-count",
        type=int,
        default=0,
        help="space-filling probe locations to append to the batch; their "
             "prediction error is an unbiased estimate of generalization error",
    )
    parser.add_argument(
        "--probe-index",
        type=int,
        default=1,
        help="Halton index to start the probe sequence from",
    )
    parser.add_argument(
        "--suggest-min-distance-normalized",
        type=float,
        default=0.01,
        help=(
            "Hard floor on the distance to an attempted or pending location, "
            "in the model unit square."
        ),
    )
    parser.add_argument(
        "--suggest-min-distance-lengthscales",
        type=float,
        default=0.15,
        help=(
            "Preferred separation in fitted-lengthscale units; adapts to the "
            "anisotropy the data has revealed. 0 disables it."
        ),
    )
    parser.add_argument(
        "--interior-margin-normalized",
        type=float,
        default=0.05,
        help="Model unit-square margin separating interior from boundary LOO.",
    )
    parser.add_argument("--attempted-locations-csv", type=Path, default=None)
    parser.add_argument("--pending-locations-csv", type=Path, default=None)
    parser.add_argument("--boundary-tolerance-normalized", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true", help="accepted from the coordinator; no plots are written")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--defer-acquisition", action="store_true",
                        help="Write diagnostics and a private checkpoint; select no locations")
    stages.add_argument("--resume-acquisition", action="store_true",
                        help="Select from this run's trusted checkpoint without refitting")
    parser.add_argument("--resume-probe-only-count", type=int, default=None,
                        help="Resume only: replace acquisition by this many probes, "
                             "bounded by the original batch size; never refit")
    return parser


def decision_signature(args):
    signature = vars(args).copy()
    for key in ("defer_acquisition", "resume_acquisition", "resume_probe_only_count"):
        signature.pop(key)
    return signature


def main(argv=None, session=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.r_min < args.r_max:
        parser.error("--r-min must be smaller than --r-max")
    if not 0.0 < args.gamma_min < args.gamma_max:
        parser.error("gamma bounds must satisfy 0 < gamma-min < gamma-max")
    if args.batch_size < 0:
        parser.error("--batch-size must be non-negative")
    if args.resume_probe_only_count is not None:
        if not args.resume_acquisition:
            parser.error("--resume-probe-only-count requires --resume-acquisition")
        if not 0 < args.resume_probe_only_count <= args.batch_size:
            parser.error("resume probe count must be positive and within original batch size")
    if args.integration_count < 2 or args.candidate_count < 1:
        parser.error("integration-count must be >=2 and candidate-count >=1")
    if not 0.0 <= args.variation_weight <= 1.0:
        parser.error("--variation-weight must be in [0,1]")
    if args.suggest_min_distance_normalized < 0.0:
        parser.error("--suggest-min-distance-normalized must be non-negative")
    if args.suggest_min_distance_lengthscales < 0.0:
        parser.error("--suggest-min-distance-lengthscales must be non-negative")
    if not 0.0 <= args.interior_margin_normalized < 0.5:
        parser.error("--interior-margin-normalized must be in [0,0.5)")

    rows = read_surface_rows(args.input)
    rows = [
        row
        for row in rows
        if within_closed_interval(row.R, args.r_min, args.r_max)
        and within_closed_interval(row.gamma, args.gamma_min, args.gamma_max)
    ]
    if len(rows) < 3:
        raise SystemExit("Need at least three valid surface thresholds")

    checkpoint_path = args.out_dir / f"{args.prefix}_acquisition.pt"
    if args.resume_acquisition:
        import torch
        # Only load the private checkpoint created by this advisor invocation.
        # Full model serialization preserves the fitted warp and posterior;
        # resuming must never fit a second, potentially different GP.
        if session is None:
            state = torch.load(checkpoint_path, weights_only=False)
        else:
            state = session.pop("decision", None)
            if state is None:
                raise ValueError("No retained GP decision to resume")
        signature = decision_signature(args)
        if state["arguments"] != signature or state["rows"] != rows:
            raise ValueError("Acquisition checkpoint does not match this decision")
        model, space = state["model"], state["space"]
        transform_scores = state["transform_scores"]
        selection_reason, chosen_warp = state["selection_reason"], state["chosen_warp"]
    else:
        if session is not None and session:
            raise ValueError("Previous GP decision must be resumed or discarded")
        # A long-lived interpreter must not inherit fitting randomness from a
        # previous decision. Use the same explicit request seed in BOTH the
        # one-shot and persistent prototype paths (including optimizer retries).
        import torch
        import numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        (
            torch, model, space, transform_scores, selection_reason, chosen_warp,
        ) = select_input_space(
            rows, args.r_min, args.r_max, args.gamma_min, args.gamma_max,
            args.input_transform_r, args.input_transform_gamma,
            args.target_transform, args.noise_floor, args.input_warp,
        )
    bounds = space.bounds()
    lengthscales = fitted_lengthscales(model)

    attempted_locations = read_locations(args.attempted_locations_csv)
    if not attempted_locations:
        attempted_locations = [(row.R, row.gamma) for row in rows]
    pending_locations = read_locations(args.pending_locations_csv)
    attempted_locations = list(dict.fromkeys(attempted_locations + pending_locations))
    attempted = space.forward(
        torch.tensor(attempted_locations, dtype=torch.double).reshape(-1, 2)
    )
    training_locations = space.forward(
        torch.tensor(
            [[row.R, row.gamma] for row in rows], dtype=torch.double
        ).reshape(-1, 2)
    )
    pending = (
        space.forward(torch.tensor(pending_locations, dtype=torch.double).reshape(-1, 2))
        if pending_locations
        else torch.empty((0, 2), dtype=torch.double)
    )
    use_lengthscales = (
        args.suggest_min_distance_lengthscales > 0.0 and lengthscales is not None
    )
    guard = DuplicateGuard(
        bounds=bounds,
        scale=lengthscales if use_lengthscales else None,
        radius=args.suggest_min_distance_lengthscales
        if use_lengthscales
        else args.suggest_min_distance_normalized,
        floor=args.suggest_min_distance_normalized,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / f"{args.prefix}_training.csv"
    predictions_path = args.out_dir / f"{args.prefix}_predictions.csv"
    metrics_path = args.out_dir / f"{args.prefix}_metrics.csv"
    suggestions_path = args.out_dir / f"{args.prefix}_suggestions.csv"

    if args.resume_acquisition:
        prediction, metrics = state["prediction"], state["metrics"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        import numpy as np
        np.random.set_state(state["numpy_rng"])
    else:
        write_surface_rows(rows, rows_path)
        prediction_X = structured_prediction_grid(
            torch, bounds, args.prediction_r_count, args.prediction_gamma_count
        )
        prediction = posterior_at(model, prediction_X, args.target_transform)
        write_predictions(prediction, space, predictions_path, args.target_transform)
        metrics = surface_metrics(
            model, rows, space, bounds, args.target_transform,
            args.integration_count, args.seed, args.boundary_tolerance_normalized,
            args.interior_margin_normalized, lengthscales, training_locations,
            selection_reason, chosen_warp,
        )
        metrics["input_model_next_selection_n"] = next_input_model_selection_n(
            len(rows), args.r_min, args.r_max, args.gamma_min, args.gamma_max,
            args.input_transform_r, args.input_transform_gamma, args.input_warp,
        )
        write_dict_row(metrics, metrics_path)
    if args.defer_acquisition:
        import numpy as np
        signature = decision_signature(args)
        retained = dict(
            arguments=signature, rows=rows, model=model, space=space,
            transform_scores=transform_scores, selection_reason=selection_reason,
            chosen_warp=chosen_warp, prediction=prediction, metrics=metrics,
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
        )
        if session is None:
            torch.save(retained, checkpoint_path)
        else:
            session["decision"] = retained
        write_suggestions([], suggestions_path)
        print(f"surface GP diagnostics ready: n={len(rows)}; acquisition deferred")
        return

    # Only the explicit resume control may replace acquisition after diagnostics.
    # Signature/row checks and posterior restoration have already succeeded.
    # The original batch/probe arguments remain part of the checkpoint identity.
    if args.resume_probe_only_count is not None:
        args.batch_size = args.resume_probe_only_count
        args.probe_count = args.resume_probe_only_count
    candidates = candidate_locations(torch, args.candidate_count, bounds, args.seed)
    ordinary_count = max(0, args.batch_size - args.probe_count)
    if ordinary_count == 0 or args.resume_probe_only_count is not None:
        suggestions = []
    else:
        suggestions = suggest_greedy_nipv(
            model,
            candidates,
            attempted,
            pending,
            guard,
            space,
            bounds,
            ordinary_count,
            args.integration_count,
            args.acquisition_chunk_size,
            args.variation_weight,
            args.target_transform,
            args.seed,
            args.acquisition,
            args.acquisition_value_target,
            args.stop_max_relative_half_width,
            args.stop_rms_relative_half_width,
            sobol_locations(torch, args.integration_count, bounds, args.seed + 101),
            nipv_backend=args.nipv_backend,
        )
    if args.probe_count > 0:
        probes, _next_index = probe_locations(
            torch,
            args.probe_count,
            args.probe_index,
            space,
            attempted_locations + [
                (float(s["R"]), float(s["gamma"])) for s in suggestions
            ],
            args.suggest_min_distance_normalized,
        )
        for R_probe, gamma_probe in probes:
            X_probe = space.forward(
                torch.tensor([[R_probe, gamma_probe]], dtype=torch.double)
            )
            post = posterior_at(model, X_probe, args.target_transform)
            mean = float(post["physical_mean"][0])
            lower = float(post["physical_lower"][0])
            upper = float(post["physical_upper"][0])
            suggestions.append(
                {
                    "R": R_probe,
                    "gamma": gamma_probe,
                    "posterior_std": float(post["model_std"][0]),
                    "relative_half_width95": 0.5 * (upper - lower) / max(mean, 1e-300),
                    "pred_mean": mean,
                    "pred_lower95": lower,
                    "pred_upper95": upper,
                    "acquisition_value": "",
                    "marginal_gain": "",
                    "marginal_gain_ratio": "",
                    "requested_method": "probe",
                    "selection_method": "probe",
                    "fallback_used": "false",
                    "pending_count": len(pending),
                }
            )
        print(f"appended {len(probes)} space-filling probe location(s)")
    write_suggestions(suggestions, suggestions_path)

    print(f"read surface thresholds: {len(rows)}")
    print(
        f"surface domain: R=[{args.r_min:g}, {args.r_max:g}], "
        f"gamma=[{args.gamma_min:g}, {args.gamma_max:g}]"
    )
    print(
        f"surface model space: {space.describe()}, "
        f"warp={'on' if chosen_warp else 'off'} (by {selection_reason})"
    )
    if len(transform_scores) > 1:
        ranked = sorted(transform_scores.items(), key=lambda item: -item[1])
        print(
            "surface coordinate scores (mean LOO log-probability): "
            + ", ".join(
                f"R={key[0]}/gamma={key[1]}/warp={'on' if key[2] else 'off'}:"
                f" {value:.3f}"
                for key, value in ranked
            )
        )
    print(f"surface duplicate guard: {guard.describe()}")
    print(
        "surface GP uncertainty: "
        f"max/RMS relative 95% half-width="
        f"{100.0 * float(metrics['max_relative_half_width95']):.3f}%/"
        f"{100.0 * float(metrics['rms_relative_half_width95']):.3f}%"
    )
    print(
        "GP leave-one-out (interior): "
        f"n={int(metrics['interior_loo_n'])}, "
        f"RMS={100.0 * float(metrics['interior_loo_rms_relative_error']):.3f}%, "
        f"max={100.0 * float(metrics['interior_loo_max_relative_error']):.3f}% "
        f"at (R,gamma)=({float(metrics['interior_loo_worst_R']):.6g}, "
        f"{float(metrics['interior_loo_worst_gamma']):.6g})"
    )
    print(
        "GP leave-one-out (boundary): "
        f"n={int(metrics['boundary_loo_n'])}, "
        f"RMS={100.0 * float(metrics['boundary_loo_rms_relative_error']):.3f}%, "
        f"max={100.0 * float(metrics['boundary_loo_max_relative_error']):.3f}%"
    )
    acquisition_picks = [
        s for s in suggestions if s.get("selection_method") != "probe"
    ]
    if args.acquisition_value_target > 0.0 and acquisition_picks:
        # Probes carry no marginal gain: they are not chosen to reduce variance,
        # so they are excluded from the value-sizing diagnostic.
        print(
            f"surface batch sized by value: {len(acquisition_picks)}/"
            f"{args.batch_size} points, cumulative variance capture "
            + ", ".join(f"{s['marginal_gain_ratio']:.2f}" for s in acquisition_picks)
            + f" (target {args.acquisition_value_target:g})"
        )
    print(f"suggested surface batch: {len(suggestions)}/{args.batch_size}")
    print(f"wrote {rows_path}")
    print(f"wrote {predictions_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {suggestions_path}")


if __name__ == "__main__":
    main()
