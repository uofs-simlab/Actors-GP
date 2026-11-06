"""GP advisor of the final runs, the script the coordinator starts.

It runs the fit of gp_avstar_surface_botorch.py with these additions:

- Warped models get N(0, 1) priors on the log warp concentrations.
- Warped models try a cold start and a warm start from the previous fit of the same run, and keep the
  larger MAP objective. Each fit writes an audit, ``<prefix>_fit_policy.json``.
- Each deferred fit writes two forecasts from the fitted model, without changing it.
  ``<prefix>_pending_predictions.csv`` has the predictive mean and 95% interval at the locations still
  in flight, which the coordinator uses to promote an in-flight probe once a fit is ready.
  ``<prefix>_readiness_forecast.csv`` has the maximum and RMS relative 95% half-width, before and after
  conditioning on those locations, which the coordinator uses to anticipate readiness.

With ``--persistent-server`` the advisor stays running for the whole run and serves the coordinator's
requests one at a time (``serve`` below); without it, it handles one request from the command line.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import traceback
import warnings

import gp_avstar_surface_botorch as api

ORIGINAL_FIT = api.fit_surface_gp
ORIGINAL_SELECT = api.select_input_space
NOISE_FLOOR = .003
PRIOR_SIGMA = 1.


# ---- warped model

def _log_zero(module):
    return module.concentration0.log()


def _log_one(module):
    return module.concentration1.log()


def _set_log_zero(module, value):
    module._set_concentration(i=0, value=value.exp())


def _set_log_one(module, value):
    module._set_concentration(i=1, value=value.exp())


def warped_model(rows, space, sigma):
    """Warped GP of the log threshold with N(0, sigma) priors on the log warp concentrations, before fitting."""
    from botorch.models.transforms.input import ChainedInputTransform, Warp
    from gpytorch.priors import NormalPrior
    torch, SingleTaskGP, Normalize, Standardize, _, _ = api.require_botorch()
    X = space.forward(torch.tensor([[r.R, r.gamma] for r in rows], dtype=torch.double))
    Y = torch.tensor([[math.log(r.Av_star_mid)] for r in rows], dtype=torch.double)
    Yvar = torch.tensor([[api.noise_to_model_space(
        r.Av_star_mid, r.half_width, 'log', NOISE_FLOOR)] for r in rows], dtype=torch.double)
    transform = ChainedInputTransform(normalize=Normalize(d=2, bounds=space.bounds()),
                                      warp=Warp(indices=[0, 1]))
    model = SingleTaskGP(X, Y, train_Yvar=Yvar, input_transform=transform,
                         outcome_transform=Standardize(m=1))
    warp = model.input_transform.warp
    for index, closure, setter in ((0, _log_zero, _set_log_zero),
                                    (1, _log_one, _set_log_one)):
        reference = warp.concentration0
        prior = NormalPrior(reference.new_tensor(0.), reference.new_tensor(sigma))
        warp.register_prior(f'log_concentration{index}_prior', prior, closure, setter)
    return model


def training_objective(model):
    """Marginal likelihood plus log priors of a fitted model, per observation."""
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    with torch.no_grad():
        model.train()
        value = ExactMarginalLogLikelihood(model.likelihood, model)(
            model(model.train_inputs[0]), model.train_targets)
        objective = float(value)
        model.eval()
    return objective


# ---- cold and warm starts

def row_signature(rows):
    return sorted((asdict(row) for row in rows), key=lambda r: (r['R'], r['gamma']))


def context(rows, space, warp):
    return dict(rows=row_signature(rows), transforms=list(space.transforms),
                lower=list(space.physical_lower), upper=list(space.physical_upper), warp=bool(warp))


def eligible(previous, current):
    if previous is None:
        return False
    if any(previous[k] != current[k] for k in ('transforms', 'lower', 'upper', 'warp')):
        return False
    before = {(r['R'], r['gamma']): r for r in previous['rows']}
    after = {(r['R'], r['gamma']): r for r in current['rows']}
    return (len(before) == len(previous['rows']) and len(after) == len(current['rows'])
            and before.keys() <= after.keys() and all(after[k] == v for k, v in before.items()))


def parameter_state(model):
    return dict(parameters={k: v.detach().cpu().tolist() for k, v in model.named_parameters()},
                output_mean=float(model.outcome_transform.means.reshape(())),
                output_std=float(model.outcome_transform.stdvs.reshape(())))


def transfer(state, model):
    import torch
    params = dict(model.named_parameters())
    if params.keys() != state['parameters'].keys():
        raise ValueError('Warm parameter families differ')
    old_std = float(state['output_std'])
    if not math.isfinite(old_std) or old_std <= 0:
        raise ValueError('Invalid previous output scale')
    with torch.no_grad():
        for name, param in params.items():
            value = param.new_tensor(state['parameters'][name])
            if value.shape != param.shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid learned parameter: ' + name)
            param.copy_(value)
        new_std = model.outcome_transform.stdvs.reshape(())
        physical_mean = state['output_mean'] + old_std * model.mean_module.constant
        model.mean_module.constant = (physical_mean-model.outcome_transform.means.reshape(()))/new_std
        model.covar_module.outputscale = model.covar_module.outputscale*(old_std/new_std).square()


class FitPolicy:
    def __init__(self, previous=None):
        self.previous = previous
        self.selected = None
        self.attempts = []

    def fit(self, rows, space, target_transform, noise_floor, warp=False):
        import numpy as np
        import torch
        if target_transform != 'log' or noise_floor != NOISE_FLOOR:
            raise ValueError('The fitting policy requires a log target and noise floor .003')
        current = context(rows, space, warp)
        starts = [('cold', None)]
        if warp and eligible(self.previous, current):
            starts.append(('warm', self.previous['state']))
        before = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        after_cold = None
        candidates = []
        try:
            for start, old in starts:
                torch.set_rng_state(before[0]); np.random.set_state(before[1]); random.setstate(before[2])
                record = dict(n=len(rows), transforms=list(space.transforms), warp=bool(warp),
                              start=start, sigma=PRIOR_SIGMA if warp else None, success=False)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        if warp:
                            model = warped_model(rows, space, PRIOR_SIGMA)
                            if old is not None:
                                transfer(old, model)
                            _, _, _, _, MLL, fit = api.require_botorch()
                            fit(MLL(model.likelihood, model))
                            model.eval()
                        else:
                            _, model = ORIGINAL_FIT(rows, space, target_transform, noise_floor, False)
                        objective = training_objective(model)
                        if not math.isfinite(objective):
                            raise ArithmeticError('Nonfinite fit objective')
                        record.update(success=True, objective=objective, parameters=parameter_state(model))
                        candidates.append((objective, start == 'cold', model, record))
                    except Exception as error:
                        record.update(exception=type(error).__name__, message=str(error))
                record['warnings'] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
                self.attempts.append(record)
                if start == 'cold':
                    after_cold = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        finally:
            state = after_cold or before
            torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])
        if not candidates:
            raise RuntimeError('All declared starts failed; see fit audit')
        _, _, model, chosen = max(candidates, key=lambda item: item[:2])
        for _, _, _, record in candidates:
            record['selected_start'] = record is chosen
        return torch, model

    def select(self, rows, *args, **kwargs):
        result = ORIGINAL_SELECT(rows, *args, **kwargs)
        _, model, space, _, _, warp = result
        self.selected = dict(**context(rows, space, warp), state=parameter_state(model))
        return result


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def previous_state(directory, current_prefix):
    """Only completed earlier decision records in this run; never load pickle."""
    found = []
    current_round = int(current_prefix.rsplit('_', 1)[1]) if current_prefix.startswith('adaptive_surface_round_') else math.inf
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path in directory.glob('adaptive_surface_round_*_fit_policy.json'):
        if path.name == current_prefix + '_fit_policy.json':
            continue
        record = json.loads(path.read_text())
        number = int(record['prefix'].rsplit('_', 1)[1])
        if record['advisor_sha256'] != digest:
            raise RuntimeError('Warm state belongs to another advisor version')
        if number < current_round and record['status'] == 'completed' and record['selected'] is not None:
            found.append((number, record))
    return max(found, key=lambda x: x[0])[1]['selected'] if found else None


# ---- forecasts

def _retained_decision(args, session):
    if session is not None:
        return session.get('decision')
    import torch
    path = args.out_dir / f'{args.prefix}_acquisition.pt'
    return torch.load(path, weights_only=False) if path.exists() else None


def _write_new(path: Path, header, rows):
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([repr(v) if isinstance(v, float) else v for v in row])


def write_forecasts(args, session) -> None:
    import torch
    decision = _retained_decision(args, session)
    if decision is None:
        raise RuntimeError('No retained GP decision for forecast files')
    model, space, rows = decision['model'], decision['space'], decision['rows']
    bounds = space.bounds()
    pending = api.read_locations(args.pending_locations_csv)
    out = []
    if pending:
        X = space.forward(torch.tensor(pending, dtype=torch.double).reshape(-1, 2))
        post = api.posterior_at(model, X, args.target_transform)
        for i, (R, gamma) in enumerate(pending):
            out.append((float(R), float(gamma), float(post['physical_mean'][i]),
                        float(post['physical_lower'][i]), float(post['physical_upper'][i]),
                        float(post['model_std'][i]), len(rows)))
    _write_new(args.out_dir / f'{args.prefix}_pending_predictions.csv',
               ('R', 'gamma', 'pred_mean', 'pred_lower95', 'pred_upper95',
                'posterior_std', 'n_training'), out)

    grid = api.sobol_locations(torch, args.integration_count, bounds, args.seed + 101)
    now_max, now_rms = api.predicted_relative_half_width(model, grid, None, args.target_transform)
    if pending:
        X = space.forward(torch.tensor(pending, dtype=torch.double).reshape(-1, 2))
        after_max, after_rms = api.predicted_relative_half_width(
            model, grid, X, args.target_transform)
    else:
        after_max, after_rms = now_max, now_rms
    for value in (now_max, now_rms, after_max, after_rms):
        if not math.isfinite(value) or value < 0:
            raise ArithmeticError('Nonfinite readiness forecast')
    _write_new(args.out_dir / f'{args.prefix}_readiness_forecast.csv',
               ('n_training', 'n_pending', 'n_grid', 'max_relative_half_width95_now',
                'rms_relative_half_width95_now', 'max_relative_half_width95_after_pending',
                'rms_relative_half_width95_after_pending'),
               [(len(rows), len(pending), int(grid.shape[0]), now_max, now_rms,
                 after_max, after_rms)])


def main(argv=None, session=None):
    args = api.build_parser().parse_args(argv)
    if args.resume_acquisition:
        return api.main(argv, session=session)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = args.out_dir / (args.prefix + '_fit_policy.json')
    if audit.exists():
        raise FileExistsError('Refusing to overwrite fitting evidence: ' + str(audit))
    policy = FitPolicy(previous_state(args.out_dir, args.prefix))
    api.fit_surface_gp, api.select_input_space = policy.fit, policy.select
    status = 'failed'
    try:
        result = api.main(argv, session=session)
        status = 'completed'
    finally:
        api.fit_surface_gp, api.select_input_space = ORIGINAL_FIT, ORIGINAL_SELECT
        save_new(audit, dict(prefix=args.prefix, status=status, sigma=PRIOR_SIGMA,
                 selection='training MAP within family; existing LOO between representations',
                 selected=policy.selected, attempts=policy.attempts,
                 advisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if args.defer_acquisition:
        write_forecasts(args, session)
    return result


# ---- persistent process

# Private, sequential, framed IPC with the coordinator: no network listener, threads, shared memory,
# eval or shell execution. One fitted model may be kept between diagnostics and acquisition.
def serve(advisor):
    # Reserve the original stdout for protocol replies; even native library
    # writes to fd 1 now go to the inherited log, never into the IPC stream.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    session = {}
    last_id = 0
    print(f"persistent GP advisor started pid={os.getpid()}", flush=True)
    try:
        while True:
            header = sys.stdin.buffer.readline(129)
            if not header:
                return
            parts = header.split()
            if not header.endswith(b"\n") or len(parts) != 2:
                raise ValueError("Malformed advisor request header")
            request_id, size = map(int, parts)
            if request_id <= last_id or not 0 <= size <= 1024 * 1024:
                raise ValueError("Invalid advisor request ID or length")
            payload = sys.stdin.buffer.read(size)
            if len(payload) != size:
                raise EOFError("Truncated advisor request")
            last_id = request_id
            command = payload.decode("utf-8")
            try:
                if command == "DISCARD":
                    session.clear()
                else:
                    argv = shlex.split(command)
                    if len(argv) < 3:
                        raise ValueError("Advisor request needs executable, script and arguments")
                    # C++ supplies its ordinary shell-quoted command. Only the
                    # arguments are parsed; executable/script are never run.
                    advisor(argv[2:], session=session)
            except (Exception, SystemExit):
                session.clear()
                traceback.print_exc()
                protocol.write(f"{request_id} 1\n".encode())
                return  # fail closed; never consume stale output after failure
            sys.stdout.flush()
            sys.stderr.flush()
            protocol.write(f"{request_id} 0\n".encode())
    finally:
        session.clear()
        protocol.close()
        print(f"persistent GP advisor stopped pid={os.getpid()}", flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--persistent-server']:
        serve(main)
    else:
        main()
