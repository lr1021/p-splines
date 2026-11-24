"""Helpers for specifying the fixed prior precision tau from the model keys file.

Convention: tau = 10 ** (-tau_exp), and run names contain f"tau{tau_exp}".
e.g. tau_exp = 1 -> tau = 0.1 -> 'CDtau1_...';  tau_exp = -2 -> tau = 100 -> 'Qtau-2_...'

The f"tau{tau_exp}" token in the run folder name is what the distortion report
scripts substitute (re.sub(r'tau[-+eE.\\d]+', ...)), so keep it in run_name.
"""
import re

import numpy as np
import xarray as xr


def fmt_num(v):
    """Compact, filename-safe number formatting: 1.0 -> '1', 0.1 -> '0.1', -0.5 -> '-0.5'."""
    return f"{float(v):g}"


def tau_from_exp(tau_exp):
    return 10.0 ** (-float(tau_exp))


def make_run_name(run_prefix, tau_exp, a, b, n_internal_knots):
    return f"{run_prefix}tau{fmt_num(tau_exp)}_a{fmt_num(a)}b{fmt_num(b)}k{int(n_internal_knots)}"


TAU_RETRY_REL_NUDGE = 1e-9


def nudge_tau(tau_val):
    """Fallback tau for fits that fail at the exact value.

    Some (data, tau) combinations crash in pytensor's numba code generation
    (seen for CD20 replications 35/48 at exactly tau = 0.1: 'Function type returns ... but resty=...').
    tau + tau*1e-9 gives exactly the 0.1000000001 used for those fits in the original runs.
    """
    return tau_val + tau_val * TAU_RETRY_REL_NUDGE


def builder_tau_kwargs(keys):
    """kwargs to pass to builder_dict[builder](...).

    Empty if the keys file does not define tau_val, so builders without a
    tau_val argument (and old keys files) behave exactly as before.
    """
    tau_val = getattr(keys, 'tau_val', None)
    return {} if tau_val is None else {'tau_val': tau_val}


_TAU_VAR = re.compile(r'^tau(_\d+)?$')  # 'tau' (ortho_diag) or 'tau_0', 'tau_1', ... (ortho_diag_mult); not tau_p


def read_idata_tau(idata_path):
    """Return {var_name: value} of the fixed tau value(s) stored in an idata's posterior."""
    with xr.open_dataset(idata_path, group='posterior') as post:
        return {k: float(np.asarray(post[k].values).flat[0])
                for k in post.data_vars if _TAU_VAR.match(k)}


def check_idata_tau(idata_path, tau_val, rtol=1e-6):
    """Warn (print) if an existing idata was sampled with a different tau than tau_val.

    rtol=1e-6 treats the old hard-coded 0.1000000001 as equal to 0.1.
    Returns True if consistent (or nothing to check), False on mismatch.
    """
    if tau_val is None:
        return True
    try:
        stored = read_idata_tau(idata_path)
    except Exception as e:
        print(f"WARNING: could not read tau from {idata_path}: {e}")
        return False
    bad = {k: v for k, v in stored.items() if not np.isclose(v, tau_val, rtol=rtol, atol=0.0)}
    if bad:
        print(f"WARNING: tau mismatch in existing idata {idata_path}: stored {bad}, keys tau_val={tau_val}. "
              f"Set replace_idatas=True to resample.")
        return False
    return True
