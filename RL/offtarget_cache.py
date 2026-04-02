"""
Cached off-target score computation for RL training.

Wrapper around CUDA/CPU off-target search and CRISPRspec surrogate.
Caches results per gRNA sequence to avoid recomputing during training.
Supports batch-style usage: compute for multiple sequences, cache each.
"""

import os
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _REPO_ROOT)

# Module-level cache: sequence (20-mer) -> (n_off_targets, crisprspec_score)
_OFFTARGET_CACHE: Dict[str, Tuple[int, float]] = {}
_REFERENCE_FASTA_PATH: Optional[str] = None
_REFERENCE_GENOME_SEQ: Optional[str] = None
_OFFTARGET_FN = None


def _get_offtarget_fn():
    """Lazy-init off-target search function (CUDA or CPU)."""
    global _OFFTARGET_FN
    if _OFFTARGET_FN is not None:
        return _OFFTARGET_FN
    try:
        from RL.grna_rl_adapters_cuda import find_off_targets_in_genome_cuda
        _OFFTARGET_FN = find_off_targets_in_genome_cuda
    except Exception:
        try:
            from RL.grna_rl_adapters import find_off_targets_in_genome
            _OFFTARGET_FN = find_off_targets_in_genome
        except Exception:
            _OFFTARGET_FN = None
    return _OFFTARGET_FN


def _force_cpu_offtarget_fn():
    """Force CPU off-target function (used when CUDA context is broken)."""
    global _OFFTARGET_FN
    try:
        from RL.grna_rl_adapters import find_off_targets_in_genome

        _OFFTARGET_FN = find_off_targets_in_genome
    except Exception:
        _OFFTARGET_FN = None
    return _OFFTARGET_FN


def load_reference_genome(fasta_path: str) -> Optional[str]:
    """Load reference genome from FASTA; cache and return sequence."""
    global _REFERENCE_FASTA_PATH, _REFERENCE_GENOME_SEQ
    path = os.path.abspath(fasta_path)
    if path == _REFERENCE_FASTA_PATH and _REFERENCE_GENOME_SEQ is not None:
        return _REFERENCE_GENOME_SEQ
    if not os.path.isfile(path):
        return None
    try:
        from RL.grna_rl_adapters import load_genome
        _REFERENCE_GENOME_SEQ = load_genome(path)
        _REFERENCE_FASTA_PATH = path
        return _REFERENCE_GENOME_SEQ
    except Exception:
        return None


def compute_offtarget_score_cuda(
    grna_20: str,
    reference_fasta_path: str,
    use_cache: bool = True,
    pam_3: str = "GGG",
) -> Tuple[int, float]:
    """
    Compute off-target count and CRISPRspec score for one gRNA.
    
    Parameters
    ----------
    grna_20 : str
        20-mer gRNA sequence
    reference_fasta_path : str
        Path to reference FASTA (e.g. genes_with_flankers.fna)
    use_cache : bool
        If True, return cached result when available
    pam_3 : str
        3-letter PAM for exclude (default GGG)
    
    Returns
    -------
    tuple
        (n_off_targets, crisprspec_score)
        crisprspec_score in ~[0, 10], higher = more specific.
        Returns (0, 10.0) on error or if disabled.
    """
    grna_20 = grna_20.upper()[:20]
    if len(grna_20) != 20:
        return (0, 10.0)
    
    if use_cache and grna_20 in _OFFTARGET_CACHE:
        return _OFFTARGET_CACHE[grna_20]
    
    genome_seq = load_reference_genome(reference_fasta_path)
    if not genome_seq:
        if not getattr(compute_offtarget_score_cuda, '_no_genome_warned', False):
            compute_offtarget_score_cuda._no_genome_warned = True
            import sys
            sys.stderr.write(
                f"[offtarget_cache] WARNING: Failed to load genome from {reference_fasta_path}. "
                f"Returning (0, 10.0) — CRISPRspec will be constant!\n"
            )
        return (0, 10.0)
    
    fn = _get_offtarget_fn()
    if fn is None:
        if not getattr(compute_offtarget_score_cuda, '_no_fn_warned', False):
            compute_offtarget_score_cuda._no_fn_warned = True
            import sys
            sys.stderr.write(
                "[offtarget_cache] WARNING: No off-target search function available. "
                "Returning (0, 10.0) — CRISPRspec will be constant!\n"
            )
        return (0, 10.0)
    
    try:
        guide_23_exclude = grna_20 + (pam_3 if len(pam_3) == 3 else "GGG")
        off_targets = fn(
            grna_20, genome_seq,
            max_mismatches=4,
            exclude_23=guide_23_exclude,
        )
        from RL.grna_rl_adapters import predict_crisprspec_surrogate
        cs = predict_crisprspec_surrogate(grna_20, off_targets, pam_3=pam_3)
        n_ot = len(off_targets)
        score = float(cs) if cs is not None else (10.0 if n_ot == 0 else max(0.0, 10.0 - __import__("math").log10(1 + n_ot)))
        result = (n_ot, score)
        if use_cache:
            _OFFTARGET_CACHE[grna_20] = result
        return result
    except Exception as e:
        # Common SLURM/driver issue with numba: non-primary CUDA context.
        # In that case, force CPU fallback and retry once to get a real score.
        msg = str(e)
        if not getattr(compute_offtarget_score_cuda, '_error_logged', False):
            compute_offtarget_score_cuda._error_logged = True
            import sys
            sys.stderr.write(f"[offtarget_cache] Error in off-target search: {e}\n")
            sys.stderr.flush()
        if "non-primary CUDA context" in msg or "non primary CUDA context" in msg:
            fn_cpu = _force_cpu_offtarget_fn()
            if fn_cpu is not None:
                try:
                    guide_23_exclude = grna_20 + (pam_3 if len(pam_3) == 3 else "GGG")
                    off_targets = fn_cpu(
                        grna_20, genome_seq,
                        max_mismatches=4,
                        exclude_23=guide_23_exclude,
                    )
                    from RL.grna_rl_adapters import predict_crisprspec_surrogate
                    cs = predict_crisprspec_surrogate(grna_20, off_targets, pam_3=pam_3)
                    n_ot = len(off_targets)
                    score = float(cs) if cs is not None else (10.0 if n_ot == 0 else max(0.0, 10.0 - __import__("math").log10(1 + n_ot)))
                    result = (n_ot, score)
                    if use_cache:
                        _OFFTARGET_CACHE[grna_20] = result
                    return result
                except Exception:
                    return (0, 10.0)
        return (0, 10.0)


def compute_offtarget_score_batch(
    grna_list: List[str],
    reference_fasta_path: str,
    use_cache: bool = True,
) -> List[Tuple[int, float]]:
    """
    Compute off-target scores for a list of gRNAs (one call per unique sequence, cache used).
    Minimizes GPU calls by processing in one go and caching each result.
    """
    genome_seq = load_reference_genome(reference_fasta_path)
    if not genome_seq:
        return [(0, 10.0) for _ in grna_list]
    
    fn = _get_offtarget_fn()
    if fn is None:
        return [(0, 10.0) for _ in grna_list]
    
    results = []
    for grna in grna_list:
        r = compute_offtarget_score_cuda(grna, reference_fasta_path, use_cache=use_cache)
        results.append(r)
    return results


def clear_cache():
    """Clear the off-target cache (e.g. when switching reference genome)."""
    global _OFFTARGET_CACHE
    _OFFTARGET_CACHE = {}
