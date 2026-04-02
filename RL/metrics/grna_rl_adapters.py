"""
Adapters for gRNA RL environment: genome loading, off-target search, CRISPRspec surrogate.
Uses energy/CRISPRspec_CRISPRoff_pipeline.py for CRISPRspec scoring.
"""

import os
import sys
from typing import List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reverse complement for DNA (PAM / off-target on both strands)
_REV_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def _rev_comp(seq: str) -> str:
    """Reverse complement of DNA sequence."""
    return seq.translate(_REV_MAP)[::-1]


def load_genome(path: str) -> str:
    """
    Load genome from FASTA file; return single concatenated sequence (uppercase).
    """
    from Bio import SeqIO
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for record in SeqIO.parse(f, "fasta"):
            seqs.append(str(record.seq).upper().replace("U", "T"))
    return "".join(seqs)


def find_off_targets_in_genome(
    guide_20: str,
    genome_seq: str = None,
    max_mismatches: int = 4,
    pam_suffixes: tuple = ("GG",),
    exclude_23: Optional[str] = None,
    gene_id: Optional[str] = None,
    fasta_path: Optional[str] = None,
    flank_size: int = 100_000_000,
) -> List[str]:
    """
    Find off-target sites (23-mer: 20 spacer + 3 PAM) in genome for a 20-mer guide.
    
    Searches both strands. Returns list of 23-mer strings (spacer + PAM).
    Considers only PAM GG (plus strand) / CC (minus strand). Includes 0–max_mismatches;
    if exclude_23 is given, that 23-mer (the on-target gRNA site) is not included.
    
    **NEW**: Windowed search mode
    If gene_id and fasta_path are provided, searches only within a flanked genomic window
    [gene_start - flank_size, gene_end + flank_size] instead of the full genome.
    This dramatically speeds up off-target search for large genomes.
    
    Parameters
    ----------
    guide_20 : str
        20-mer gRNA sequence
    genome_seq : str, optional
        Full genome sequence (required if gene_id/fasta_path not provided)
    max_mismatches : int
        Maximum Hamming distance (default 4)
    pam_suffixes : tuple
        PAM suffixes for plus strand (default ("GG",))
    exclude_23 : str, optional
        23-mer to exclude (on-target site)
    gene_id : str, optional
        Gene ID with coordinates (e.g. "NC_037347.1:c32043372-31869704")
        If provided with fasta_path, enables windowed search
    fasta_path : str, optional
        Path to genome FASTA file for windowed search
    flank_size : int
        Flank size in bp for windowed search (default 100 Mbp)
    
    Returns
    -------
    list of str
        23-mer off-target sequences
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return []
    
    # Windowed search mode: load only flanked region around gene
    if gene_id and fasta_path:
        try:
            from RL.genome_window import extract_gene_window
            result = extract_gene_window(fasta_path, gene_id, flank_size)
            if result is None:
                # Fallback to full genome if window extraction fails
                if genome_seq is None:
                    return []
                genome = genome_seq.upper().replace("U", "T")
            else:
                genome, window_start, window_end = result
                # genome is already uppercase and clean from extract_gene_window
        except Exception as e:
            import sys
            sys.stderr.write(f"[find_off_targets] Windowed search failed for {gene_id}: {e}, using full genome\n")
            if genome_seq is None:
                return []
            genome = genome_seq.upper().replace("U", "T")
    else:
        # Original mode: use provided genome_seq
        if genome_seq is None:
            return []
        genome = genome_seq.upper().replace("U", "T")
    
    n = len(genome)
    results = []
    # GG-only window: on minus strand PAM is reverse-complemented GG -> CC
    pam_suffixes_rc = ("CC",)
    if exclude_23 is not None:
        exclude_23 = exclude_23.upper()[:23]

    def hamming(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def add_if_not_self(w23: str) -> None:
        if exclude_23 and len(w23) == 23 and w23 == exclude_23:
            return
        results.append(w23)

    # + strand: genome layout [20 spacer][3 PAM], PAM ends with GG
    for i in range(n - 22):
        w = genome[i : i + 23]
        if len(w) != 23:
            continue
        if w[-2:] not in pam_suffixes:
            continue
        if 0 <= hamming(guide_20, w[:20]) <= max_mismatches:
            add_if_not_self(w)

    # - strand: genome layout [3 PAM][20 spacer]; after rev_comp => (20+3), PAM ends with CC
    for i in range(n - 22):
        w = genome[i : i + 23]
        w_rc = _rev_comp(w)
        if w_rc[-2:] not in pam_suffixes_rc:
            continue
        if 0 <= hamming(guide_20, w_rc[:20]) <= max_mismatches:
            add_if_not_self(w_rc)

    return list(dict.fromkeys(results))


def predict_crisprspec_surrogate(
    guide_20: str,
    off_targets_23: List[str],
    use_grna_folding: bool = False,
    pam_3: Optional[str] = None,
) -> Optional[float]:
    """
    Compute CRISPRspec surrogate score using energy/CRISPRspec_CRISPRoff_pipeline.
    guide_20: 20-mer gRNA spacer.
    off_targets_23: list of 23-mer off-target sequences (20 + PAM).
    pam_3: 3-letter PAM (e.g. "TGG", "AGG"). If None, uses "GGG" so guide_23 = 23-mer.
    Returns score in ~[0, 10] (higher = more specific). Returns None on error.
    """
    import math
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None

    try:
        from energy.CRISPRspec_CRISPRoff_pipeline import (
            read_energy_parameters,
            compute_CRISPRspec,
            calcRNADNAenergy,
        )
    except ImportError:
        try:
            sys.path.insert(0, _REPO_ROOT)
            from energy.CRISPRspec_CRISPRoff_pipeline import (
                read_energy_parameters,
                compute_CRISPRspec,
                calcRNADNAenergy,
            )
        except ImportError:
            return None

    energy_pkl = os.path.join(_REPO_ROOT, "energy", "energy_dics.pkl")
    if not os.path.isfile(energy_pkl):
        return None
    read_energy_parameters(energy_pkl)

    # 23-mer = 20 spacer + 3 PAM (previously +"GG" gave 22-mer and caused failures with off-targets)
    if pam_3 and len(pam_3) == 3 and all(c in "ACGT" for c in pam_3.upper()):
        guide_23 = guide_20 + pam_3.upper()
    else:
        guide_23 = guide_20 + "GGG"
    if len(guide_23) != 23:
        return None
    
    # Filter and validate off-targets: must be exactly 23-mer, only ACGT (no N or other IUPAC)
    # If genome contains N, all off-targets may be filtered out; use site-count heuristic instead of fixed 10.0
    valid_off_targets = []
    for ot in off_targets_23:
        ot_upper = ot.upper()
        if len(ot_upper) == 23 and all(c in "ACGT" for c in ot_upper):
            valid_off_targets.append(ot_upper)
    
    if not valid_off_targets:
        # No valid off-targets (e.g., all contain N). Use site-count-based score so metric still varies
        if not off_targets_23:
            return 10.0
        n = len(off_targets_23)
        return max(0.0, 10.0 - min(9.0, math.log10(1 + n)))
    
    ontarget = (guide_23, guide_23, "", 0, 0, "")
    off_seqs = [(ot, "") for ot in valid_off_targets]

    try:
        # Filter off-targets: ensure they have same length as guide_23 after RI_REV_NT_MAP transformation
        from energy.CRISPRspec_CRISPRoff_pipeline import RI_REV_NT_MAP
        filtered_off_seqs = []
        for ot_seq, chrom in off_seqs:
            if len(ot_seq) != len(guide_23):
                continue
            # Check transformed length - must match guide_23 length
            transformed = ''.join([RI_REV_NT_MAP.get(c, '') for c in ot_seq])
            if len(transformed) == len(guide_23):
                filtered_off_seqs.append((ot_seq, chrom))
        
        if not filtered_off_seqs:
            # No off-targets remain after RI_REV_NT_MAP filtering; score by number of input sites
            if not valid_off_targets:
                return 10.0
            n = len(valid_off_targets)
            return max(0.0, 10.0 - min(9.0, math.log10(1 + n)))
        
        on_prob, _ = compute_CRISPRspec(
            ontarget,
            filtered_off_seqs,
            calcRNADNAenergy,
            GU_allowed=False,
            pos_weight=True,
            pam_corr=True,
            grna_folding=use_grna_folding,
            dna_opening=True,
            dna_pos_wgh=False,
            ignored_chromosomes=set(),
        )
    except Exception as e:
        import sys
        sys.stderr.write(f"[predict_crisprspec_surrogate] Error: {e}\n")
        sys.stderr.write(f"  guide_23={guide_23} (len={len(guide_23)})\n")
        sys.stderr.write(f"  off_seqs count={len(off_seqs)}\n")
        if off_seqs:
            sys.stderr.write(f"  first off_seq={off_seqs[0][0]} (len={len(off_seqs[0][0])})\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
        return None

    # compute_CRISPRspec returns P_off (off-target probability). Higher means lower specificity.
    # score = -log10(P_off), capped at 10. To avoid saturation at high off-target counts,
    # apply a small penalty based on the number of off-targets.
    if on_prob <= 0:
        return 10.0
    raw = -math.log10(on_prob)
    penalty = min(2.0, math.log10(1 + len(filtered_off_seqs)) * 0.5)
    return max(0.0, min(10.0, raw - penalty))


def compute_hybridization_energy_single(guide_20: str) -> Optional[float]:
    """
    Compute hybridization energy (deltaG) for on-target: 20-mer guide vs self (perfect match).
    Used when use_energy=True in the env. Returns float or None on error.
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None
    try:
        from energy.CRISPRspec_CRISPRoff_pipeline import (
            read_energy_parameters,
            get_eng,
            calcRNADNAenergy,
        )
    except ImportError:
        try:
            sys.path.insert(0, _REPO_ROOT)
            from energy.CRISPRspec_CRISPRoff_pipeline import (
                read_energy_parameters,
                get_eng,
                calcRNADNAenergy,
            )
        except ImportError:
            return None
    energy_pkl = os.path.join(_REPO_ROOT, "energy", "energy_dics.pkl")
    if not os.path.isfile(energy_pkl):
        return None
    read_energy_parameters(energy_pkl)
    guide_23 = guide_20 + "GGG"
    try:
        dg = get_eng(
            guide_23,
            guide_23,
            calcRNADNAenergy,
            GU_allowed=False,
            pos_weight=False,
            pam_corr=False,
            grna_folding=False,
            dna_opening=False,
            dna_pos_wgh=False,
        )
    except Exception:
        return None
    return float(dg)


def get_on_target_and_energy(guide_20: str) -> Optional[Tuple[float, float]]:
    """
    Returns (on_target_binding_score, hybridization_dG_kcal_mol).
    binding_score: from get_eng with pos_weight, pam_corr, dna_opening (CRISPRoff-style).
    dG: hybridization energy in kcal/mol (negative = favorable).
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None
    try:
        from energy.CRISPRspec_CRISPRoff_pipeline import (
            read_energy_parameters,
            get_eng,
            calcRNADNAenergy,
        )
    except ImportError:
        try:
            sys.path.insert(0, _REPO_ROOT)
            from energy.CRISPRspec_CRISPRoff_pipeline import (
                read_energy_parameters,
                get_eng,
                calcRNADNAenergy,
            )
        except ImportError:
            return None
    energy_pkl = os.path.join(_REPO_ROOT, "energy", "energy_dics.pkl")
    if not os.path.isfile(energy_pkl):
        return None
    read_energy_parameters(energy_pkl)
    guide_23 = guide_20 + "GGG"
    try:
        dg = get_eng(
            guide_23, guide_23, calcRNADNAenergy,
            GU_allowed=False, pos_weight=False, pam_corr=False,
            grna_folding=False, dna_opening=False, dna_pos_wgh=False,
        )
        binding = get_eng(
            guide_23, guide_23, calcRNADNAenergy,
            GU_allowed=False, pos_weight=True, pam_corr=True,
            grna_folding=False, dna_opening=True, dna_pos_wgh=False,
        )
    except Exception:
        return None
    # Pipeline get_eng returns -sum(scores) → positive for favorable binding.
    # Thermodynamic ΔG (kcal/mol): negative = favorable; sweet spot [-64.53, -47.09].
    # So return dG_kcal = -dg so that good binding gives negative dG.
    dG_kcal = -float(dg)
    return (float(binding), dG_kcal)
