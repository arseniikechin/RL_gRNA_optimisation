"""
Gym-compatible CRISPR gRNA optimization environment — proper RL (MDP).

MDP:
  State: observation = (20, 4) one-hot encoded gRNA sequence (Markov).
  Action: Discrete(80) = (position, nucleotide); action a -> position a//4, base BASES[a%4].
  Transition: If action (pos, base) and sequence[pos] != base and new_mismatches <= max_mismatches,
              then apply mutation; else no-op (state unchanged).
  Reward: Dense — incremental improvement at each step: reward = new_score - previous_score. Score = composite multi-objective function.
  Termination: When step_count >= max_steps (full trajectory).

No action substitution: if policy picks same nucleotide, step is no-op (correct RL).
Compatible with Stable-Baselines3 (PPO).
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any

import numpy as np

# Try gymnasium first, fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USING_GYMNASIUM = False

# Add repo root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


BASES = ["A", "C", "G", "T"]
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}


def seq_to_onehot(seq: str) -> np.ndarray:
    """Convert 20-mer string to (20, 4) one-hot float32 array."""
    out = np.zeros((20, 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in BASE_TO_ID:
            out[i, BASE_TO_ID[c]] = 1.0
    return out


def onehot_to_seq(arr: np.ndarray) -> str:
    """Convert (20, 4) or (80,) one-hot array to 20-mer string."""
    if arr.ndim == 1:
        arr = arr.reshape(20, 4)
    return "".join(BASES[int(np.argmax(arr[i]))] for i in range(20))


def compute_gc_content(seq: str) -> float:
    """GC content in [0, 1]. (G + C) / len(seq)."""
    if not seq:
        return 0.0
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s)


def compute_gc_term(seq: str) -> float:
    """Reward term for GC content: 1.0 at 50% GC, drops outside 40-60% (higher is better)."""
    gc = compute_gc_content(seq)
    # Prefer 40-60%: 1 - 4*(gc - 0.5)^2, clipped to [0, 1]
    term = 1.0 - 4.0 * (gc - 0.5) ** 2
    return max(0.0, min(1.0, term))


def compute_max_homopolymer(seq: str) -> int:
    """Length of longest run of the same nucleotide (e.g. AAAA -> 4)."""
    if not seq:
        return 0
    s = seq.upper()
    best = 1
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
        else:
            best = max(best, run)
            run = 1
    return max(best, run)


def compute_homopolymer_term(seq: str, max_acceptable: int = 3) -> float:
    """Reward term for homopolymers: 1.0 if max run <= max_acceptable, else penalize (higher is better)."""
    m = compute_max_homopolymer(seq)
    if m <= max_acceptable:
        return 1.0
    # Linear penalty: 4 -> 0.5, 5 -> 0, etc.
    return max(0.0, 1.0 - 0.5 * (m - max_acceptable))


def compute_gc_penalty(seq: str) -> float:
    """Penalty for GC deviation from 40-60%: 0 at 50%, up to 1 at 0% or 100%."""
    gc = compute_gc_content(seq)
    return 2.0 * abs(gc - 0.5)  # in [0, 1]


def compute_homopolymer_penalty(seq: str, max_acceptable: int = 3) -> float:
    """Penalty for runs >= 4 identical bases; optionally weighted by run length."""
    m = compute_max_homopolymer(seq)
    if m <= max_acceptable:
        return 0.0
    return min(1.0, 0.25 * (m - max_acceptable))  # 4->0.25, 5->0.5, 6->0.75, 7+->1.0


class CRISPRGymEnv(gym.Env):
    """
    Gymnasium environment for optimizing a 20-mer gRNA sequence.
    
    Observation Space: Box(0, 1, shape=(20, 4)) - one-hot encoded sequence
    Action Space: Discrete(n_mutable_positions * 4) - (position, nucleotide) pairs
    
    Reward: Dense — incremental improvement at each step (new_score - previous_score) to encourage improvement
    
    Seed Zone: Last `seed_len` positions (near PAM) are FIXED and cannot be mutated.
    Mismatch Limit: Episode ends when `max_mismatches` changes from original are made.
    """
    
    metadata = {"render_modes": ["human"]}
    LEN = 20
    
    def __init__(
        self,
        initial_sequences: Optional[list] = None,
        seed_len: int = 12,  # Default: fix last 12 positions (seed region near PAM)
        max_steps: int = 20,
        max_mismatches: int = 4,  # Max mutations from original (1-4 typically)
        min_mismatches: int = 1,  # Require at least 1 mutation
        use_crisprspec: bool = True,
        crisprspec_weight: float = 1.0,
        gc_weight: float = 0.1,
        homopolymer_weight: float = 0.1,
        genome_seq: Optional[str] = None,
        use_cuda_offtarget: bool = True,
        render_mode: Optional[str] = None,
        use_off_target: Optional[bool] = None,  # None = use use_crisprspec
        off_target_weight: Optional[float] = None,  # None = use crisprspec_weight
        use_gc_penalty: bool = True,
        gc_penalty_weight: float = 0.1,
        use_homopolymer_penalty: bool = True,
        homopolymer_penalty_weight: float = 0.1,
        reference_fasta_path: Optional[str] = None,
        disable_off_target: bool = False,
    ):
        """
        Parameters
        ----------
        initial_sequences : list of str, optional
            Pool of 20-mer sequences to sample from at reset(). If None, random.
        seed_len : int
            Number of FIXED positions at the END (seed region near PAM). Default 12.
            Mutable region = positions 0 to (20 - seed_len - 1).
        max_steps : int
            Maximum steps per episode.
        max_mismatches : int
            Maximum number of mutations allowed from original sequence. Episode ends when reached.
        min_mismatches : int
            Minimum mutations required. Penalty if episode ends with fewer.
        use_crisprspec : bool
            Use CRISPRspec (off-target specificity) as reward. Default True — reward is CRISPRspec only.
        gc_weight : float
            Weight for GC content term (prefer 40-60% GC).
        homopolymer_weight : float
            Weight for homopolymer term (penalize long runs of same base).
        genome_seq : str, optional
            Genome sequence for off-target search (required for CRISPRspec).
        """
        super().__init__()
        
        self.initial_sequences = initial_sequences or []
        # Seed zone is at the END (near PAM), so mutable region is at the START
        self.seed_len = max(0, min(seed_len, self.LEN))
        self.mutable_len = self.LEN - self.seed_len  # Positions 0 to mutable_len-1 can be changed
        self.max_steps = max_steps
        self.max_mismatches = max_mismatches
        self.min_mismatches = min_mismatches
        
        # Reward settings: only CRISPRspec (off-target) + GC + homopolymer
        self.use_crisprspec = use_crisprspec
        self.crisprspec_weight = crisprspec_weight
        self.gc_weight = gc_weight
        self.homopolymer_weight = homopolymer_weight
        self.reference_fasta_path = reference_fasta_path
        if reference_fasta_path and os.path.isfile(reference_fasta_path):
            try:
                from RL.grna_rl_adapters import load_genome
                self.genome_seq = load_genome(reference_fasta_path)
                print(f"  [CRISPRspec] Genome loaded: {len(self.genome_seq)} bp from {reference_fasta_path}")
            except Exception as e:
                import sys
                sys.stderr.write(f"[CRISPRspec] Failed to load genome from {reference_fasta_path}: {e}\n")
                self.genome_seq = genome_seq
        else:
            self.genome_seq = genome_seq
        
        # Validate: warn if off-target requested but no genome
        if (use_off_target if use_off_target is not None else use_crisprspec) and not self.genome_seq:
            import sys
            sys.stderr.write(
                f"[CRISPRspec] WARNING: off-target scoring requested but no genome loaded. "
                f"reference_fasta_path={reference_fasta_path}, genome_seq={'provided' if genome_seq else 'None'}. "
                f"CRISPRspec will be 0.\n"
            )
        
        self.use_cuda_offtarget = use_cuda_offtarget
        self._offtarget_fn = None

        self.use_off_target = (use_off_target if use_off_target is not None else use_crisprspec) and not disable_off_target
        self.off_target_weight = off_target_weight if off_target_weight is not None else crisprspec_weight
        self.use_gc_penalty = use_gc_penalty
        self.gc_penalty_weight = gc_penalty_weight
        self.use_homopolymer_penalty = use_homopolymer_penalty
        self.homopolymer_penalty_weight = homopolymer_penalty_weight

        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.LEN, 4), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.mutable_len * 4)
        
        # Internal state
        self.sequence: str = "A" * self.LEN
        self.original_sequence: str = "A" * self.LEN  # Track original for mismatch counting
        self.initial_score: float = 0.0  # For tracking initial score (used in info)
        self.step_count: int = 0
        self.previous_score: float = 0.0
        self._cached_components: Dict[str, float] = {}
    
    def _count_mismatches(self) -> int:
        """Count number of mismatches between current and original sequence."""
        return sum(1 for a, b in zip(self.sequence, self.original_sequence) if a != b)
    
    def _compute_score(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute the total score and component breakdown for current sequence.
        R = w_on*on_target + w_energy*energy_term + w_off*off_target - w_gc*gc_penalty - w_hp*homopolymer_penalty
        """
        components = {
            "on_target": 0.0,
            "energy_term": 0.0,
            "off_target_specificity": 0.0,
            "gc_penalty": 0.0,
            "homopolymer_penalty": 0.0,
            "deltaG_h": 0.0,
            "on_target_binding_raw": 0.0,
            "crisprspec": 0.0,
            "n_off_targets": 0,
            "gc_content": 0.0,
            "gc_term": 0.0,
            "max_homopolymer": 0,
            "homopolymer_term": 0.0,
        }
        
        components["gc_content"] = compute_gc_content(self.sequence)
        components["gc_penalty"] = compute_gc_penalty(self.sequence)
        components["gc_term"] = compute_gc_term(self.sequence)
        components["max_homopolymer"] = compute_max_homopolymer(self.sequence)
        components["homopolymer_penalty"] = compute_homopolymer_penalty(self.sequence)
        components["homopolymer_term"] = compute_homopolymer_term(self.sequence)
        
        # Off-target specificity (CRISPRspec)
        # Use genome_seq directly with CUDA or CPU off-target search.
        # For large genomes (>50 Mbp), skip CPU validation (too slow) and trust CUDA.
        if self.use_off_target and self.genome_seq:
            try:
                from RL.grna_rl_adapters import predict_crisprspec_surrogate
                genome_is_large = len(self.genome_seq) > 50_000_000
                
                if self._offtarget_fn is None:
                    from RL.grna_rl_adapters import find_off_targets_in_genome as _cpu_fn
                    
                    if self.use_cuda_offtarget:
                        try:
                            from RL.grna_rl_adapters_cuda import find_off_targets_in_genome_cuda, HAS_NUMBA_CUDA, HAS_CUDA
                            if HAS_CUDA and HAS_NUMBA_CUDA:
                                if genome_is_large:
                                    # Skip CPU validation on large genomes (would take minutes)
                                    self._offtarget_fn = find_off_targets_in_genome_cuda
                                    import sys
                                    sys.stderr.write(
                                        f"[Off-target] Large genome ({len(self.genome_seq)/1e6:.0f} Mbp) — "
                                        f"using CUDA directly (CPU validation skipped)\n"
                                    )
                                    sys.stderr.flush()
                                else:
                                    # Small genome: validate CUDA vs CPU
                                    test_guide = self.sequence
                                    test_exclude = test_guide + "NGG"
                                    cuda_results = find_off_targets_in_genome_cuda(
                                        test_guide, self.genome_seq, max_mismatches=4, exclude_23=test_exclude
                                    )
                                    cpu_results = _cpu_fn(
                                        test_guide, self.genome_seq, max_mismatches=4, exclude_23=test_exclude
                                    )
                                    import sys
                                    sys.stderr.write(
                                        f"[Off-target] Validation: CUDA found {len(cuda_results)}, "
                                        f"CPU found {len(cpu_results)} off-targets for {test_guide}\n"
                                    )
                                    sys.stderr.flush()
                                    if len(cpu_results) > 0 and len(cuda_results) == 0:
                                        sys.stderr.write(
                                            f"[Off-target] WARNING: CUDA missed all off-targets! Using CPU.\n"
                                        )
                                        self._offtarget_fn = _cpu_fn
                                    elif abs(len(cuda_results) - len(cpu_results)) > max(5, len(cpu_results) * 0.5):
                                        sys.stderr.write(
                                            f"[Off-target] WARNING: CUDA/CPU mismatch too large! Using CPU.\n"
                                        )
                                        self._offtarget_fn = _cpu_fn
                                    else:
                                        self._offtarget_fn = find_off_targets_in_genome_cuda
                            else:
                                self._offtarget_fn = _cpu_fn
                        except Exception as e:
                            import sys
                            sys.stderr.write(f"[WARNING] CUDA off-target search failed: {e}, using CPU\n")
                            self._offtarget_fn = _cpu_fn
                    else:
                        self._offtarget_fn = _cpu_fn
                    
                    fn_name = getattr(self._offtarget_fn, "__name__", str(self._offtarget_fn))
                    import sys
                    sys.stderr.write(f"[Off-target] Selected: {fn_name}, genome: {len(self.genome_seq)} bp\n")
                    sys.stderr.flush()

                # Initialize CRISPRspec cache: guide_20 -> (n_off_targets, crisprspec_score)
                if not hasattr(self, "_crisprspec_cache"):
                    self._crisprspec_cache = {}
                
                # Check CRISPRspec cache first
                cached = self._crisprspec_cache.get(self.sequence)
                if cached is not None:
                    components["n_off_targets"] = cached[0]
                    components["off_target_specificity"] = cached[1]
                    components["crisprspec"] = cached[1]
                else:
                    guide_23_exclude = self.sequence + "NGG"
                    off_targets = self._offtarget_fn(
                        self.sequence, self.genome_seq, max_mismatches=4, exclude_23=guide_23_exclude
                    )
                    components["n_off_targets"] = len(off_targets)
                    
                    # Limit off-targets passed to CRISPRspec: keep only closest matches
                    # (lowest mismatch count). CRISPRspec energy is O(n) per off-target.
                    # Top 100 by similarity captures the most dangerous off-targets.
                    off_targets_for_scoring = off_targets
                    if len(off_targets) > 100:
                        # Sort by hamming distance to guide, keep top 100
                        def _hamming(ot23):
                            return sum(1 for a, b in zip(self.sequence, ot23[:20]) if a != b)
                        off_targets_for_scoring = sorted(off_targets, key=_hamming)[:100]
                    
                    # Use default PAM "GGG" — avoids slow find_pam_in_genome scan on large genomes
                    # Real PAM only matters for on-target energy; for relative scoring "GGG" is fine
                    cs = predict_crisprspec_surrogate(self.sequence, off_targets_for_scoring, pam_3="GGG")
                    if cs is not None:
                        spec = cs / 10.0
                        components["off_target_specificity"] = spec
                        components["crisprspec"] = spec
                        # Cache result (limit cache size to avoid memory bloat)
                        if len(self._crisprspec_cache) < 10000:
                            self._crisprspec_cache[self.sequence] = (len(off_targets), spec)
                    
            except Exception as e:
                if not getattr(self, "_crisprspec_warned", False):
                    self._crisprspec_warned = True
                    import sys
                    sys.stderr.write(f"[CRISPRspec] Error: {e}\n")
                    import traceback
                    sys.stderr.write(traceback.format_exc())
                    sys.stderr.flush()
        elif self.use_off_target and not self.genome_seq:
            if not getattr(self, "_no_genome_warned", False):
                self._no_genome_warned = True
                import sys
                sys.stderr.write(
                    f"[CRISPRspec] WARNING: use_off_target=True but genome_seq is None! "
                    f"reference_fasta_path={self.reference_fasta_path}. "
                    f"CRISPRspec will be 0.0 (no off-target search possible).\n"
                )
                sys.stderr.flush()
        
        # Total reward: CRISPRspec (off-target) - GC penalty - homopolymer penalty
        total = (
            (self.off_target_weight * components["off_target_specificity"] if self.use_off_target else 0.0)
            - (self.gc_penalty_weight * components["gc_penalty"] if self.use_gc_penalty else 0.0)
            - (self.homopolymer_penalty_weight * components["homopolymer_penalty"] if self.use_homopolymer_penalty else 0.0)
        )
        components["total_score"] = total
        return total, components
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (one-hot encoded sequence)."""
        return seq_to_onehot(self.sequence)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs,
    ):
        """
        Reset environment to a new episode.
        
        If initial_sequences provided, samples from them.
        Otherwise generates a random sequence.
        
        Returns:
            - gymnasium: (obs, info)
            - gym: obs
        """
        if USING_GYMNASIUM:
            super().reset(seed=seed)
            rng = self.np_random
        else:
            # gym doesn't have seed in reset, use np.random
            if seed is not None:
                np.random.seed(seed)
            rng = np.random
        
        if options and "sequence" in options:
            # Allow specifying exact sequence
            self.sequence = options["sequence"].upper()
        elif self.initial_sequences:
            # Sample from pool
            if USING_GYMNASIUM:
                idx = rng.integers(0, len(self.initial_sequences))
            else:
                idx = rng.randint(0, len(self.initial_sequences))
            self.sequence = self.initial_sequences[idx].upper()
        else:
            # Random sequence
            if USING_GYMNASIUM:
                self.sequence = "".join(rng.choice(BASES) for _ in range(self.LEN))
            else:
                self.sequence = "".join(np.random.choice(BASES) for _ in range(self.LEN))
        
        self.step_count = 0
        self.original_sequence = self.sequence  # Store original for mismatch counting
        self.previous_score, self._cached_components = self._compute_score()
        self.initial_score = self.previous_score  # Store for tracking
        
        info = {
            "sequence": self.sequence,
            "original_sequence": self.original_sequence,
            "score": self.previous_score,
            "initial_score": self.initial_score,
            "components": self._cached_components.copy(),
            "mismatches": 0,
        }
        
        if USING_GYMNASIUM:
            return self._get_obs(), info
        else:
            # gym returns only obs
            return self._get_obs()
    
    def step(self, action: int):
        """
        RL step: execute action (no substitution). Dense reward at each step.
        
        Action (pos, base): if sequence[pos] != base and new_mismatches <= max_mismatches,
        apply mutation; else no-op. Reward = new_score - previous_score (incremental improvement).
        """
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action {action}, must be in [0, {self.action_space.n})")
        
        # Decode action — no substitution: policy's action is executed as-is (proper RL)
        pos = action // 4
        base_idx = action % 4
        new_base = BASES[base_idx]
        current_base = self.sequence[pos]
        
        # Transition: apply mutation only if (1) different base, (2) within max_mismatches; else no-op
        if self.sequence[pos] != new_base:
            new_seq = self.sequence[:pos] + new_base + self.sequence[pos + 1:]
            new_mismatches = sum(1 for a, b in zip(new_seq, self.original_sequence) if a != b)
            if new_mismatches <= self.max_mismatches:
                self.sequence = new_seq
        
        self.step_count += 1
        current_mismatches = self._count_mismatches()
        
        # Store previous score before computing new score (for dense reward)
        old_score = self.previous_score
        
        # Compute new score
        new_score, components = self._compute_score()
        self._cached_components = components
        
        # Termination: only when max_steps (full RL trajectory)
        reached_max_mismatches = current_mismatches >= self.max_mismatches
        reached_max_steps = self.step_count >= self.max_steps
        done = reached_max_steps
        
        # Dense reward: incremental improvement at each step
        reward = new_score - old_score
        
        # Additional penalty at episode end if min_mismatches not met
        if done and current_mismatches < self.min_mismatches:
            reward -= 0.5 * (self.min_mismatches - current_mismatches)
        
        # Update previous_score for next step
        self.previous_score = new_score
        
        info = {
            "sequence": self.sequence,
            "original_sequence": self.original_sequence,
            "score": new_score,
            "initial_score": self.initial_score,
            "components": components.copy(),
            "step": self.step_count,
            "mismatches": current_mismatches,
            "crisprspec": components.get("crisprspec", 0.0),
            "reached_max_mismatches": reached_max_mismatches,
        }
        # Track episode-level metrics for callback
        if done:
            # Episode total reward (sum of dense rewards)
            info["episode_reward"] = new_score - self.initial_score
            info["terminal_reward"] = reward
        
        if USING_GYMNASIUM:
            # gymnasium API: (obs, reward, terminated, truncated, info)
            return self._get_obs(), reward, reached_max_mismatches, reached_max_steps, info
        else:
            # gym API: (obs, reward, done, info)
            return self._get_obs(), reward, done, info
    
    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: {self.sequence}")
            print(f"  Score: {self.previous_score:.4f}")
            print(f"  Components: {self._cached_components}")
    
    def get_sequence(self) -> str:
        """Get current sequence."""
        return self.sequence
    
    def set_sequence(self, seq: str):
        """Set sequence (for evaluation)."""
        if len(seq) != self.LEN:
            raise ValueError(f"Sequence must be {self.LEN} nucleotides")
        self.sequence = seq.upper()


def make_training_env(
    sequences: Optional[list] = None,
    seed_len: int = 0,
    max_steps: int = 20,
    max_mismatches: int = 4,
    min_mismatches: int = 1,
    use_crisprspec: bool = False,
    genome_path: Optional[str] = None,
    gc_weight: float = 0.1,
    homopolymer_weight: float = 0.1,
    use_cuda_offtarget: bool = True,
    reference_fasta_path: Optional[str] = None,
    off_target_weight: Optional[float] = None,
    gc_penalty_weight: float = 0.1,
    homopolymer_penalty_weight: float = 0.1,
    disable_off_target: bool = False,
):
    """Factory to create a training environment (reward: CRISPRspec + GC + homopolymer only)."""
    genome_seq = None
    if not reference_fasta_path and genome_path and os.path.isfile(genome_path):
        from RL.grna_rl_adapters import load_genome
        genome_seq = load_genome(genome_path)
    return CRISPRGymEnv(
        initial_sequences=sequences,
        seed_len=seed_len,
        max_steps=max_steps,
        max_mismatches=max_mismatches,
        min_mismatches=min_mismatches,
        use_crisprspec=use_crisprspec,
        genome_seq=genome_seq,
        gc_weight=gc_weight,
        homopolymer_weight=homopolymer_weight,
        use_cuda_offtarget=use_cuda_offtarget,
        reference_fasta_path=reference_fasta_path,
        off_target_weight=off_target_weight,
        gc_penalty_weight=gc_penalty_weight,
        homopolymer_penalty_weight=homopolymer_penalty_weight,
        disable_off_target=disable_off_target,
    )


# Register with Gym/Gymnasium (optional)
# Uncomment if you want to use gym.make("CRISPRgRNA-v0")
# try:
#     gym.register(
#         id="CRISPRgRNA-v0",
#         entry_point="RL.grna_gym_env:CRISPRGymEnv",
#     )
# except Exception:
#     pass  # Already registered
