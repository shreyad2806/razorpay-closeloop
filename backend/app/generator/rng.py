"""
Deterministic random number generator wrapper.

Ensures all random operations in the generator are reproducible given the same seed.
Uses Python's built-in random.Random with an independent state per generator.
"""

import random
from datetime import datetime, timedelta
from typing import Sequence


class DeterministicRNG:
    """
    Wrapper around random.Random for deterministic, reproducible generation.

    Each instance maintains its own state, so different generators can operate
    independently while still being reproducible from a single master seed.
    """

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        """Re-seed the RNG."""
        self._rng.seed(seed)

    def randint(self, min_val: int, max_val: int) -> int:
        """Generate a random integer in [min_val, max_val]."""
        return self._rng.randint(min_val, max_val)

    def uniform(self, min_val: float, max_val: float) -> float:
        """Generate a random float in [min_val, max_val]."""
        return self._rng.uniform(min_val, max_val)

    def choice(self, seq: Sequence) -> any:
        """Choose a random element from a non-empty sequence."""
        return self._rng.choice(seq)

    def choices(self, seq: Sequence, weights: Sequence[float] = None, k: int = 1) -> list:
        """Choose k elements with optional weights (with replacement)."""
        return self._rng.choices(seq, weights=weights, k=k)

    def shuffle(self, lst: list) -> None:
        """Shuffle a list in-place."""
        self._rng.shuffle(lst)

    def random_timestamp(self, start: datetime, end: datetime) -> datetime:
        """Generate a deterministic timestamp between start and end."""
        time_delta = end - start
        random_seconds = self._rng.randint(0, int(time_delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def random_amount(self, min_paise: int, max_paise: int) -> int:
        """Generate a random amount in paise within the given range."""
        return self._rng.randint(min_paise, max_paise)

    def random_percentage(self, min_pct: float, max_pct: float) -> float:
        """Generate a random percentage/rate value."""
        return self._rng.uniform(min_pct, max_pct)

    def should_trigger(self, probability: float) -> bool:
        """Returns True if a random event should trigger based on probability."""
        return self._rng.random() < probability
