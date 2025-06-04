# models/super_token.py
import dataclasses
import uuid
from typing import Optional, List, Any
import numpy as np
import gmpy2 # For path_code_c type hint

@dataclasses.dataclass
class SuperToken:
    claim: str # Human-readable description of the derived knowledge
    path_node_labels: List[str] # The sequence of node labels forming the path
    edge_atomic_primes: List[Any] # The sequence of small atomic primes for the edges (gmpy2.mpz)
    path_code_c: Any # The Gödel number (C) (gmpy2.mpz)
    path_code_d: int     # The depth of encoding (d), initially 0
    id: str = dataclasses.field(default_factory=lambda: f"st_{uuid.uuid4()}")
    # Optional fields for later:
    # embedding: Optional[np.ndarray] = None
    # confidence_score: Optional[float] = None
    # energy_complexity: Optional[float] = None
    # constituent_super_tokens: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self):
        return f"SuperToken(id='{self.id}', claim='{self.claim[:30]}...', path_code_c={self.path_code_c}, depth={self.path_code_d})" 