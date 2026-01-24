# Gradient Optimization for HyperGraph Router

This document describes gradient estimation methods for the discrete routing decision in HyperGraph Sparse Attention.

## The Problem

The router assigns each token to one of K timelines (hard assignment). But `argmax` has zero gradient everywhere:

```
Forward:  soft_probs → argmax → hard_assignment → sparse_attention
Backward: ∂hard/∂soft = 0 (step function has no gradient)
```

We need gradient estimation to train the router.

---

## Method Comparison

| Method | Gradient Quality | Stability | Complexity | xformers Compatible |
|--------|------------------|-----------|------------|---------------------|
| **STE (current)** | ⚠️ Biased (~15%) | ⚠️ Can oscillate | ✅ Simple | ✅ Yes |
| **Gumbel-Softmax** | ✅ Less biased | ✅ Stable | ✅ Simple | ✅ Yes |
| **Temperature Annealing** | ✅ Good | ✅ Stable | ⚠️ Medium | ✅ Yes |
| **Soft Train/Hard Infer** | ✅ Exact | ✅ Very stable | ⚠️ Medium | ❌ No (soft breaks it) |
| **REINFORCE** | ⚠️ High variance | ❌ Unstable | ❌ Complex | ✅ Yes |

---

## 1. Straight-Through Estimator (STE) — CURRENT

```python
def _straight_through_hard(self, node_probs: torch.Tensor) -> torch.Tensor:
    """
    Forward: hard assignment (argmax → one-hot)
    Backward: gradient flows through soft probabilities
    """
    hard = F.one_hot(node_probs.argmax(dim=-1), self.num_hyper_nodes).float()
    return hard - node_probs.detach() + node_probs
```

**How it works:**
- Forward: `hard - soft.detach() + soft = hard` (soft terms cancel)
- Backward: gradient of `soft` (hard and detached have no grad)

**Pros:** Simple, no hyperparameters, zero overhead
**Cons:** Biased gradient, router may not learn optimal policy

---

## 2. Gumbel-Softmax

```python
def gumbel_softmax_hard(logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    Differentiable approximation to categorical sampling.
    
    Args:
        logits: Router logits (batch, heads, seq, K)
        tau: Temperature (lower = sharper, typically 0.1-1.0)
    
    Returns:
        Hard one-hot with soft gradients
    """
    # Sample Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    
    # Soft sample
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1)
    
    # Hard sample with STE
    y_hard = F.one_hot(y_soft.argmax(-1), logits.size(-1)).float()
    return y_hard - y_soft.detach() + y_soft
```

**Pros:** Stochastic exploration, better gradient signal, standard technique
**Cons:** Still uses STE for hard part, τ hyperparameter to tune

**Recommended τ values:**
- τ = 1.0: Soft, good for exploration
- τ = 0.5: Balanced
- τ = 0.1: Near-hard, matches inference

---

## 3. Temperature Annealing ⭐ RECOMMENDED

```python
class AnnealedRouter:
    def __init__(self, tau_start=1.0, tau_end=0.1, anneal_steps=1000):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
    
    def get_tau(self) -> float:
        """Get current temperature based on training progress."""
        progress = min(self.current_step / self.anneal_steps, 1.0)
        return self.tau_start - (self.tau_start - self.tau_end) * progress
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        tau = self.get_tau()
        
        if tau > 0.2:
            # Soft regime: use Gumbel-Softmax for exploration
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            y_soft = F.softmax((logits + gumbel) / tau, dim=-1)
        else:
            # Hard regime: standard softmax + STE
            y_soft = F.softmax(logits / tau, dim=-1)
        
        # Always return hard assignment (for xformers compatibility)
        y_hard = F.one_hot(y_soft.argmax(-1), logits.size(-1)).float()
        return y_hard - y_soft.detach() + y_soft
    
    def step(self):
        """Call after each training step."""
        self.current_step += 1
```

**Schedule:**
```
Step 0-30%:   τ=1.0 → 0.7   Exploration, good gradients
Step 30-70%:  τ=0.7 → 0.3   Sharper routing
Step 70-100%: τ=0.3 → 0.1   Matches inference
```

**Pros:** Best of both worlds, good early gradients, converges to hard
**Cons:** Needs scheduler, more hyperparameters

---

## 4. Soft Training / Hard Inference

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.router(x)
    probs = F.softmax(logits, dim=-1)  # (batch, heads, seq, K)
    
    if self.training:
        # Soft attention: weighted mixture over all timelines
        # EXACT gradients, but O(N²) compute (no speedup)
        outputs = []
        for k in range(K):
            # Compute attention for timeline k
            attn_k = self.compute_attention_for_timeline(x, k)
            outputs.append(attn_k)
        
        # Weighted sum: output = Σ_k probs[k] * attn_k
        output = sum(probs[..., k:k+1] * out for k, out in enumerate(outputs))
        return output
    else:
        # Hard attention: sparse, O(N²/K) compute
        return self.hard_sparse_attention(x, probs.argmax(-1))
```

**Pros:** Exact gradients, router learns optimal policy
**Cons:** 
- Train/inference gap (soft ≠ hard behavior)
- Training loses O(N²/K) speedup (computes all K timelines)
- Incompatible with xformers BlockDiagonalMask

---

## 5. REINFORCE (Policy Gradient)

```python
class REINFORCERouter:
    def __init__(self, baseline_momentum=0.99):
        self.baseline = 0.0
        self.momentum = baseline_momentum
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action (timeline assignment)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Compute output with hard assignment
        output = self.hard_sparse_attention(x, action)
        
        return output, log_prob
    
    def compute_loss(self, log_prob: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Call after computing task loss.
        reward = -task_loss (negative because we maximize reward)
        """
        # Update baseline (moving average of rewards)
        with torch.no_grad():
            self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward.mean()
        
        # Policy gradient with baseline
        advantage = reward - self.baseline
        policy_loss = -(log_prob * advantage.detach()).mean()
        
        # Optional: entropy bonus for exploration
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        
        return policy_loss - 0.01 * entropy
```

**Pros:** Unbiased gradient, true RL formulation, can optimize any reward
**Cons:** High variance, needs baseline/critic, complex implementation

---

## Implementation Priority

### Phase 1: Quick Improvement (Gumbel-Softmax)
Add Gumbel noise to current STE for better exploration:
```python
# In _straight_through_hard():
if self.training:
    gumbel = -torch.log(-torch.log(torch.rand_like(node_probs) + 1e-10) + 1e-10)
    node_probs = F.softmax((node_logits + gumbel) / self.tau, dim=-1)
```

### Phase 2: Full Implementation (Temperature Annealing)
1. Add `tau` parameter to HyperGraphSparseAttention
2. Add `training_step` tracking
3. Implement annealing schedule
4. Update training loop to call `router.step()`

### Phase 3: Research (Soft Training)
For maximum quality at cost of training speed:
1. Implement soft mixture attention path
2. Compare perplexity vs STE
3. Measure train/inference gap

---

## Hyperparameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `tau_start` | 1.0 | Soft start for exploration |
| `tau_end` | 0.1 | Near-hard to match inference |
| `anneal_steps` | 30-50% of total | Don't anneal too fast |
| `gumbel_noise` | True during anneal | Disable when τ < 0.2 |

---

## References

1. **STE**: Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (2013)
2. **Gumbel-Softmax**: Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
3. **Temperature Annealing**: Common in VQ-VAE, discrete diffusion
4. **REINFORCE**: Williams, "Simple Statistical Gradient-Following Algorithms" (1992)

---

## Quick Reference: Adding Gumbel-Softmax

Minimal change to current implementation:

```python
# In hypergraph_attention.py

def _compute_node_probs(self, x: torch.Tensor, use_gumbel: bool = True) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    
    node_logits = self.node_router(x)
    node_logits = node_logits.view(batch, seq_len, self.num_heads, self.num_hyper_nodes)
    node_logits = node_logits.transpose(1, 2)
    
    if self.training and use_gumbel:
        # Add Gumbel noise for better exploration
        gumbel = -torch.log(-torch.log(torch.rand_like(node_logits) + 1e-10) + 1e-10)
        return F.softmax((node_logits + gumbel) / self.tau, dim=-1)
    else:
        return F.softmax(node_logits, dim=-1)
```

---

*Last updated: January 2026*
*Status: STE implemented, Gumbel/Annealing ready for Phase 2*

