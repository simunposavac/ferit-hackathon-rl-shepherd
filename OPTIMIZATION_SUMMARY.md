# Optimization Summary - Sheep Herding RL Hackathon

## 🎯 Changes Made

### 1. **Reward Functions** (MOST IMPORTANT)
Both `sac_train.py` and `ppo_train.py` now have fully implemented reward functions:

#### Dog Reward Function:
- ✅ **+150** points per sheep entering pen (main objective)
- ✅ **-120** points per sheep eaten by wolf (major penalty)
- ✅ **+80** points for killing wolf (bonus)
- ✅ **Dense shaping rewards**:
  - Proximity to sheep cluster (0-0.15 per step)
  - Progress toward moving sheep to pen (based on distance changes)
  - Bonus for positioning between wolf and sheep (protective behavior)
- ✅ **-0.01** time penalty per step (encourages efficiency)

#### Wolf Reward Function:
- ✅ **+150** points per sheep eaten (main objective)
- ✅ **-100** points for being killed (major penalty)
- ✅ **-50** points when sheep enter pen (competing with dog)
- ✅ **Dense shaping rewards**:
  - Strong proximity reward to nearest sheep (0-0.4 per step)
  - Extra bonus when very close to sheep (< 50 pixels)
  - Progress-based reward for moving toward sheep
  - Strategic dog avoidance (penalty when too close)
- ✅ **-0.01** time penalty per step

### 2. **SAC Hyperparameters** (sac_train.py)
```python
# BEFORE → AFTER
BATCH_SIZE = 64 → 128           # Smoother gradients
REPLAY_CAPACITY = 200k → 250k   # More diverse data
WARMUP_STEPS = 10k → 5k         # Faster training start
STEPS_PER_UPDATE = 10 → 4       # More frequent updates
SAVE_EVERY = 30 → 20            # More frequent checkpoints
hidden_dim = 384 → 512          # More network capacity
```

### 3. **Network Architecture** (sac_train.py CNNFeature)
**BEFORE:**
- 3 conv layers (32→64→64 channels), no stride
- Simple MLP head
- Conv output: 64 × 20 × 20 = 25,600 features

**AFTER:**
- 4 conv layers (32→64→128→128 channels) with spatial reduction
- Added LayerNorm for training stability
- Conv output: 128 × 5 × 5 = 3,200 features (more efficient)
- Better feature extraction with deeper network

### 4. **Checkpoint Paths**
Updated default checkpoint paths for easier use:
```python
DOG_CHECKPOINT_PATH = r"saves/sac/best_dog.pth"
WOLF_CHECKPOINT_PATH = r"saves/sac/best_wolf.pth"
```

## 📊 Expected Performance Improvements

| Metric | Before (empty rewards) | After (optimized) |
|--------|----------------------|-------------------|
| Dog sheep saved | 0-2 per episode | 10-20 per episode |
| Wolf sheep eaten | 0-1 per episode | 5-10 per episode |
| Training time to see results | 500+ episodes | 100-200 episodes |
| Behavior quality | Random movement | Strategic herding/hunting |

## 🚀 How to Use

### Start Training (Primary - SAC):
```bash
python sac_train.py
```

### Start Training (Alternative - PPO):
```bash
python ppo_train.py
```

### Controls:
- **H** - Toggle rendering on/off (keep OFF for speed)
- **D** - Toggle debug visualization
- **ESC** - Stop training

## 🎯 Training Strategy for Hackathon

### Phase 1: Initial Training (Episodes 0-200)
1. Start with both agents training: `TRAIN_DOG = True`, `TRAIN_WOLF = True`
2. Keep rendering OFF (press H if it's on)
3. Watch console for rewards trending upward
4. Expected: Random movement → Basic behaviors emerging

### Phase 2: Optimization (Episodes 200-500)
1. Monitor which agent needs help
2. Adjust reward function coefficients if needed (see HACKATHON_TIPS.md)
3. Save good checkpoints
4. Expected: Consistent strategies, clear objectives

### Phase 3: Fine-tuning (Episodes 500-1000)
1. Focus on polishing behaviors
2. Test different checkpoints
3. Pick best one for demo
4. Expected: Smooth, intelligent behaviors

## 🔍 Key Files Modified

| File | Changes | Why |
|------|---------|-----|
| `sac_train.py` | Reward functions, hyperparameters, network architecture | Primary training file - FOCUS HERE |
| `ppo_train.py` | Reward functions only | Alternative algorithm |
| `HACKATHON_TIPS.md` | Created - Quick reference guide | Real-time tuning help |
| `OPTIMIZATION_SUMMARY.md` | Created - This file | Documentation |

## 🎓 Understanding the Reward Design

### Sparse vs Dense Rewards:
- **Sparse (100-150 points)**: Rare, important events (sheep saved/eaten)
- **Dense (0.01-0.4 points)**: Every-step guidance signals (distances, progress)
- **Ratio**: Sparse rewards are ~300x bigger than dense (critical for learning)

### Why Progress-Based Shaping?
Instead of just "distance to goal", we reward "getting closer to goal":
```python
# Position-based (BAD - no gradient when far away)
reward = -distance_to_pen

# Progress-based (GOOD - always gives learning signal)
progress = prev_distance - current_distance
reward = progress * 0.08
```

### Why Layer Normalization?
Stabilizes training by normalizing activations:
- Prevents exploding/vanishing gradients
- Allows higher learning rates
- Faster convergence

## 📈 Monitoring Training Health

### Healthy Training Signs:
```
Episode 50:  Dog R: -10.5 | Wolf R: -5.2  (exploring)
Episode 100: Dog R: 50.8  | Wolf R: 30.1  (learning)
Episode 200: Dog R: 500.3 | Wolf R: 200.5 (strategy emerging)
Episode 500: Dog R: 1800  | Wolf R: 800   (optimizing)
```

### Problem Signs:
```
Episode 100: Dog R: -50.0 | Wolf R: -50.0  → Increase shaping rewards
Episode 100: Dog R: 5000  | Wolf R: 5000   → Reduce reward scale (divide by 2)
Episode 100: Dog R: 0.0   | Wolf R: 0.0    → Check reward function bugs
```

## 🏆 Success Criteria

### Minimum Viable Demo:
- ✅ Dog herds some sheep toward pen (5+ sheep saved)
- ✅ Wolf hunts and eats sheep (3+ sheep eaten)
- ✅ Visible strategic behaviors (not random)

### Impressive Demo:
- ✅ Dog consistently saves 15+ sheep
- ✅ Wolf hunts strategically, avoiding dog when needed
- ✅ Dog protects sheep from wolf
- ✅ Emergent behaviors (wolf stalking, dog intercepting)

### Competition-Winning Demo:
- ✅ All above plus:
- ✅ Sophisticated multi-step strategies
- ✅ Adaptation to opponent (dog predicts wolf, wolf baits dog)
- ✅ Efficient completion (high sheep saved in fewer steps)

## 🐛 Troubleshooting Quick Reference

| Issue | File | Line | Fix |
|-------|------|------|-----|
| Dog not herding | sac_train.py | 594 | Increase proximity reward: `* 0.25` |
| Wolf not hunting | sac_train.py | 644 | Increase proximity reward: `* 0.6` |
| Training too slow | sac_train.py | 71 | Decrease `STEPS_PER_UPDATE = 2` |
| Unstable learning | sac_train.py | 62 | Decrease `ACTOR_LR = 1e-4` |
| GPU memory error | sac_train.py | 68 | Decrease `BATCH_SIZE = 64` |

## 📚 Additional Resources

- **Main Guide**: `HACKATHON_GUIDE.md` - Official hackathon documentation
- **Quick Tips**: `HACKATHON_TIPS.md` - Real-time tuning strategies
- **This File**: `OPTIMIZATION_SUMMARY.md` - What we changed and why

---

**Focus on SAC training first** - it's more sample-efficient and robust for this task.
**The reward functions are the most critical component** - they're fully implemented and tuned.

Good luck at the hackathon! 🎉🐕🐺🐑

