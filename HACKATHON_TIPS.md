# Hackathon Training Tips & Quick Reference

## 🚀 Quick Start
1. Start SAC training: `python sac_train.py`
2. Press **H** to toggle rendering (keep OFF during training for speed)
3. Watch the console for reward trends

## 📊 What to Watch During Training

### Good Signs ✅
- **Dog**: Average reward increasing, sheep saved going up (aim for 15-20+ sheep)
- **Wolf**: Average reward increasing, should start eating sheep after ~50-100 episodes
- **Alpha (α)**: Should stabilize between 0.05-0.3 after warmup

### Bad Signs ❌
- Rewards stuck at 0 or negative after 100+ episodes → reward function issue
- Rewards oscillating wildly → learning rate too high or reward scale wrong
- Agent moving in circles → needs more training or better shaping rewards

## 🎛️ Quick Hyperparameter Tuning

### If Dog isn't learning to herd:
```python
# In sac_train.py, dog_reward_fn:
# Increase shaping rewards:
reward += (1.0 - min(avg_sheep_dist / 400.0, 1.0)) * 0.25  # from 0.15
reward += progress * 0.15  # from 0.08
```

### If Wolf isn't learning to hunt:
```python
# In sac_train.py, wolf_reward_fn:
# Increase proximity reward:
reward += (1.0 - min(min_dist / 500.0, 1.0)) * 0.6  # from 0.4
# Reduce dog avoidance (make wolf braver):
reward -= (1.0 - dog_dist / 100.0) * 0.15  # from 0.25
```

### If training is too slow:
```python
# In sac_train.py, top section:
STEPS_PER_UPDATE = 2  # from 4 (more frequent updates)
WARMUP_STEPS = 3_000  # from 5_000 (start learning sooner)
```

### If training is unstable:
```python
# In sac_train.py, top section:
ACTOR_LR = 1e-4  # from 3e-4 (slower, more stable)
BATCH_SIZE = 256  # from 128 (smoother gradients)
STEPS_PER_UPDATE = 8  # from 4 (less frequent updates)
```

## 🎯 Reward Function Strategy

### Current Implementation:
- **Sparse rewards** (big events): 80-150 points
- **Dense shaping** (guidance): 0.01-0.4 points per step
- **Time penalty**: -0.01 per step

### Tuning Tips:
1. **Keep sparse >> dense**: Main objectives should dominate (100x bigger)
2. **Progress-based shaping** works better than position-based
3. **Don't over-reward proximity**: Can cause circling behavior

## 📈 Expected Performance Timeline

### SAC Training (1000 episodes):
- **Episodes 0-50**: Random exploration, filling replay buffer
- **Episodes 50-200**: Learning basic movements, occasional success
- **Episodes 200-500**: Clear strategy emergence, consistent behavior
- **Episodes 500-1000**: Fine-tuning and optimization

### Good Final Results:
- **Dog**: 10-20 sheep saved per episode, ~2000+ average reward
- **Wolf**: 5-10 sheep eaten per episode, ~800+ average reward

## 🔧 Advanced Optimizations

### If you have more time:

1. **Curriculum Learning**: Train dog alone first, then add wolf
```python
TRAIN_DOG = True
TRAIN_WOLF = False  # Set to True after episode 200
```

2. **Reward Scaling**: If one agent dominates, balance rewards
```python
# Make wolf rewards bigger if dog is too strong:
reward += float(sheep_eaten) * 200.0  # from 150.0
```

3. **Architecture Tweaks**: Already optimized, but you can try:
```python
hidden_dim=512  # from 384 (more capacity, slower)
BATCH_SIZE=256  # from 128 (more stable)
```

## 💾 Checkpointing Strategy

- Checkpoints saved every 20 episodes in `saves/sac/[timestamp]/`
- If training looks good at episode X, note it down
- To resume from checkpoint:
```python
LOAD_DOG_CHECKPOINT = True
DOG_CHECKPOINT_PATH = r"saves/sac/2025-XX-XX_XX-XX-XX/dog_sac_episode_XXX.pth"
```

## 🐛 Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Agents not moving | Check if HEADLESS is affecting observation, increase shaping rewards |
| Dog avoiding sheep | Increase proximity reward, decrease time penalty |
| Wolf dying too much | Increase dog avoidance penalty in wolf_reward_fn |
| Rewards exploding | Scale down all rewards by 0.5x |
| GPU out of memory | Reduce BATCH_SIZE to 64, or hidden_dim to 256 |

## 🏆 Competition Strategy

1. **Train both agents**: You want a good demo with conflict
2. **Prioritize dog first**: Herding is harder to learn than hunting
3. **Balance the challenge**: Wolf shouldn't be too easy or impossible
4. **Visual appeal matters**: Emergent behaviors are impressive (dog protecting, wolf stalking)

## 📝 Pre-Competition Checklist

- [ ] Dog can herd at least 10 sheep consistently
- [ ] Wolf can hunt and eat sheep (but not dominate completely)
- [ ] Interesting behaviors emerge (dog protecting, wolf stalking)
- [ ] Saved best checkpoints with clear naming
- [ ] Tested checkpoint loading works
- [ ] Rendering works smoothly for demo

---

**Good luck! 🎉 Focus on getting basic behaviors working first, then optimize.**

