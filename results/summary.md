# HikingSafetyLite 实验总结

## 实验设计

| 实验 | Terrain | Curriculum | Foothold Reward | Detector | 训练 |
|------|---------|-----------|:---:|------|------|
| Adaptive (baseline) | ROUGH_TERRAINS_CFG | adaptive | 无 | — | 3000 iter |
| HikingTerrainOnly | PARKOUR_TERRAINS_CFG | adaptive | 0 (metric-only) | Version A (binary) | 3000 iter |
| HikingSafetyLite | PARKOUR_TERRAINS_CFG | adaptive | +0.02/-0.02 | Version 2 (tri-class) | 3000 iter |

## 最终结果 (model_2999.pt, 50 episodes/terrain)

| Model | Easy | Medium | Hard | **Avg** | Safe TD Rate |
|-------|:----:|:------:|:----:|:-------:|:------------:|
| Adaptive (ROUGH) | 88.0% | 88.0% | 86.0% | **87.3%** | — |
| HikingTerrainOnly | 72.0% | 66.0% | 72.0% | **70.0%** | 88.4% |
| **HikingSafetyLite** | **86.0%** | **88.0%** | **94.0%** | **89.3%** | **92.6%** |

## 详细指标

### HikingTerrainOnly

| Terrain | Success | Mean Dist | Mean Time | Safe TD Count | Unsafe TD Count | Safe TD Rate | Fall Rate |
|---------|:-------:|-----------|-----------|:-------------:|:---------------:|:------------:|:---------:|
| Easy    | 72.0%   | 23.7m     | 32.3s     | 20,470        | 2,506           | 88.1%        | 28.0%     |
| Medium  | 66.0%   | 22.1m     | 30.1s     | 19,418        | 2,266           | 88.3%        | 34.0%     |
| Hard    | 72.0%   | 22.8m     | 31.3s     | 19,542        | 2,384           | 88.8%        | 28.0%     |

### HikingSafetyLite

| Terrain | Success | Mean Dist | Mean Time | Safe TD Count | Unsafe TD Count | Safe TD Rate | Fall Rate |
|---------|:-------:|-----------|-----------|:-------------:|:---------------:|:------------:|:---------:|
| Easy    | 86.0%   | 21.8m     | 36.7s     | 26,037        | 1,004           | 94.8%        | 14.0%     |
| Medium  | 88.0%   | 21.6m     | 36.3s     | 24,905        | 1,506           | 90.2%        | 12.0%     |
| Hard    | 94.0%   | 22.7m     | 38.5s     | 27,493        | 1,191           | 92.9%        | 6.0%      |

## 关键发现

1. **PARKOUR terrain 可学** — Adaptive curriculum 成功迁移，HikingTerrainOnly 70% avg success
2. **三分类 detector 稳定** — safe / unsafe / uncertain 分离后 safe TD rate 合理 (88-95%)
3. **极小权重 safety reward 有效** — +0.02/-0.02 不压制正常学习，反而提升了 success
4. **SafetyLite 全面最优** — 89.3% avg success + 92.6% safe TD rate

## SafetyLite Reward 配置

```
safe_touchdown_v2:   +0.02  (high-confidence safe touchdown)
unsafe_touchdown_v2: -0.02  (high-confidence unsafe touchdown)
swing_clearance:     -0.01  (swing foot terrain clearance)
stance_edge_risk:     0.00  (disabled)
```

## 文件说明

- `results/hiking_terrain_only/eval_*.csv` — TerrainOnly 50-episode eval per terrain level
- `results/hiking_safety_lite/eval_*.csv` — SafetyLite 50-episode eval per terrain level
- `results/summary.md` — 本文件
