# Videos

All videos recorded with 1 env, headless rendering, 1000-step episodes.

## ROUGH Terrain (ROUGH_TERRAINS_CFG)

Task: Isaac-Velocity-Parkour-Adaptive-G1-Play-v0
Policy: HikingSafetyLite model_2999.pt

| File | Terrain | Level Range |
|------|---------|:---:|
| rough_easy.mp4 | ROUGH Easy | 0-2 |
| rough_medium.mp4 | ROUGH Medium | 0-5 |
| rough_hard.mp4 | ROUGH Hard | 0-8 |

## PARKOUR Terrain (PARKOUR_TERRAINS_CFG)

Task: Isaac-Velocity-HikingSafetyLite-G1-Play-v0
Policy: HikingSafetyLite model_2999.pt

Terrain types per row: 30% stairs, 25% gaps, 25% stepping stones, 20% boxes (randomly assigned).

| File | Terrain Row | Types (to identify) |
|------|:---:|------|
| parkour_level0.mp4 | 0 | stairs / gaps / stones / boxes |
| parkour_level2.mp4 | 2 | stairs / gaps / stones / boxes |
| parkour_level3.mp4 | 3 | stairs / gaps / stones / boxes |
| parkour_level5.mp4 | capped→4 | stairs / gaps / stones / boxes |
| parkour_level8.mp4 | capped→4 | stairs / gaps / stones / boxes |

Note: PLAY variant sets num_rows=5, so levels >= 5 are capped to row 4.
Terrain type in each video depends on random column assignment.
