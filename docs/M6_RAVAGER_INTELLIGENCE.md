# M6 — Ravager hunting, carcass transport and hibernation

**Status:** Implemented world rules, reward/observation extensions, role-specific PPO and reproducible evaluation. **Learning is partial**, not full unscripted hunting competence.

## Goal and behavior contract

Ravagers can choose to walk, rest, eat carrion, fight at melee range, and use five new mechanics:

1. **Sneak:** cardinal movement, extra stamina and calorie cost, and one-cell movement no more often than every two ticks. Cover reduces detectability at range; adjacent observers always see the ravager. Sneak never grants invisibility everywhere and ends when ordinary running or fighting occurs.
2. **Charge:** directional, visible wind-up for one tick, then committed movement up to three tiles. Other actors move before the charge resolves, so prey can dodge. On contact, the ravager hits the first non-ravager actor, stops before entering the occupied tile and suffers a recovery period. Walls stop the attack and inflict an injury. Charge uses 25 stamina plus 3 energy by default.
3. **Carcass relocation:** grab from the same or adjacent tile, drag as a single finite world object, drop, or eat. Carrying consumes stamina and halves practical movement rate. A carcass cannot be claimed by two actors and cannot be duplicated by moving or dropping it.
4. **Satiation:** actually eating meat raises ravager energy and satiety; no infinite feeding or abstract kill reward. The carcass decreases in size and decays normally.
5. **Hibernate and wake:** if sufficiently sated, the ravager may sleep and spend only a fraction of usual metabolic energy. It is not immune to attack; only wake is available while sleeping.

All states are serialized into `World.snapshot()` and deterministic replay hashes. Existing trained **cow** (60/66 inputs), **farmer** (76) and **nomad** (72/75) policy formats and action indices are unchanged. Ravager policies use a new fixed **82-feature** observation schema and **20 discrete actions**, validated when loading checkpoints.

## Visibility and world physics

A ravenous ravager is not directed by a fixed state machine: neural outputs choose all actions. The hand-written predator `scripted_actions` implementation is used **only as a reference baseline and as a scripted opponent for other roles**.

An observer can see a sneaking ravager in open land, close up, or at the edge of cover. Significant cover can obstruct a distant observer's detection; open farmland does not magically grant invisibility. The observation includes local prey, carcass scent, cover direction, defender density, stamina, satiety, and readiness/cooldown variables. Actions obey masks, but unsafe decisions remain possible.

**Collision resolution:** walking cows and nomads move before an already-wound-up charge is resolved, then the charging ravager advances along its previously selected direction until it hits the first blocker or reaches maximum range. The ravager cannot change direction mid-charge.

## Training and evaluation

All commands run on CPU without Docker, PostgreSQL or a GPU. PettingZoo is an optional adapter; the trainer uses the same dependency-light simultaneous multi-agent core and PyTorch.

```powershell
python -m pip install -e ".[ml,dev,viewer]"

# Train the first ravager on a reachable, vulnerable cow.
qox train-ravager --scenario ravager_ambush --steps 32768 --seed 119 --max-steps 48 --output runs/ravager_ambush.pt
qox benchmark-ravager --scenario ravager_ambush --checkpoint runs/ravager_ambush.pt --episodes 48 --seed 510000 --max-steps 48 --report runs/m6-ambush.json

# Isolate safe carcass recovery and feeding.
qox train-ravager --scenario ravager_carcass --steps 32768 --seed 74 --output runs/ravager_carcass.pt

# Isolate the reward for spending fewer calories while sated.
qox train-ravager --scenario ravager_hibernate --steps 8192 --seed 150 --output runs/ravager_hibernate.pt

# Combined ecosystem (other species scripted), with deterministic trace replay.
qox run --ticks 200 --mode learned-ravager --checkpoint runs/ravager_ambush.pt --trace runs/ravager-trace.json
qox replay --trace runs/ravager-trace.json
qox viewer --mode learned-ravager --checkpoint runs/ravager_ambush.pt
```

Curriculum worlds deliberately differ from the full ecosystem:
- **`ravager_ambush`:** one hungry ravager, one wounded cow, some cover and patchy grass; meat comes from a real kill.
- **`ravager_carcass`:** one finite carcass in defended territory, two stationary nomad sentries that threaten a predator approaching within melee range, and a cover patch farther away. A ravager can benefit by removing the carcass rather than feeding next to defenders.
- **`ravager_hibernate`:** one already-fed ravager with no local food or prey. Energy saved by successful sleep is measured at episode end.
- **`ravager_mixed`:** rotates between all three scenarios; varying populations change ravager agent IDs and are handled by the trainer. The PettingZoo adapter requires a fixed-population scenario.

Rewards are deliberately based on usable meat, actual charge damage, prey/resource approach, injury/survival, and energy at the end of an episode, **not a bonus for pressing sneak, charge, drag, or hibernate**. The scripted defender is an adversarial baseline, not training labels. Rollout seeds and evaluation seeds are disjoint. Actions use fixed schemas, and the evaluator reports per-episode outcomes and ablations.

## Benchmark results: 48 previously held-out world seeds

Evaluation seed range: **510000–510047**. The named specialist checkpoints were trained on separate seeds and tasks. For each scenario, all modes were evaluated on the same seeds with frozen model weights and the same tick limits. Greedy selects maximum-probability action; stochastic samples from the learned distribution. The random baseline samples legal actions, not all action indices.

### Ambush: 48 ticks per episode

| Policy | Charge hits / episode | Cow kill rate | Meat consumed / episode |
|---|---:|---:|---:|
| Learned ravager (greedy) | 0.3333 | **33.3%** | 3.9733 |
| Learned ravager (sampled) | 0.2708 | 27.1% | 2.7300 |
| Same policy, **charge disabled** | 0.0000 | 0% | 0.0000 |
| Random legal actions | 0.0417 | 4.2% | 0.1667 |
| Scripted reference | 0.9167 | 100% | 11.8800 |

The no-charge ablation removes the learned policy's successful hunts, which is evidence that it makes meaningful use of charge in the simplified ambush task. Removing sneak did **not** reduce greedy hunting success in this scenario; stealth has not been proven causally useful yet.

### Carcass recovery: 48 ticks per episode

| Policy | Meat consumed / episode | Carcass drag steps | Note |
|---|---:|---:|---|
| Learned carcass policy (greedy) | 11.9600 | 0.0000 | Usually feeds immediately and is killed by sentries |
| Learned carcass policy (sampled) | 10.4150 | 1.0417 | Occasional dragging; survival inconsistent |
| Random legal actions | 0.0417 | 0.1875 | Rare food retrieval |
| Scripted reference | 15.5042 | 6.8958 | Can also kill defenders and eat resulting carcasses |

**Interpretation:** Learnt food retrieval is promising, but **the policy has not learned safe relocation**. An agent can accumulate meat-reward before dying. This is an important benchmark failure, not a successful demonstration of sustained predatory intelligence.

### Satiated rest: 64 ticks per episode

| Policy | Hibernation ticks | Final energy / 100 |
|---|---:|---:|
| Learned hibernation policy (greedy) | 36.0 | **79.70** |
| Learned policy, **hibernate disabled** | 0 | 76.33 |
| Random legal actions | 9.19 | 73.12 |
| Scripted reference | 64.0 | 91.23 |

The policy learned to use hibernation in a simplified satiated episode and saved energy compared with removing the option, but does not fully optimize rest behavior.

## Validation and caveats

The implementation adds tests for the wind-up, dodge-before-impact, obstacle impact, stamina/recovery masks, corpse ownership and decay, limited stealth visibility, sleep/wake constraints, checkpoint load and legacy schema compatibility, as well as normal-world trace replay.

**No claim is made** that ravagers have learned the full sneak → charge → drag → feed → hibernate sequence under general conditions. That sequence is mechanically possible and the scripted reference demonstrates it, while neural policies currently master only pieces under controlled tasks. Stronger training curricula, multi-agent defense, and long-horizon evaluation remain research work.

The next milestone (M7) concerns scavenger competition for dung and carrion, although a focused ravager learning follow-up should target dragging and stealth in more varied layouts.
