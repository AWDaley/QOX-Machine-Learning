# M8 — Festivals, shared surplus, and predator attraction

## Status

M8 introduces **three functioning, distinct festivals** and an opt-in
reinforcement-learning curriculum for deciding whether to invest scarce
resources and when to initiate a celebration. The current results show that a
policy can learn the short *prepared festival* task; they do **not** prove
long-term autonomous prosperity, friendship, negotiation, or joint learned
cross-species planning.

## World rules

| Type | Site | Required finite resources | Required attendance |
|---|---|---|---|
| Farmer | Farmer's home settlement | Committed crops/grain | Healthy, exposed home farmers |
| Nomad | Nomad band's fixed camp | Committed milk | Healthy band nomads and cattle |
| Combined | Farmer settlement | Committed crops AND milk | Healthy home farmers, nomads, and cattle |

Amounts, duration, attendance radii, noise radius, and cooldowns are validated
configuration parameters. In a natural world the defaults are farmer grain 8,
nomad milk 4, combined grain 8 plus milk 4, two members of each needed role,
festival duration 8 ticks, and site cooldown 32 ticks.

**Resources:** Farmer `stored_food` remains a survival reserve. Explicit
`CONTRIBUTE_FESTIVAL` moves carried grain from farmers to the home settlement's
festival escrow, or milk from nomads to their camp or a physically visited
settlement. There is no free transfer from survival storage. Event start
atomically deducts required committed resources **once**, then reserves the
site for an event and a cooldown. Premature failure does not refund resources.

**Proposals:** Only eligible farmers and nomads can request an appropriate
festival at a physical site. Requirements are checked before contributions
later in the same tick, so a simultaneous contributor cannot retroactively
make a festival possible. Competing proposals for a single site are resolved
by a seeded lottery, not actor ID order. The policy must choose to propose;
meeting a threshold does not trigger celebration automatically.

**Duration and risk:** The event lasts several actual world steps. It requires
live exposed attendees, plus cattle for nomad or combined events. Loss of
quorum, or a ravager reaching the site entrance, fails the celebration.
Festival noise/smell originates at the real site and is attenuated by distance.
**No predators are spawned, teleported, or given global coordinates.** A
scripted ravager can use its locally available signal as a scent/navigation
cue; a separate opt-in feature exposes directional intensity to future ravager
networks. Other learned networks keep their older feature/action lengths.

**No magical behavior:** The rules do not directly command gathering,
protection, or cattle obedience. A transparent scripted baseline can perform
contribution/proposal tasks, but learned models make their own decisions.

## Model contracts

Festival policy training opts in via `festival_actions=True` (automatically
set for `festival_*` curricula). Farmer: 88 features and 16 actions. Nomad:
84 features and 18 actions. Ravager: 85 inputs if M8 festival scent features
are explicitly enabled. Legacy farmer, nomad, ravager, cow, and scavenger
contracts remain unchanged by default. Checkpoints record species, scenario,
feature ordering, action ordering, and schema version, with strict loading.

## Curriculum, rewards and limitations

Four deterministic scenario families:

- `festival_farmer`: a farmer has enough carried grain to finish funding an
  existing incomplete escrow, while another farmer is present.
- `festival_nomad`: a nomad contributes milk at a camp where a band member and
  cattle are present.
- `festival_combined`: a farmer and a **scripted** nomad contributor must
  jointly fund a combined festival.
- `festival_ravager`: like combined, but an existing ravager approaches from
  outside its initial prey-search range after sensing the event.

All scenarios initialize real carried resources and real escrow stock to
represent surplus accumulated before the short training episode. That is a
controlled training *fixture*, not evidence of learned farming/milking over a
season. The helper actors supply attendance, but **cannot start a festival for
the learner**. Success reward is issued only on actual event completion;
contribution receives small credit for committing finite resources; a valid
started event earns a smaller milestone reward. Failure is penalized, and
repeated start actions are cooldown/resource gated. Farm deposit reward is
reduced in the festival curriculum so survival storage isn't a more lucrative
way to exploit the old shaping objective.

### Held-out initial benchmarks

Train PPO with four CPU worlds, masked actions and two hidden layers of 64
neurons. Seeds used to evaluate are distinct from training seeds.

| Scenario | Training decisions | Evaluation episodes | Learned greedy success | Learned sampled success | Random success | Scripted success | No-start control |
|---|---:|---:|---:|---:|---:|---:|---:|
| Farmer | 16,384 | 48 | 100% | 87.5% | 8.3% | 100% | 0% |
| Nomad | 16,384 | 48 | 100% | 85.4% | 8.3% | 100% | 0% |
| Combined | 16,384 | 12 | 100% | 100% | 25% | 100% | 0% |

The combined episode has a scripted nomad partner. The three networks were
trained separately for their own scenario. **These results are an isolated
proof of learned short-term festival planning, not joint multi-group learning.**
No claim of ecological stability is made from the prepared scenarios. The
simulator also supports longer normal-world, scripted runs where production,
resource competition, predation, and festivals all interact; these are baseline
checks, not learned ecosystem outcomes.

### Safety and evaluation checks

- Resource escrow cannot be eaten by refuge occupants or reused to start two
  festivals; no negative balance or fictional income.
- Simultaneous proposals at a site resolve once; same-tick contributions
  cannot make a pre-invalid proposal valid.
- Quorum, cow attendance, age/duration, threat interruption, and cooldowns
  are enforced deterministically.
- Existing ravagers follow a local signal; no actor is inserted as a penalty.
- Old model feature/action ordering and deterministic replay remain valid.
- Evaluate policies on held-out seeds, report greedy **and** sampled behavior,
  and compare with scripted/random and no-start ablations.

## Commands

```bash
python -m pip install -e '.[dev,ml,viewer]'
qox festival-demo --scenario festival_combined --ticks 20
qox viewer --festival-scenario festival_combined --mode scripted
qox run --festivals --ticks 250 --mode scripted --report runs/ecology.json
qox train-festival --scenario festival_farmer --steps 16384 --output runs/festival_farmer.pt
qox benchmark-festival --scenario festival_farmer --checkpoint runs/festival_farmer.pt --episodes 24
qox viewer --festival-scenario festival_farmer --mode learned-festival --checkpoint runs/festival_farmer.pt
```

`--festivals` for `run`/`viewer` enables the M8 rules without modifying
existing baseline scenarios. `--festival-scenario` for the viewer starts a
small controlled world with positioned guests/resources; it is not a
persistent full-ecology run. Training a policy for one of these scenarios does
not imply transfer to other festival modes or a larger world.

## Next research work

1. Evaluate survival reserve vs festival commitment across **multiple real
   harvests**, without starting with nearly enough resources.
2. Alternate training farmer and nomad policies under a shared *combined*
   festival objective, rather than using a scripted partner.
3. Compare attack rates and successful celebrations when festival noise is
   disabled, different sizes of festival are held, and predators are trained
   on the attraction cue.
4. Evaluate the tradeoff between cultural/economic success and ecological
   persistence over many held-out seeds once reproduction is added in M9.
