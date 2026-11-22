# Controlled Craftax tasks

The manifest in `tasks.json` stages resources and neighboring terrain on
native seeded resets of the full Craftax symbolic environment. Each action
then uses the upstream Craftax transition. The tasks form a small integration
pilot with a deterministic recipe policy.

| Split | Task IDs | Skill or composition |
| --- | --- | --- |
| Train | `collect-wood`, `craft-wood-pickaxe`, `mine-stone`, `craft-stone-pickaxe` | Single thresholds and tool recipes |
| Validation | `wood-pickaxe-then-stone` | Make a wooden pickaxe and use it to mine stone |
| Test | `wood-to-stone-pickaxe` | Gather wood, make a wooden pickaxe, mine stone and upgrade it while retaining wood |

The test starts with two wood, zero stone and pickaxe tier zero. A tree is
directly ahead, stone is to the south and a crafting table is to the north.
The goal combines at least one wood and pickaxe tier two. The wood threshold
must also hold throughout execution. The reference action sequence is
`DO → MAKE_WOOD_PICKAXE → DOWN → DO → MAKE_STONE_PICKAXE`. Attempting to walk
into the stone changes the facing direction for the subsequent mining action.

## Task and observation contract

`inventory_at_least` refers to a numeric inventory field.
`tool_tier_at_least` refers to a pickaxe or sword tier.
`effects` are required together at completion, and `preserve` predicates
must hold at the start and after every action. A missing fact remains
unknown. The environment adapter supplies authoritative inventory for
checking. The model-facing input consists of the visible observation, the
public goal or current intent, and the controller's action memory.

The following is an excerpt of the test's initial observation. The adapter
also publishes its cropped terrain, visible items and entities, and the HUD.

```json
{
  "inventory": {"wood": 2, "stone": 0, "pickaxe": 0},
  "facing": "RIGHT",
  "neighbors": {"UP": "CRAFTING_TABLE", "RIGHT": "TREE", "DOWN": "STONE", "LEFT": "GRASS"}
}
```

The loader checks disjoint seeds and dependency chains across splits. Chain
identity includes milestone order, resource types and tool tiers. It
normalizes inventory thresholds, conjunct order and consecutive identical
milestones. The reference chain is split and evaluation metadata. The mock
generator and planner derive their instructions from known recipes, visible
inventory and the public goal.

## Plan-and-Act baseline

`plan-and-act-on-failure` requests an ordered list of typed intents before
acting. The same controller used in direct, one-step and full-search executes
the plan's current intent. Replanning occurs when a safe intent attempt fails
or the plan is exhausted. Unsafe execution stops the plan. Each plan request
and replan is charged to the shared logical call budget.

The variant takes the explicit planner/executor split from Erdogan et al.
Its deterministic recipe planner, intent-level replanning and environment
checks define the controlled comparison. The paper uses trained planner and
executor models, generated supervision, and dynamic replanning after executor
steps in web tasks. [Plan-and-Act, ICML 2025](https://proceedings.mlr.press/v267/erdogan25a.html)
