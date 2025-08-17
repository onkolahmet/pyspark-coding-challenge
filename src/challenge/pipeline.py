"""
Pipeline for building training inputs from impressions + actions.

This version enforces a single source of truth for histories:
  actions_union -> histories -> training_examples

Outputs:
  - If explode=False (default): one row per (as_of_date, customer_id, ranking_id),
    with columns: impressions (array<struct<item_id:int,is_order:boolean>>),
                  actions (array<int>[max_actions]),
                  action_types (array<int>[max_actions]).
  - If explode=True: one row per impression item with
      (impression_item_id, is_order, actions, action_types).

Spec requirements enforced:
- Keep only actions STRICTLY BEFORE as_of_date (no same-day).
- Respect lookback_days window for actions.
- Sort actions by occurred_at DESC (tie-break: action_type DESC, item_id DESC).
- Cap at max_actions, right-pad with zeros.
"""

from pyspark.sql import DataFrame

from .actions_union import build_actions_union
from .history_builder import build_histories_for_impressions
from .impressions import normalize_impressions
from .training_examples import build_training_examples


def build_training_inputs(
    impressions_df: DataFrame,
    clicks_df: DataFrame,
    atc_df: DataFrame,
    orders_df: DataFrame,
    *,
    max_actions: int = 1000,
    lookback_days: int = 365,
    explode: bool = False,
) -> DataFrame:
    """End-to-end assembly using the canonical modules."""

    # 1) Normalize impressions (also serves as the contexts source)
    impr_norm = normalize_impressions(impressions_df)

    # 2) Union actions using spec-compliant schema (ints for action_types)
    actions_union = build_actions_union(clicks_df, atc_df, orders_df)

    # (Perf) Pre-partition by customer to reduce window shuffle later
    actions_union = actions_union.repartition("customer_id")
    impr_ctx = impr_norm.select("as_of_date", "customer_id").dropDuplicates()
    impr_ctx = impr_ctx.repartition("customer_id", "as_of_date")

    # 3) Build fixed-length histories (arrays of ints)
    histories = build_histories_for_impressions(
        actions_union, impr_ctx, max_actions=max_actions, lookback_days=lookback_days
    )

    # 4) Join histories back to impressions and (optionally) explode
    training = build_training_examples(impr_norm, histories, explode=explode)
    return training
