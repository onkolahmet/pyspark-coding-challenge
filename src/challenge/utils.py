# src/challenge/utils.py
from pyspark.sql import functions as F
from pyspark.sql.column import Column

# Action-type coding convention used across the project
ACTION_CLICK = 1
ACTION_ATC = 2
ACTION_ORDER = 3
ACTION_MISSING = 0  # for padding


def pad_array(arr_col: Column, size: int, fill_value) -> Column:
    """
    Right-pad or truncate an array column to exactly `size` elements.

    Implementation detail:
    - We concat with a fixed-length array of fill_value (length `size`)
      and then slice the first `size` elements. This avoids needing a
      dynamic `array_repeat` count (which would require a literal).
    - Works for any primitive element types compatible with `fill_value`.
    """
    fills = F.array(*[F.lit(fill_value) for _ in range(size)])
    return F.slice(F.concat(arr_col, fills), 1, size)
