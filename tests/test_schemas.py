# tests/test_schemas.py
import pytest
from pyspark.sql import types as T

import challenge.schemas as S


def _iter_schemas(module):
    """Yield (name, schema) for any StructType exported by the module."""
    for name in dir(module):
        if name.startswith("_"):
            continue
        val = getattr(module, name)
        if isinstance(val, T.StructType):
            yield name, val


def _has_fields(schema: T.StructType, required: set[str]) -> bool:
    return required.issubset({f.name for f in schema.fields})


def test_module_exports_structtypes():
    """Ensure the schemas module actually defines at least one StructType."""
    schemas = list(_iter_schemas(S))
    assert len(schemas) > 0, "challenge.schemas defines no StructType exports"


def test_all_schemas_have_unique_field_names():
    """No duplicate field names inside any defined schema."""
    for name, schema in _iter_schemas(S):
        names = [f.name for f in schema.fields]
        assert len(names) == len(set(names)), f"Duplicate fields in {name}"


def test_training_like_schema_shapes_and_types():
    """
    Find the schema that looks like the TRAINING OUTPUT (whatever its exact export name is)
    by matching a characteristic field set, then validate key field types.
    """
    target = None
    required = {
        "ranking_id",
        "customer_id",
        "impressions",
        "actions",
        "action_types",
        "as_of_date",
    }
    for name, schema in _iter_schemas(S):
        if _has_fields(schema, required):
            target = (name, schema)
            break

    if target is None:
        pytest.skip(
            "No training-like schema found (fields: ranking_id, customer_id, impressions, actions, action_types, as_of_date)"
        )

    name, schema = target
    fields = {f.name: f.dataType for f in schema.fields}

    # as_of_date should be DateType
    assert isinstance(fields["as_of_date"], T.DateType), f"{name}.as_of_date must be DateType"

    # ranking_id should be StringType
    assert isinstance(fields["ranking_id"], T.StringType), f"{name}.ranking_id must be StringType"

    # customer_id should be LongType/BigInt
    assert isinstance(fields["customer_id"], T.LongType), f"{name}.customer_id must be LongType"

    # impressions should be Array<Struct<item_id:int, is_order:boolean>>
    assert isinstance(fields["impressions"], T.ArrayType), f"{name}.impressions must be ArrayType"
    elem = fields["impressions"].elementType
    assert isinstance(elem, T.StructType), f"{name}.impressions element must be StructType"
    inner = {f.name: f.dataType for f in elem.fields}
    assert isinstance(inner.get("item_id"), T.IntegerType), f"{name}.impressions.item_id must be IntegerType"
    assert isinstance(inner.get("is_order"), T.BooleanType), f"{name}.impressions.is_order must be BooleanType"

    # actions, action_types should be Array<Integer>
    for col in ("actions", "action_types"):
        assert isinstance(fields[col], T.ArrayType), f"{name}.{col} must be ArrayType"
        assert isinstance(fields[col].elementType, T.IntegerType), f"{name}.{col} elements must be IntegerType"


def test_actions_union_like_schema_time_columns():
    """
    Locate schema that looks like the unified actions schema by the presence
    of occurred_at and action_date, then validate their types.
    """
    target = None
    required = {"customer_id", "item_id", "action_type", "occurred_at", "action_date"}
    for name, schema in _iter_schemas(S):
        if _has_fields(schema, required):
            target = (name, schema)
            break

    if target is None:
        pytest.skip(
            "No actions-union-like schema found (fields: customer_id, item_id, action_type, occurred_at, action_date)"
        )

    name, schema = target
    fields = {f.name: f.dataType for f in schema.fields}

    assert isinstance(fields["occurred_at"], T.TimestampType), f"{name}.occurred_at must be TimestampType"
    assert isinstance(fields["action_date"], T.DateType), f"{name}.action_date must be DateType"

    # sanity on id types
    assert isinstance(fields["customer_id"], T.LongType), f"{name}.customer_id must be LongType"
    assert isinstance(fields["item_id"], T.IntegerType), f"{name}.item_id must be IntegerType"
    assert isinstance(fields["action_type"], T.IntegerType), f"{name}.action_type must be IntegerType"
