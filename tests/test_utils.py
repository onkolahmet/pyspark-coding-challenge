from pyspark.sql import functions as F

from challenge.utils import ACTION_MISSING, pad_array


def test_pad_array_pad_and_truncate(spark):
    df = spark.createDataFrame([([1, 2],), ([1, 2, 3, 4],)], ["arr"])

    padded = df.select(pad_array(F.col("arr"), 5, ACTION_MISSING).alias("out")).collect()
    assert padded[0]["out"] == [1, 2, 0, 0, 0]
    assert padded[1]["out"] == [1, 2, 3, 4, 0]

    truncated = df.select(pad_array(F.col("arr"), 2, ACTION_MISSING).alias("out")).collect()
    assert truncated[0]["out"] == [1, 2]
    assert truncated[1]["out"] == [1, 2]
