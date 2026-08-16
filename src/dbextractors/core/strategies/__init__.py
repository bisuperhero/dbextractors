"""Strategies — how a table gets from the source to the target.

How many tables each one serves in production today:

| strategy | when | tables |
|---|---|---|
| ``full`` | small tables, or no usable key | ~90 |
| ``hash_diff`` | no CDC log, but a stable PK | ~530 |
| ``incremental`` | a trustworthy change-timestamp column exists | ~27 |
| ``id_watermark`` | append-only, increasing PK | 4 |
| ``full_by_source`` | one target table assembled from several sources | 16 |
| ``parent_incremental`` | window derived from a parent table | 24 |

``hash_diff`` is both the dominant and the most expensive strategy — optimising
this one touches ~530 tables.
"""

from dbextractors.core.strategies.base import LoadContext, LoadResult, LoadStrategy, TargetRef

__all__ = ["LoadContext", "LoadResult", "LoadStrategy", "TargetRef"]
