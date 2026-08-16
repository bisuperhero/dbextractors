# The Mage loader block

The package itself does not depend on Mage — there is a single lazy import of
`mage_ai`, used only to locate `io_config.yaml`. This page describes the block that
calls it, on the other side of that boundary: what it has to pass in, and what it
gets back.

One `data_loader` covers **all** database extractions in a repository — in one
deployment a single `data_loaders/dbx_extractor.py` serves 101 pipelines. Under Mage
the pipeline stays two blocks: a **config block** builds the dict, and this
**data_loader block** hands it to `run()`.

If you are not running under Mage, none of this applies: call
[`run()`](../README.md#configuration) directly.

---

## The block

The whole body is one call to `run()`:

```python
from dbextractors import run

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load_data(config, *args, **kwargs):
    dialect = str(config.get('DIALECT') or '').strip()
    if not dialect:
        raise ValueError(
            "The configuration is missing the 'DIALECT' key "
            "(firebird / postgres / mysql / mssql)."
        )
    # kwargs has to be passed through whole — it carries the pipeline's runtime
    # variables (forced_full_load, incremental_lookback_hours) and the Mage logger.
    return run(config, dialect=dialect, **kwargs)


@test
def test_output(output, *args) -> None:
    """Looks at every row, not just the first — the frame has one row per source."""
    assert output is not None, 'The output is undefined'
    failed = output[~output['success'].astype(bool)]
    assert not len(failed), f'Extraction failed for {len(failed)} sources'
```

`dialect` is `'mysql' | 'mssql' | 'postgres' | 'firebird'` (`'postgresql'` is taken
as an alias). The package takes it **as an argument to `run()`**, not from the
configuration — the `DIALECT` key is read by the calling block. A missing dialect is
meant to fail: previously it was determined by the extractor's file name, and a mix-up
would only have shown up at run time.

The logger is passed inside `kwargs`; passing it separately as well ends in
`got multiple values for keyword argument 'logger'`.

## What comes back

A `DataFrame` describing the run is returned — for a multi-source run, one row per
database:

| column | what it holds |
|---|---|
| `table`, `source` | the target `schema.table` and which source database |
| `rows_written`, `load_method`, `success`, `is_incremental` | as before |
| `data_present` | it ran, but did anything arrive? |
| `fallback_reason` | why the more expensive path was taken (fallback to a full load) |
| `connection_mode` | whether it went direct or through a tunnel |
| `error` | the failure text of a source that failed — in a frame you actually receive it is always `None`, see below |

Ten columns, always the same ten in the same order (`core.status.STATUS_COLUMNS`) —
a missing value is filled in as `None` rather than dropping the column, so one run
never returns a different shape from another. Compared with the predecessor's six
these are four extra columns with none missing, so an existing `test_output` passes
unchanged.

An unreachable source **fails the run with an exception**
(`SourceExtractionError`), not with a row carrying `success=False` — which is why
`error` and `success=False` exist in the shape but never reach the caller: a
multi-source run collects them so the message can name every source that failed,
and then raises.
