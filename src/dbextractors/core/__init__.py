"""Package core — everything that does not depend on the type of the source database.

Separation of responsibilities, as described in ARCHITECTURE.md:

- **a dialect knows nothing about the target** — ``dbextractors.dialects``
- **a strategy knows nothing about the SQL dialect** — ``dbextractors.core.strategies``
- **``target_pg`` knows nothing about the source** — it is handed a DataFrame plus
  column metadata
"""
