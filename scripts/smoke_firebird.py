"""Checks that firebird+fdb exists at all on the production Mage image.

SQLAlchemy 1.4 marked its Firebird dialect as deprecated and removed it in 2.0.
If 1.4.54 no longer carried it, `FirebirdDialect` would be dead from day one and
nobody would find out until the first night in production. No connection is
attempted here - this only checks that the URL parses and that the dialect finds
its driver.
"""

import warnings

import fdb
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

from dbextractors.core import secrets
from dbextractors.dialects.firebird import FirebirdDialect

print("SQLAlchemy:", sqlalchemy.__version__)
print("fdb:", fdb.__version__)

d = FirebirdDialect()
url = d.build_conn_str(
    {"user": "SYSDBA", "password": "secret@pass", "database": "/var/db/example.fdb"},
    "10.0.0.1",
    3050,
)
print("URL:", secrets.redact(url))

parsed = make_url(url)
print("  drivername:", parsed.drivername)
print("  database  :", parsed.database)
print("  query     :", dict(parsed.query))
print("  password parses back correctly:", parsed.password == "secret@pass")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    engine = create_engine(url, connect_args=dict(d.connect_args))
print("  engine dialect:", type(engine.dialect).__module__)
print("  DBAPI driver  :", engine.dialect.dbapi.__name__)

# A Windows path with doubled backslashes - what handing the config from one block
# to the next produces in practice.
win = d.build_conn_str(
    {"user": "u", "password": "p", "database": "D:\\\\erp_data\\\\x.fdb"}, "h", 3050
)
print("Windows path:", secrets.redact(win))

print("\nDONE: firebird+fdb is alive on this image.")
