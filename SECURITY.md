# Security

## Reporting a vulnerability

Email **robert@bisuperhero.cz**. Do not open a public issue.

Please include what an attacker would need (network position, configuration,
credentials they already hold) and what they gain. You will get an
acknowledgement within a week. If a fix is warranted it ships as a patch
release with the CHANGELOG entry naming the versions affected; you will be
credited unless you ask otherwise.

## Supported versions

The latest minor release, plus the v1.0.x line for as long as Mage deployments
run on it. v1.0.x pins its dependencies to the exact versions shipped in the
Mage 0.9.79 image, so a fix in a transitive dependency is not automatically
available there — it is a deliberate decision each time, and that decision
belongs in the release notes. Later releases declare ranges, so such a fix
arrives with an ordinary upgrade.

## Known accepted risks

**`mysql-connector-python` 8.4.0 — CVE-2024-21272 (CVSS 7.5)**, v1.0.x only.
This driver parses everything a MySQL source sends back, so it is genuinely on
the connection path. It is fixed in 9.1.0. v1.0.x pins 8.4.0 because the Mage
0.9.79 image **itself ships 8.4.0** and the package is installed bare into that
image, so the extras never apply there; the real remedy on that line is an
image upgrade, which is outside this package. Later releases require 9.1 or
newer.

Worth knowing in combination: the exploit precondition is a hostile or
impersonated MySQL server. When the connection is tunnelled and
`ssh_host_key_checking` is left at its default `off`, the tunnel endpoint is
not authenticated either — so an attacker on the path can supply that hostile
server. The two defaults compound. Setting `ssh_host_key_checking` closes the
cheap route.

Two further advisories used to be reported, in `paramiko`, and are gone as of
1.0.0 — not by upgrading but by removing the package. The `ssh` extra no longer
installs `paramiko` or `sshtunnel`, because nothing here imports them: the
tunnel is the OpenSSH binary run through `subprocess`. `pip install
dbextractors[ssh]` still works and now pulls nothing; what it needs is an `ssh`
client on `PATH`.

## What this package handles that you should know about

**Database credentials.** They arrive in the configuration dict and are used to
build connection URLs. `src/dbextractors/core/secrets.py` redacts them from every log line and
exception message this package produces; exceptions raised from connection
setup are re-raised with `from None` so the original — which carries the full
URL, password included — is not left attached as `__cause__` for a traceback
dump to print. If you find a path where a credential reaches a log, that is a
security bug and belongs in the email above, not the issue tracker.

Every path was audited before 1.0.0 by triggering it rather than reading it —
wrong passwords, unreachable hosts, unparseable DSNs, rejected session
statements, a tunnel that could not bind — with a distinctive password to grep
for. Four leaks were found and fixed, and the tests in
`tests/core/test_credential_leaks.py` assert on the message, the rendered
traceback and the `__cause__` chain. Where a path is safe only because a driver
happens not to quote the URL today, there is a test pinning that, so a driver
upgrade cannot quietly change it.

**Credentials on the command line are still visible.** `dbx-golden --dsn`, and
several scripts under `scripts/`, take a DSN or a SQLAlchemy URL as an argument.
Anything passed that way shows up in `ps` and in shell history, and no amount of
redaction inside the process changes that. Use the environment variables
(`DBX_TARGET_DSN`, `DBX_GOLDEN_DSN`, `POSTGRES_*`) for anything that is not a
throwaway local database.

**SSH tunnels — host keys are not verified by default.** This is worth stating
plainly rather than burying: `SOURCE_DB.ssh_host_key_checking` defaults to
`off`, which opens the tunnel with `StrictHostKeyChecking=no` and
`UserKnownHostsFile=/dev/null`. The other end is not authenticated, so a
machine-in-the-middle on the path to the bastion can present its own key.

The default is `off` because the configuration contract is frozen and the
implementations this package replaced behaved that way; flipping it would break
working deployments on upgrade. **Set it.** `accept-new` pins the key on first
connection and refuses it if it ever changes; `strict` requires the host to be
in `known_hosts` beforehand and is what you want for anything crossing a
network you do not own.

Tunnel processes are started with `PR_SET_PDEATHSIG` so they die with their
parent instead of surviving as orphans holding a forwarded port open.

**SQL construction.** Identifiers (schema, table, column) come from
configuration, not from user input, and are quoted rather than parameterised —
that is what a database identifier requires. Values are always parameterised.
The `where_clause` key is passed through verbatim by design: it is a fragment
of SQL supplied by whoever writes the pipeline. Treat a configuration dict as
trusted input, the same way you treat the pipeline definition it comes from.

**The golden test tool** creates and drops schemas. It refuses to touch
anything without the `dbx_golden_` prefix, and that guard is tested at each
call site rather than only in isolation. Do not disable it and do not point the
tool at a production database.
