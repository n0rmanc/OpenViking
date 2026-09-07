# OpenViking Qdrant Online Blue-Green Migration Design

## Status

Proposed design for the follow-up to `volcengine/OpenViking#4458`.

The approved approach is a Qdrant blue-green migration with physical
generations, a current-format target, resumable reconciliation, a short final
write barrier, and explicit source retention.

## Goal

Migrate a pre-`#3872` OpenViking Qdrant collection to a new physical
generation that the current-format adapter can read and write. The legacy
deployment remains the source of truth during the long copy phase; it is not
dual-written because the current adapter intentionally cannot read or safely
write the legacy layout.

The migration must provide:

- a target generation built without mutating the source generation;
- bounded-memory, resumable background copying;
- a safe way to reconcile writes and deletes that happen during copying;
- an atomic publication of the target data and metadata aliases;
- a retained legacy source for audit and a bounded rollback window;
- explicit recovery states for interrupted or failed operations;
- no new third-party dependency.

The design covers only **legacy-source migration**. The upstream
`volcengine/OpenViking/main` branch does not contain the current-format Qdrant
adapter or migration code, so current-format-source migration and application
dual-write are out of scope for this follow-up.

## Non-goals

- Zero write downtime. A short final write barrier is required because the
  legacy adapter has no current-format dual-write hook and Qdrant has no
  cross-collection transaction.
- A transparent rollback from a running current-format deployment back to the
  legacy collection. The current adapter cannot consume the legacy marker; once
  target writes are accepted, rollback requires a separate reverse migration.
- A generic migration framework for non-Qdrant adapters.
- Automatic adoption of an unmarked or ambiguous physical collection.
- Automatic deletion of the source generation.
- Migration from a current-format source or any application dual-write mode.
- A server-side RRF redesign. Hybrid ranking remains the current
  client-side weighted fusion contract.
- A distributed workflow scheduler. One migration controller owns a migration
  at a time; the target marker and migration ID reject later conflicting
  controllers.

## Current-state constraints

The current Qdrant implementation in the fork:

- binds `QdrantCollection` directly to a collection name;
- stores its OpenViking marker and sparse dictionary in a sidecar collection;
- writes through `CollectionAdapter.upsert`, `update_data`, and `delete`;
- uses the standard-library `QdrantRestClient`;
- has a frozen-write `qdrant_migrate.py` that copies into a separate target;
- treats `setup_complete=false` as an adapter read gate.

The legacy source has the pre-`#3872` marker and physical-ID encoding. The
current adapter intentionally refuses to load that metadata, so migration
reads the source through raw Qdrant REST and writes only the new
current-format target. Existing direct-name operation remains unchanged until
the operator rolls out the alias-aware current adapter. Existing unmarked
collections remain fail-closed.

The design was verified against `upstream/main` at
`a843ab6bf220b2b3bc82321576d623d1c55c6598`: it contains no
`qdrant_collection.py`, `qdrant_rest.py`, `qdrant_adapter.py`, or
`scripts/maintenance/qdrant_migrate.py`. Those current-format Qdrant files
exist only in the fork/PR work.

## Terminology

- **Logical collection**: the OpenViking configuration identity, such as
  `default/context`.
- **Data alias**: the Qdrant alias published for the current-format target
  data generation.
- **Metadata alias**: the Qdrant alias published for the current-format
  target metadata and sparse-dictionary generation.
- **Physical generation**: an immutable-name data collection plus its matching
  metadata sidecar, for example:
  `default__context__gen_20260908_abc` and
  `default__context__gen_20260908_abc__openviking_meta`.
- **Legacy source**: the pre-`#3872` physical data collection and metadata
  sidecar that remain authoritative until the final write barrier ends.
- **Target generation**: the generation being prepared.
- **Write barrier**: a short operator-controlled pause of source writes used
  for the final exact reconciliation, target verification, alias publication,
  and application rollout.

## Architecture

### 1. Paired target aliases

The current-format deployment uses two aliases after cutover:

```text
<logical-data-alias>     -> active physical data generation
<logical-metadata-alias> -> active physical metadata generation
```

Both aliases are published or replaced in one Qdrant
`POST /collections/aliases` request. The request contains delete/create
operations for the two aliases, so the current-format deployment cannot
intentionally publish data and metadata independently.

The legacy deployment does not use these aliases. It continues to use its
direct physical source names until the operator rolls out the current-format
adapter and configuration after final verification.

Target alias rules:

1. The data and metadata alias names must be pairwise distinct from every
   physical source or target name.
2. An existing alias is inspected and must be absent or owned by the declared
   migration ID and target generation.
3. An existing physical collection is never silently adopted as an alias.
4. Alias publication is idempotent when both aliases already point to the
   requested target generation.

### 2. Physical generations and markers

Every target generation has its own metadata sidecar. The marker payload is
extended with:

```json
{
  "logical_collection": "default/context",
  "data_alias": "default__context__active",
  "metadata_alias": "default__context__active__openviking_meta",
  "physical_collection": "default__context__gen_20260908_abc",
  "physical_metadata_collection": "default__context__gen_20260908_abc__openviking_meta",
  "generation": "20260908_abc",
  "migration_id": "20260908_abc",
  "migration_state": "building",
  "source_collection": "default__context",
  "source_marker_fingerprint": "...",
  "source_fingerprint": "...",
  "metadata_fingerprint": "...",
  "sparse_map_fingerprint": "...",
  "migrator_version": "...",
  "setup_complete": false
}
```

`setup_complete` remains for compatibility with the adapter read gate.
`migration_state` carries the online lifecycle and is not overloaded to mean
both “copy complete” and “currently active”.

Allowed states are:

```text
building       target is being created or copied
ready          target passed exact verification
cutting_over   final write barrier and alias switch are in progress
active         target is the alias target after cutover
retained       target remains available for audit or later cleanup
rolled_back    current-format rollout was reverted before target writes
failed         operator-visible failure requiring resume or cleanup
```

`building` and `failed` have `setup_complete=false`. `ready`,
`cutting_over`, `active`, `retained`, and `rolled_back` have
`setup_complete=true`. A failed or interrupted target remains migration-owned
and resumable; it is not adopted by another migration ID.

### 3. Collection reference handling

`QdrantCollection` accepts a data collection reference and metadata collection
reference independently. The legacy source uses direct physical names. The
current-format deployment uses the paired target aliases.

All normal data-plane operations use the configured data reference:

- reads and searches use the data alias;
- writes use the data alias after the current-format rollout;
- marker and sparse-dictionary reads/writes use the metadata alias or the
  explicit generation sidecar;
- marker validation accepts the logical alias and verifies the physical
  generation fields.

The adapter does not resolve an alias to a physical name for every request.
Qdrant resolves the alias at the data plane; the migration controller uses
physical names when preparing and inspecting generations.

### 4. Migration controller

`scripts/maintenance/qdrant_migrate.py` becomes an explicit phase controller.
Each command is idempotent and records progress in the target marker.

Commands:

```text
preflight
prepare
backfill
reconcile
verify
cutover
rollback
retire
```

`apply` remains accepted as a compatibility alias for the offline
`prepare + backfill + verify` path, but it does not publish aliases or roll out
the current-format application.

#### `preflight`

Read-only validation records:

- source and target names;
- alias ownership;
- collection layout and scalar indexes;
- metadata and sparse-map fingerprints;
- source point count and source snapshot fingerprint;
- ACL completeness;
- migration mode (`legacy`; the only supported mode);
- expected target generation and migration ID.

The plan JSON contains counts, fingerprints, names, and configuration only.
It does not serialize a full `id_map` or target ID set.

#### `prepare`

Creates the target data and metadata collections, writes the incomplete marker,
creates the required indexes, and creates the target sparse dictionary. A
target that already belongs to the same migration ID is resumed. A target
with a different marker, missing marker, or mismatched fingerprints fails
closed.

#### `backfill`

Scrolls the legacy source in bounded batches and transforms each page into the
current-format target. The scroll cursor is persisted after every successful
batch in the target marker:

```json
{
  "last_source_cursor": null
}
```

The cursor is treated as an opaque Qdrant value and may be an integer or
string. A repeated cursor or malformed page fails closed.

The process retains only one source page, one write batch, and bounded
fingerprint state in memory. Sparse dictionary writes are also chunked by the
configured batch size.

#### `reconcile`

Reconciles source and target after backfill:

- upserts source records whose transformed fingerprint differs;
- validates payload security fields and vector shapes;
- removes target records absent from the source;
- verifies sparse dictionary completeness;
- repeats until the source and target fingerprints/counts converge.

This repeated reconciliation is the only online safety mechanism available
while legacy writes continue: the legacy adapter cannot dual-write
current-format points. A final write barrier is still mandatory, and the
controller must report non-convergence rather than claiming readiness when
source churn prevents the fingerprints from matching.

#### `verify`

Performs exact, streamed validation:

- source and target point counts;
- transformed source fingerprint;
- target ID coverage and no extras;
- per-point payload/security fields;
- dense and sparse vector layout;
- scalar indexes;
- metadata and sparse-map fingerprints;
- ACL gate;
- marker state and migration ID.

Successful verification writes `migration_state=ready` and
`setup_complete=true`. Verification before the barrier is a point-in-time
check; `cutover` always repeats it while the barrier is held.

#### `cutover`

Cutover requires all of the following:

1. target marker is `ready`;
2. source and target fingerprints match;
3. no concurrent migration owns either alias;
4. the legacy write barrier is held;
5. a final reconcile/verify succeeds while the barrier is held.

The controller then sends one atomic aliases request that creates or replaces
both target aliases. The target marker becomes `active`. The operator rolls
out the current-format adapter configured with those aliases and only then
releases the barrier. The old application must not resume writes to the legacy
physical collection after the switch.

#### `rollback`

Rollback is a deployment rollback to the legacy adapter and direct source
names, not an alias switch: the current adapter cannot consume the legacy
marker. It is allowed only while the barrier is still held or after the
operator proves that the current-format target has accepted no writes since
cutover. Otherwise the command fails closed and a separate reverse migration
is required. Before that boundary, rollback removes target aliases owned by
this migration in one Qdrant request, restores the legacy deployment, and
records the target as `rolled_back`. Rollback never deletes the target.

#### `retire`

Deletion is a separate explicit command. It requires the operator to name the
generation and migration ID, verifies that neither alias points to it, and
prints the exact collections before requiring `--confirm`. Normal cutover and
rollback never delete source data. A legacy source without a current marker is
protected by the migration plan and explicit source name rather than being
silently treated as an unowned collection.

## Legacy-source flow

The pre-`#3872` collection uses a different metadata marker and ID encoding.
The current adapter must not read or write current-format points into it. The
flow therefore uses:

- raw REST reads from the old collection;
- the current-format transformer for the target;
- repeated streamed reconciliation;
- a short final source-write barrier;
- atomic publication of the paired target aliases;
- configuration rollout to the current adapter using those aliases.

The controller never claims continuous dual-write. The maintenance window is
part of the runbook and is required for exact delete handling. The source is
retained unchanged; it is the rollback source only before the target accepts
new writes.

## Failure and recovery

- A source read or target write failure leaves the legacy application and
  physical source unchanged.
- An interrupted `backfill` resumes from the last persisted cursor.
- A target created before its marker is persisted is deleted only when the
  target is still migration-owned and no marker was written.
- Once a migration marker is persisted, cleanup leaves the target in an
  explicit resumable state instead of deleting it.
- Cleanup catches `BaseException` only to remove pre-marker orphan resources,
  then immediately re-raises the original exception.
- A completed target is never temporarily marked incomplete during a
  same-fingerprint rerun; application reads remain available.
- A version, fingerprint, alias, or count mismatch fails before mutation.
- Alias publication is one atomic Qdrant request; a rejected request leaves
  the previous alias mapping in place.
- Deployment rollback is available only before target writes are accepted;
  after that boundary a reverse migration is required.

Only one controller may operate a given migration ID. A different migration
ID cannot adopt an existing target or marker. The runbook explicitly forbids
two controllers targeting the same alias.

## Configuration and compatibility

Direct-name configuration remains the default for the legacy deployment and
remains compatible with current deployments. The post-cutover current-format
deployment opts into the paired target aliases through the existing Qdrant
collection and metadata-name configuration; there is no secondary collection
or dual-write setting.

New marker fields are ignored by older adapters as unknown metadata fields,
but the legacy adapter must not be pointed at a new alias. The normal
deployment gate verifies the alias-aware current-format version before
releasing the write barrier.

Old markers without `migrator_version` are not adopted by the migration
controller as target markers. The legacy source is accepted only through an
explicit source declaration and a fresh preflight; it is never rewritten.
The application read path continues to validate the existing
`_openviking_meta_version` separately.

The existing ACL fail-open contract remains explicit: incomplete ACL records
block `verify` and `cutover` unless the operator supplies the existing
acknowledgement flag. Migration does not backfill ACL fields.

## Testing strategy

### REST and alias tests

- alias create, delete, and paired atomic switch request shapes;
- target alias ownership and refusal to adopt an unrelated physical collection;
- alias switch failure leaves both aliases unchanged;
- opaque integer and string scroll cursors round-trip;
- timeout propagation through the REST client.

### Migration controller tests

- prepare/backfill/reconcile/verify state transitions;
- bounded batch writes and sparse dictionary chunking;
- cursor resume after an injected failure;
- legacy source transformation preserves logical IDs and current-format
  physical IDs;
- legacy mode requires the write barrier before cutover;
- target extras are deleted during reconcile;
- completed-target rerun keeps `setup_complete=true`;
- marker version mismatch and stale fingerprints fail closed;
- `KeyboardInterrupt` and `SystemExit` clean pre-marker orphans;
- rollback refuses after target writes and restores the legacy deployment
  before that boundary;
- retire refuses to delete an aliased generation.

### Adapter tests

- direct-name mode remains unchanged;
- alias mode routes data and metadata operations through paired aliases;
- marker physical/logical generation fields round-trip;
- missing `_openviking_original_id` never fabricates a record ID;
- current-format rollout smoke tests read and write through the paired aliases.

### Integration and CI

The conditional Qdrant job runs:

```text
tests/maintenance/test_qdrant_migrate.py
tests/storage/test_qdrant_adapter.py
tests/storage/test_qdrant_migration_integration.py
tests/storage/test_qdrant_integration.py
tests/storage/test_collection_schemas.py
```

Fake-REST tests are mandatory in CI. Live Qdrant alias and cutover tests run
only when `QDRANT_URL` is configured and are skipped otherwise. No test
deletes a user collection or performs a live migration against Monster.

## Operational runbook

1. Confirm source and target names, aliases, sparse map, and Qdrant endpoint.
2. Confirm the current-format deployment version supports paired aliases and
   the chosen migration state.
3. Run and review read-only `preflight`.
4. Run `prepare`, `backfill`, `reconcile`, and `verify` while the legacy
   deployment continues serving the source.
5. Stop legacy writes for the documented short barrier.
6. Run final `reconcile`, `verify`, and `cutover`; publish both target aliases
   in one request.
7. Roll out the current-format application configured with the target aliases
   while the barrier is still held, then release the barrier.
8. Verify active aliases, metadata marker, point counts, ACL fields, and a
   representative dense/sparse read.
9. Retain the legacy source for the agreed audit and pre-write rollback
   window.
10. Run `retire` only after the rollback window and explicit confirmation.

## Comment coverage

This design closes the remaining PR findings as follows:

- M1: bounded batches, cursors, streamed reconciliation, and scale guidance;
- M2: completed markers remain readable during same-generation reruns;
- M3: marker and plan migrator-version binding;
- M4: interrupt-safe cleanup;
- L5: CLI timeout propagation;
- L6: compact plan JSON;
- L10: migration ownership and one-controller rule;
- legacy-only scope: no current-format source mode or application dual-write;
- CI gap: focused conditional CI job;
- RRF question: architecture documentation keeps client-side fusion;
- payload fallback: fail-closed record decoding.

The pre-existing random-vector behavior is intentionally not changed in this
migration design; it belongs to a separate adapter-wide contract change.

## References

- Qdrant collection aliases:
  <https://qdrant.tech/documentation/manage-data/collections>
- Qdrant blue-green embedding migration:
  <https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/>
