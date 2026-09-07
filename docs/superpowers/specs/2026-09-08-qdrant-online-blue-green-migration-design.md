# OpenViking Qdrant Online Blue-Green Migration Design

## Status

Proposed design for the follow-up to `volcengine/OpenViking#4458`.

The design is intentionally larger than the original frozen-write migration:
the approved approach is a Qdrant blue-green migration with physical
generations, paired collection aliases, resumable reconciliation, atomic
cutover, and explicit rollback.

## Goal

Migrate an OpenViking Qdrant collection to a new physical generation while
keeping the currently active generation available for reads and writes during
the long copy phase.

The migration must provide:

- a target generation built without mutating the source generation;
- bounded-memory, resumable background copying;
- a safe way to reconcile writes and deletes that happen during copying;
- an atomic switch of both the data and metadata references;
- a retained source generation for rollback and audit;
- explicit recovery states for interrupted or failed operations;
- no new third-party dependency.

The design covers both:

1. **legacy-source migrations** from the pre-`#3872` layout, where the old
   application cannot emit current-format dual writes; and
2. **current-format generation migrations**, where the current adapter can
   dual-write during the copy window.

## Non-goals

- Zero write downtime for the legacy-source path. A short final write barrier
  is required because the removed legacy adapter has no current-format
  dual-write hook and Qdrant has no cross-collection transaction.
- A generic migration framework for non-Qdrant adapters.
- Automatic adoption of an unmarked or ambiguous physical collection.
- Automatic deletion of the source generation.
- A server-side RRF redesign. Hybrid ranking remains the current
  client-side weighted fusion contract.
- A distributed workflow scheduler. One migration controller owns a migration
  at a time; the target marker and migration ID reject later conflicting
  controllers.

## Current-state constraints

The current Qdrant implementation:

- binds `QdrantCollection` directly to a collection name;
- stores its OpenViking marker and sparse dictionary in a sidecar collection;
- writes through `CollectionAdapter.upsert`, `update_data`, and `delete`;
- uses the standard-library `QdrantRestClient`;
- has a frozen-write `qdrant_migrate.py` that copies into a separate target;
- treats `setup_complete=false` as an adapter read gate.

The online design must preserve direct-name operation for deployments that do
not opt into aliases. Existing unmarked collections remain fail-closed.

## Terminology

- **Logical collection**: the OpenViking configuration identity, such as
  `default/context`.
- **Data alias**: the Qdrant alias used by the application for the active
  data generation.
- **Metadata alias**: the Qdrant alias used by the application for the active
  metadata and sparse-dictionary generation.
- **Physical generation**: an immutable-name data collection plus its matching
  metadata sidecar, for example:
  `default__context__gen_20260908_abc` and
  `default__context__gen_20260908_abc__openviking_meta`.
- **Source generation**: the currently authoritative generation before
  cutover.
- **Target generation**: the generation being prepared.
- **Write barrier**: a short operator-controlled pause of source writes used
  for the final exact reconciliation and alias switch.

## Architecture

### 1. Paired aliases

The application uses two aliases:

```text
<logical-data-alias>     -> active physical data generation
<logical-metadata-alias> -> active physical metadata generation
```

Both aliases are switched in one Qdrant
`POST /collections/aliases` request. The request contains delete/create
operations for the two aliases, so data and metadata cannot intentionally
cut over independently.

The aliases are explicit configuration. A deployment that has only the
existing direct physical collection continues to use direct-name mode until
an operator runs the alias bootstrap step.

Alias bootstrap rules:

1. The data and metadata alias names must be pairwise distinct from every
   physical source or target name.
2. An existing alias is inspected and must point to the declared source
   generation.
3. An existing physical collection is never silently adopted as an alias.
   `initialize-alias` is an explicit command and requires a valid current
   marker or an explicit legacy-source declaration.
4. Alias creation is idempotent when the requested alias already points to
   the requested generation.

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
dual_write     current-format source and target receive application writes
cutting_over   final write barrier and alias switch are in progress
active         target is the alias target after cutover
retained       old generation remains available for rollback
rolled_back    alias was switched back to the old generation
failed         operator-visible failure requiring resume or cleanup
```

`building` and `failed` have `setup_complete=false`. `ready`, `dual_write`,
`cutting_over`, `active`, `retained`, and `rolled_back` have
`setup_complete=true`, so a retained generation remains readable and
rollback-safe. A failed or interrupted target remains migration-owned and
resumable; it is not adopted by another migration ID.

### 3. Collection reference handling

`QdrantCollection` accepts a data collection reference and metadata collection
reference independently. In direct mode these are physical names. In alias
mode they are the paired aliases.

All normal data-plane operations use the configured data reference:

- reads and searches use the data alias;
- writes use the data alias and, when configured, the secondary physical
  target;
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
initialize-alias
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
`prepare + backfill + verify` path, but it does not perform an online alias
cutover.

#### `initialize-alias`

Creates or validates the paired aliases without changing the active physical
generation. It refuses unmarked collections unless the operator explicitly
declares the source as legacy.

#### `preflight`

Read-only validation records:

- source and target names;
- alias ownership;
- collection layout and scalar indexes;
- metadata and sparse-map fingerprints;
- source point count and source snapshot fingerprint;
- ACL completeness;
- migration mode (`legacy` or `current`);
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

Scrolls the source in bounded batches and writes the target with
insert-only semantics when the target may already contain newer dual-written
points. The scroll cursor is persisted after every successful batch in the
target marker:

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

For a legacy source, this repeated reconciliation is the online safety
mechanism because the removed legacy adapter cannot dual-write current-format
points. A final write barrier is still mandatory.

For a current-format source, application dual-write keeps new writes flowing
to the target while reconciliation repairs transient target failures.

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
`setup_complete=true`.

#### `cutover`

Cutover requires all of the following:

1. target marker is `ready`;
2. source and target fingerprints match;
3. no concurrent migration owns either alias;
4. current-format dual-write is enabled, or a legacy write barrier is held;
5. a final reconcile/verify succeeds while the barrier is held.

The controller then sends one atomic aliases request that replaces both data
and metadata aliases. The target marker becomes `active`; the previous source
marker becomes `retained`.

For current-format migrations, dual-write remains enabled until post-cutover
verification succeeds. It then becomes disabled explicitly through the normal
configuration rollout.

For legacy migrations, the application is switched to the alias-aware current
adapter after the barrier and alias operation succeed. The old application
must not resume writes to the old physical collection after the switch.

#### `rollback`

Rollback is an atomic paired-alias switch back to the retained source
generation. It is allowed only when:

- the retained source marker is valid;
- the old data and metadata generations still exist;
- the operator names the expected migration ID;
- writes are stopped or the current-format dual-write target is configured
  back toward the old generation.

Rollback never deletes the new generation. The new generation becomes
`retained` and the old generation becomes `active`.

#### `retire`

Deletion is a separate explicit command. It requires the operator to name the
generation and migration ID, verifies that neither alias points to it, and
prints the exact collections before requiring `--confirm`. Normal cutover and
rollback never delete source data. A legacy source without a current marker is
protected by the migration plan and explicit source name rather than being
silently treated as an unowned collection.

## Current-format dual-write

The Qdrant config gains an opt-in online migration section:

```yaml
qdrant:
  url: https://qdrant.example
  online_migration:
    enabled: true
    data_alias: default__context__active
    metadata_alias: default__context__active__openviking_meta
    secondary_collection: default__context__gen_20260908_abc
    secondary_metadata_collection: default__context__gen_20260908_abc__openviking_meta
```

When enabled:

- reads continue through the active aliases;
- upserts and updates write the active alias first, then the secondary
  generation;
- deletes apply to the active alias first, then the secondary generation;
- a secondary failure returns an error to the caller and leaves the source as
  the authority for reconciliation;
- the controller's reconcile phase repairs any source/target divergence before
  cutover.

The order is intentional: source success is the durability authority during
the migration window. Cross-collection writes are not transactional, so a
successful request is not reported when the secondary write fails.

After alias cutover, the active alias points at the new generation. The
secondary is the retained old generation until post-cutover verification and
the explicit configuration rollout disable dual-write.

## Legacy-source mode

The pre-`#3872` collection uses a different metadata marker and ID encoding.
The current adapter must not write current-format points into it.

Legacy mode therefore uses:

- raw REST reads from the old collection;
- the current-format transformer for the target;
- repeated streamed reconciliation;
- a short final source-write barrier;
- paired-alias cutover;
- configuration rollout to the alias-aware current adapter.

The controller never claims that legacy mode provides continuous dual-write.
The maintenance window is part of the runbook and is required for exact
delete handling.

## Failure and recovery

- A source read or target write failure leaves the source alias unchanged.
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
- Alias cutover is one atomic Qdrant request; a rejected request leaves the
  previous alias mapping in place.
- Rollback is available while the previous generation is retained.

Only one controller may operate a given migration ID. A different migration
ID cannot adopt an existing target or marker. The runbook explicitly forbids
two controllers targeting the same alias.

## Configuration and compatibility

Direct-name configuration remains the default and remains compatible with
current deployments. Alias mode is opt-in.

New marker fields are ignored by older adapters as unknown metadata fields,
but an older adapter must not be pointed at a new alias until the normal
deployment gate verifies the alias-aware version.

Old markers without `migrator_version` are not adopted by the migration
controller. They require a fresh preflight and an explicit migration target.
The application read path continues to validate the existing
`_openviking_meta_version` separately.

The existing ACL fail-open contract remains explicit: incomplete ACL records
block `verify` and `cutover` unless the operator supplies the existing
acknowledgement flag. Migration does not backfill ACL fields.

## Testing strategy

### REST and alias tests

- alias create, delete, and paired atomic switch request shapes;
- alias bootstrap refusing unmarked physical collections;
- alias switch failure leaves both aliases unchanged;
- opaque integer and string scroll cursors round-trip;
- timeout propagation through the REST client.

### Migration controller tests

- prepare/backfill/reconcile/verify state transitions;
- bounded batch writes and sparse dictionary chunking;
- cursor resume after an injected failure;
- current-format dual-write upsert, update, and delete;
- secondary write failure returns an error and is repaired by reconcile;
- legacy mode requires the write barrier before cutover;
- target extras are deleted during reconcile;
- completed-target rerun keeps `setup_complete=true`;
- marker version mismatch and stale fingerprints fail closed;
- `KeyboardInterrupt` and `SystemExit` clean pre-marker orphans;
- rollback switches both aliases and retains both generations;
- retire refuses to delete an aliased generation.

### Adapter tests

- direct-name mode remains unchanged;
- alias mode routes data and metadata operations through paired aliases;
- marker physical/logical generation fields round-trip;
- missing `_openviking_original_id` never fabricates a record ID;
- dual-write batching preserves normalized records and IDs.

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
2. Confirm the deployment version supports alias mode and the chosen
   migration state.
3. Run `initialize-alias` if the logical aliases do not exist.
4. Run and review read-only `preflight`.
5. For current-format migration, enable dual-write and verify target writes.
6. Run `prepare`, `backfill`, `reconcile`, and `verify`.
7. For legacy migration, stop writes for the documented short barrier.
8. Run final `reconcile`, `verify`, and `cutover`.
9. Roll out the alias-aware application configuration.
10. Verify active alias, metadata marker, point counts, ACL fields, and a
    representative dense/sparse read.
11. Retain the old generation for the agreed rollback window.
12. Run `retire` only after the rollback window and explicit confirmation.

## Comment coverage

This design closes the remaining PR findings as follows:

- M1: bounded batches, cursors, streamed reconciliation, and scale guidance;
- M2: completed markers remain readable during same-generation reruns;
- M3: marker and plan migrator-version binding;
- M4: interrupt-safe cleanup;
- L5: CLI timeout propagation;
- L6: compact plan JSON;
- L10: migration ownership and one-controller rule;
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
