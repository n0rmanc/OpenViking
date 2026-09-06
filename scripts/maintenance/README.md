# Qdrant maintenance

## Migrate a pre-`#3872` collection

`qdrant_migrate.py` copies one legacy Qdrant collection into a new
current-format collection. The source collection and its metadata sidecar are
never modified or deleted. The target names must be new names unless a
previous migration-owned target is being resumed.

The current Qdrant adapter cannot load pre-`#3872` metadata markers directly;
legacy collections must be migrated (or re-ingested) before configuration is
cut over to the current backend.

Run from the repository root with the project virtual environment:

```bash
./.venv/bin/python scripts/maintenance/qdrant_migrate.py \
  --url https://qdrant.example \
  --source-collection legacy__context \
  --target-collection current__context \
  --source-metadata-collection __openviking_meta \
  --sparse-map /path/to/legacy-sparse-map.json \
  preflight
```

`--source-metadata-collection` defaults to the pre-`#3872` global
`__openviking_meta`. The target sidecar defaults to
`{target_collection}__openviking_meta`; pass
`--target-metadata-collection` only when that name is explicitly reserved for
this migration. Sharing a target metadata sidecar between collections is not
supported.

### Sparse index map

Pre-`#3872` sparse vectors contain numeric indexes without a reliable term
dictionary. Do not guess the mapping. Supply an authoritative JSON object in
either direction:

```json
{"111": "hello", "222": "world"}
```

or:

```json
{"hello": 111, "world": 222}
```

If any source sparse index is absent, preflight fails closed. A source with
multiple named sparse vectors also requires `--sparse-vector-name`.

### Procedure

1. Stop or otherwise freeze all writes to both the source collection and its
   legacy metadata sidecar. Keep both freezes in place for the whole
   preflight, apply, and verification window.
2. Run `preflight` and save its JSON output. Confirm the source/target names,
   exact counts, vector layout, sparse terms, and fingerprints.
3. Review the ACL gate. Records missing or containing malformed
   `acl_enabled`, `acl_direct_grants`, or `acl_inherited_grants` remain
   fail-open after the copy. Grant values must be encoded ACL tokens. Do not
   expose the target until those records are rewritten or an operator
   explicitly accepts the risk with `--allow-acl-fail-open`.
4. Apply only after the plan is reviewed:

   ```bash
   ./.venv/bin/python scripts/maintenance/qdrant_migrate.py \
     --url https://qdrant.example \
     --source-collection legacy__context \
     --target-collection current__context \
     --source-metadata-collection __openviking_meta \
     --sparse-map /path/to/legacy-sparse-map.json \
     apply --plan /path/to/preflight.json --confirm --allow-acl-fail-open
   ```

   Set `QDRANT_API_KEY` in the environment when authentication is required;
   do not put secrets in command-line arguments.
   Remove `--allow-acl-fail-open` when all source records have complete ACL
   fields. `--confirm` is always required for writes.
5. Verify the reported `source_count`, `target_count`, and source/metadata
   fingerprints. Read back the target marker and verify that
   `setup_complete` is `true`; inspect the physical target scalar indexes and
   decode a representative dense and sparse record through the current
   adapter.
6. Change the OpenViking collection configuration to the target collection and
   its target metadata sidecar, then restart/roll out the application through
   the normal deployment process. Configuration cutover is separate from this
   script.
7. Retain the legacy source collection and metadata sidecar for the agreed
   rollback/audit window. Roll back by pointing configuration at the retained
   source; do not delete it as part of this migration.

### Failure and resume behavior

The target marker binds the target to the source collection, source snapshot,
metadata fingerprint, and vector/schema layout. A changed source, stale plan,
unowned target, count mismatch, metadata mismatch, or sparse collision fails
closed. An interrupted migration-owned setup leaves an incomplete marker and
can be resumed after the cause is fixed; an unmarked pre-existing target is
never adopted or overwritten.

This script does not freeze writes, update application configuration, restart
services, or delete legacy data.
