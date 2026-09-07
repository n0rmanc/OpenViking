#!/usr/bin/env python3
"""Migrate a pre-#3872 Qdrant collection into the current OpenViking layout.

The command deliberately implements a blue-green copy: the source collection
and its legacy metadata sidecar are read only, while a separate target
collection receives current-format metadata and records.  It is safe to
re-run after an interrupted copy; points that already belong to the same
logical record are retained rather than overwritten.

Freeze writes to the source collection and legacy metadata sidecar during the
copy, keep them for rollback/audit, and perform any application cutover
separately.  The pre-#3872 global metadata sidecar defaults to
``__openviking_meta``; pass an override when the old deployment used a
different sidecar name.  Missing ``owner_user_id`` values are derived from
user-scoped URIs when possible; ownerless roots remain ownerless.  ACL fields
are copied as-is: missing or malformed legacy ACL fields remain fail-open until
records are rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

# Make ``python scripts/maintenance/qdrant_migrate.py`` work from a checkout
# without requiring an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openviking.core.namespace import owner_fields_for_uri  # noqa: E402
from openviking.storage.acl import DirectAcl  # noqa: E402
from openviking.storage.vectordb.collection.qdrant_rest import (  # noqa: E402
    QdrantError,
    QdrantRestClient,
)
from openviking.storage.vectordb.qdrant_sparse import (  # noqa: E402
    stable_sparse_index,
)
from openviking.storage.vectordb.qdrant_utils import (  # noqa: E402
    qdrant_payload_field_schema,
    to_qdrant_point_id,
)

_META_VERSION = 1
_META_MARKER_ID = to_qdrant_point_id("openviking:metadata")
_META_VECTOR_NAME = "meta"
_ORIGINAL_ID_FIELD = "_openviking_original_id"
_INTERNAL_PAYLOAD_FIELDS = {"uri_depth", "scope_roots"}
_ACL_FIELDS = {"acl_enabled", "acl_direct_grants", "acl_inherited_grants"}
_SECURITY_PAYLOAD_FIELDS = {
    "uri",
    "account_id",
    "owner_user_id",
    *_ACL_FIELDS,
}
_CONTEXT_TYPES = {"memory", "resource", "skill"}
_SPARSE_TERM_MARKER = "_openviking_sparse_term"
_LEGACY_UINT64_MAX = 2**64 - 1
_QDRANT_SPARSE_INDEX_MAX = 0x7FFF_FFFF
_QDRANT_ID_NAMESPACE = uuid.UUID("4b6bb5a8-7f1f-5b1a-9d4c-b93f29b1d67c")
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")


def _legacy_collection_metadata_id(collection_key: str) -> str:
    """Return the deterministic pre-#3872 collection metadata point ID."""

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"openviking:qdrant:collection:{collection_key}",
        )
    )


def _legacy_index_metadata_id(collection_key: str, index_name: str) -> str:
    """Return the deterministic pre-#3872 index metadata point ID."""

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"openviking:qdrant:index:{collection_key}:{index_name}",
        )
    )


class MigrationError(RuntimeError):
    """A migration preflight or apply failure."""


class SparseMigrationError(MigrationError):
    """Sparse vectors cannot be copied without an authoritative term map."""


@dataclass(frozen=True)
class CollectionLayout:
    """The physical vector layout discovered from a Qdrant collection."""

    dense_vector_name: str
    sparse_vector_name: str
    vector_dimension: int
    distance: str
    dense_datatype: str | None
    sparse_enabled: bool
    sparse_modifier: str | None


@dataclass(frozen=True)
class LegacyMetadata:
    """The old QdrantMetaStore collection/index documents."""

    schema: dict[str, Any]
    indexes: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SourceSnapshot:
    """Validated source state captured at one point in time."""

    source_count: int
    id_map: dict[str, str]
    sparse_terms: set[str]
    fingerprint: str
    acl_incomplete_count: int
    points: tuple[dict[str, Any], ...] = ()


@dataclass
class MigrationPlan:
    """Read-only migration plan returned by :meth:`QdrantMigration.preflight`."""

    source_collection: str
    target_collection: str
    source_metadata_collection: str
    target_metadata_collection: str
    source_count: int
    target_count: int
    target_exists: bool
    dense_vector_name: str
    sparse_vector_name: str
    vector_dimension: int
    distance: str
    dense_datatype: str | None
    sparse_enabled: bool
    sparse_modifier: str | None
    sparse_weight: float
    sparse_terms: set[str] = field(default_factory=set)
    id_map: dict[str, str] = field(default_factory=dict)
    existing_target_ids: set[str] = field(default_factory=set)
    source_fingerprint: str = ""
    metadata_fingerprint: str = ""
    sparse_map_fingerprint: str = ""
    acl_incomplete_count: int = 0
    target_metadata_exists: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe plan data suitable for CLI output."""

        value = asdict(self)
        value["sparse_terms"] = sorted(self.sparse_terms)
        value["existing_target_ids"] = sorted(self.existing_target_ids)
        return value


@dataclass(frozen=True)
class MigrationResult:
    """Summary of a completed (or resumed) apply."""

    source_count: int
    migrated_count: int
    skipped_count: int
    target_count: int
    target_collection: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(response: Mapping[str, Any]) -> Any:
    value = response.get("result", response)
    return value


def _status(exc: BaseException) -> int | None:
    return getattr(exc, "status", None)


def _canonical_distance(value: Any) -> str:
    normalized = str(value or "cosine").strip().lower()
    mapping = {
        "cosine": "Cosine",
        "ip": "Dot",
        "dot": "Dot",
        "l2": "Euclid",
        "euclid": "Euclid",
    }
    if normalized not in mapping:
        raise MigrationError(f"unsupported Qdrant distance metric: {value!r}")
    return mapping[normalized]


def _optional_layout_value(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"source {field_name} is invalid")
    return value.strip().lower()


def _path_depth(path: str) -> int:
    return len([part for part in path.split("/") if part])


def _scope_roots(path: str) -> list[str]:
    parts = [part for part in path.split("/") if part]
    return ["/", *("/" + "/".join(parts[:index]) for index in range(1, len(parts) + 1))]


def _normalize_uri(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{field_name} must be a non-empty OpenViking URI")
    stripped = value.strip()
    if stripped.startswith("viking://"):
        stripped = stripped[len("viking://") :]
    elif not stripped.startswith("/"):
        raise MigrationError(f"{field_name} is not a canonical OpenViking URI: {value!r}")
    normalized = "/" + stripped.lstrip("/")
    return normalized.rstrip("/") or "/"


def _finite_vector(value: Any, *, field_name: str, dimension: int | None = None) -> list[float]:
    if not isinstance(value, list):
        raise MigrationError(f"{field_name} must be a dense vector list")
    if dimension is not None and len(value) != dimension:
        raise MigrationError(
            f"{field_name} dimension mismatch: expected {dimension}, got {len(value)}"
        )
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"{field_name} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in vector):
        raise MigrationError(f"{field_name} contains a non-finite value")
    return vector


def _point_fingerprint(
    *,
    source_point_id: Any,
    transformed: Mapping[str, Any],
) -> str:
    try:
        encoded = json.dumps(
            {
                "source_point_id": [type(source_point_id).__name__, str(source_point_id)],
                "point": transformed,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MigrationError(
            f"source point {source_point_id!r} is not JSON-serializable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_fingerprint(point_fingerprints: Iterable[str]) -> str:
    encoded = "\n".join(sorted(point_fingerprints)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_fingerprint(metadata: LegacyMetadata) -> str:
    try:
        encoded = json.dumps(
            {"schema": metadata.schema, "indexes": metadata.indexes},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MigrationError("legacy metadata is not JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sparse_map_fingerprint(sparse_map: Mapping[int, str]) -> str:
    """Fingerprint the authoritative sparse map used for this plan."""

    try:
        encoded = json.dumps(
            {str(index): term for index, term in sorted(sparse_map.items())},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MigrationError("sparse map is not JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _legacy_qdrant_point_id(value: Any) -> Any:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _LEGACY_UINT64_MAX
    ):
        return value
    value_string = str(value)
    try:
        return str(uuid.UUID(value_string))
    except (AttributeError, TypeError, ValueError):
        return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, value_string))


def _acl_complete(payload: Mapping[str, Any]) -> bool:
    if not all(field in payload for field in _ACL_FIELDS):
        return False
    if not isinstance(payload["acl_enabled"], bool):
        return False
    if not isinstance(payload["acl_direct_grants"], list) or not isinstance(
        payload["acl_inherited_grants"], list
    ):
        return False
    if not payload["acl_enabled"] and (
        payload["acl_direct_grants"] or payload["acl_inherited_grants"]
    ):
        return False
    try:
        DirectAcl.from_context_fields(payload, "acl_direct")
        DirectAcl.from_context_fields(payload, "acl_inherited")
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def _normalize_owner_user_id(
    payload: dict[str, Any],
    *,
    uri: str,
    point_id: Any,
    source_keys: set[str],
) -> None:
    expected_owner = owner_fields_for_uri(
        f"viking://{uri.lstrip('/')}"
    ).get("owner_user_id")
    actual_owner = payload.get("owner_user_id")
    if actual_owner is None:
        if expected_owner is None:
            payload.pop("owner_user_id", None)
            source_keys.discard("owner_user_id")
        else:
            payload["owner_user_id"] = expected_owner
        return
    if not isinstance(actual_owner, str) or not actual_owner.strip():
        raise MigrationError(f"point {point_id!r} has an invalid owner_user_id")
    if expected_owner is not None and actual_owner != expected_owner:
        raise MigrationError(
            f"point {point_id!r} owner_user_id does not match uri {uri!r}"
        )


def _assert_security_payload(
    payload: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    *,
    point_id: str,
) -> None:
    for field_name in _SECURITY_PAYLOAD_FIELDS:
        expected_present = field_name in expected_payload
        actual_present = field_name in payload
        if expected_present != actual_present or (
            expected_present and payload[field_name] != expected_payload[field_name]
        ):
            label = "ACL field" if field_name in _ACL_FIELDS else "security field"
            raise MigrationError(
                f"target point {point_id!r} {label} {field_name!r} differs"
            )


def _sparse_index(
    value: Any,
    *,
    field_name: str,
    allow_numeric_string: bool = False,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool):
        raise SparseMigrationError(f"{field_name} must be an integer")
    if isinstance(value, int):
        index = value
    elif allow_numeric_string and isinstance(value, str) and _INTEGER_RE.fullmatch(value.strip()):
        index = int(value)
    else:
        raise SparseMigrationError(f"{field_name} must be an integer")
    if not minimum <= index <= _QDRANT_SPARSE_INDEX_MAX:
        raise SparseMigrationError(
            f"{field_name} must be between {minimum} and {_QDRANT_SPARSE_INDEX_MAX}"
        )
    return index


def _sparse_term(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SparseMigrationError(f"{field_name} must be a non-empty string")
    return value


def _legacy_sparse_map(value: Mapping[Any, Any] | None) -> dict[int, str]:
    """Normalize either ``old-index -> term`` or ``term -> old-index`` JSON."""

    if value is None:
        return {}
    normalized: dict[int, str] = {}
    terms_to_indexes: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        # A JSON object turns integer keys into strings.  Prefer the reverse
        # form when the value is an integer so {"123": 111} remains a
        # numeric-looking term instead of being parsed as old-index -> term.
        if isinstance(raw_key, int) and not isinstance(raw_key, bool):
            raw_index, raw_term = raw_key, raw_value
        elif isinstance(raw_value, int) and not isinstance(raw_value, bool):
            raw_index, raw_term = raw_value, raw_key
        elif isinstance(raw_key, str) and _INTEGER_RE.fullmatch(raw_key.strip()):
            raw_index, raw_term = raw_key, raw_value
        else:
            raise SparseMigrationError(
                "sparse map entries must be {old_index: term} or {term: old_index}"
            )
        old_index = _sparse_index(
            raw_index,
            field_name="sparse map index",
            allow_numeric_string=isinstance(raw_index, str),
        )
        term = _sparse_term(raw_term, field_name="sparse map term")
        if old_index in normalized and normalized[old_index] != term:
            raise SparseMigrationError(
                f"sparse map contains conflicting terms for old index {old_index}"
            )
        previous_index = terms_to_indexes.get(term)
        if previous_index is not None and previous_index != old_index:
            raise SparseMigrationError(
                f"sparse map maps term {term!r} to multiple old indexes"
            )
        normalized[old_index] = term
        terms_to_indexes[term] = old_index
    return normalized


def _schema_fields(schema: Mapping[str, Any], *, label: str) -> dict[str, Mapping[str, Any]]:
    fields = schema.get("Fields", [])
    if fields is None:
        return {}
    if not isinstance(fields, list):
        raise MigrationError(f"{label} schema Fields must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for field_item in fields:
        if not isinstance(field_item, Mapping) or not field_item.get("FieldName"):
            raise MigrationError(f"{label} schema has a malformed field")
        name = str(field_item["FieldName"])
        if name in result:
            raise MigrationError(f"{label} schema has duplicate field {name!r}")
        result[name] = field_item
    return result


def _validate_legacy_schema(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping):
        raise MigrationError("legacy collection metadata schema must be an object")
    _schema_fields(schema, label="legacy")
    collection_name = schema.get("CollectionName")
    if collection_name is not None and (
        not isinstance(collection_name, str) or not collection_name.strip()
    ):
        raise MigrationError("legacy collection metadata has an invalid CollectionName")
    scalar_index = schema.get("ScalarIndex")
    if scalar_index is not None and not isinstance(scalar_index, (list, tuple, set)):
        raise MigrationError("legacy collection metadata ScalarIndex must be a list")
    if isinstance(scalar_index, (list, tuple, set)) and any(
        not isinstance(field, str) or not field.strip() for field in scalar_index
    ):
        raise MigrationError("legacy collection metadata ScalarIndex has an invalid field")


def _validate_scalar_index(value: Any, *, label: str) -> None:
    """Validate the ScalarIndex shapes understood by the adapter."""

    if value is None:
        return
    if isinstance(value, Mapping):
        fields = value.keys()
    elif isinstance(value, (list, tuple, set)):
        fields = value
    else:
        raise MigrationError(f"{label} ScalarIndex is malformed")
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise MigrationError(f"{label} ScalarIndex has an invalid field")


def _validate_schema_subset(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    """Require source fields/indexes to remain compatible with a newer target."""

    source_collection = source.get("CollectionName")
    if source_collection is not None and target.get("CollectionName") != source_collection:
        raise MigrationError("target current marker schema changed CollectionName")

    source_fields = _schema_fields(source, label="source")
    target_fields = _schema_fields(target, label="target")
    for name, source_field in source_fields.items():
        target_field = target_fields.get(name)
        if target_field is None:
            raise MigrationError(f"target current marker schema is missing field {name!r}")
        source_type = str(source_field.get("FieldType") or "").strip().lower()
        target_type = str(target_field.get("FieldType") or "").strip().lower()
        if source_type != target_type:
            raise MigrationError(
                f"target current marker schema changed field type for {name!r}: "
                f"source={source_type!r} target={target_type!r}"
            )
        for key in ("Dim", "IsPrimaryKey"):
            if key in source_field and target_field.get(key) != source_field[key]:
                raise MigrationError(
                    f"target current marker schema changed field {name!r} {key}"
                )

    source_scalar = source.get("ScalarIndex")
    target_scalar = target.get("ScalarIndex")
    if isinstance(source_scalar, (list, tuple, set)):
        if not isinstance(target_scalar, (list, tuple, set)):
            raise MigrationError("target current marker schema has no ScalarIndex list")
        missing = set(map(str, source_scalar)) - set(map(str, target_scalar))
        if missing:
            raise MigrationError(
                "target current marker schema is missing scalar indexes: "
                f"{sorted(missing)!r}"
            )


class QdrantMigration:
    """Copy a legacy Qdrant collection into the current OpenViking format."""

    def __init__(
        self,
        *,
        client: Any,
        source_collection: str,
        target_collection: str,
        source_metadata_collection: str | None = None,
        target_metadata_collection: str | None = None,
        batch_size: int = 100,
        dense_vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        sparse_map: Mapping[Any, Any] | None = None,
    ) -> None:
        self._client = client
        self.source_collection = self._name(source_collection, "source collection")
        self.target_collection = self._name(target_collection, "target collection")
        if self.source_collection == self.target_collection:
            raise ValueError("source and target collections must differ")
        # pre-#3872 QdrantMetaStore used one global sidecar by default.
        self.source_metadata_collection = source_metadata_collection or "__openviking_meta"
        self.target_metadata_collection = target_metadata_collection or (
            f"{self.target_collection}__openviking_meta"
        )
        self.source_metadata_collection = self._name(
            self.source_metadata_collection,
            "source metadata collection",
        )
        self.target_metadata_collection = self._name(
            self.target_metadata_collection,
            "target metadata collection",
        )
        collection_names = {
            "source collection": self.source_collection,
            "target collection": self.target_collection,
            "source metadata collection": self.source_metadata_collection,
            "target metadata collection": self.target_metadata_collection,
        }
        seen_names: dict[str, str] = {}
        for description, name in collection_names.items():
            previous = seen_names.get(name)
            if previous is not None:
                raise ValueError(
                    f"{description} must differ from {previous}; "
                    "all migration collections must be pairwise distinct"
                )
            seen_names[name] = description
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._dense_vector_name_override = dense_vector_name
        self._sparse_vector_name_override = sparse_vector_name
        self._sparse_map = _legacy_sparse_map(sparse_map)

    @staticmethod
    def _name(value: str, description: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{description} must not be empty")
        return normalized

    def _path(self, collection: str, suffix: str = "") -> str:
        return f"/collections/{quote(collection, safe='')}{suffix}"

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(method, path, body, params=params)
        if not isinstance(response, dict):
            raise MigrationError(f"Qdrant returned a non-object response for {method} {path}")
        return response

    def _exists(self, collection: str) -> bool:
        try:
            self._request("GET", self._path(collection))
        except Exception as exc:
            if _status(exc) == 404:
                return False
            raise
        return True

    def _collection_info(self, collection: str) -> dict[str, Any]:
        try:
            response = self._request("GET", self._path(collection))
        except Exception as exc:
            if _status(exc) == 404:
                raise MigrationError(f"Qdrant collection does not exist: {collection}") from exc
            raise
        value = _result(response)
        if not isinstance(value, dict):
            raise MigrationError(f"invalid Qdrant collection response for {collection}")
        return value

    @staticmethod
    def _params(info: Mapping[str, Any]) -> Mapping[str, Any]:
        config = info.get("config")
        if isinstance(config, Mapping):
            params = config.get("params")
            if isinstance(params, Mapping):
                return params
        params = info.get("params")
        if isinstance(params, Mapping):
            return params
        return info

    def _layout(
        self,
        info: Mapping[str, Any],
        *,
        honor_overrides: bool = True,
    ) -> CollectionLayout:
        params = self._params(info)
        vectors = params.get("vectors")
        if not isinstance(vectors, Mapping):
            raise MigrationError("Qdrant collection has no dense vector configuration")

        dense_name: str | None = None
        dense_config: Mapping[str, Any] | None = None
        dense_override = self._dense_vector_name_override if honor_overrides else None
        if "size" in vectors:
            if dense_override:
                raise MigrationError(
                    "cannot rename an unnamed source dense vector; "
                    "remove --dense-vector-name"
                )
            dense_name = dense_override or "vector"
            dense_config = vectors
        else:
            named = [
                (str(name), value)
                for name, value in vectors.items()
                if isinstance(value, Mapping) and "size" in value
            ]
            if dense_override:
                selected = next(
                    ((name, value) for name, value in named if name == dense_override),
                    None,
                )
                if selected is None:
                    raise MigrationError(
                        f"configured dense vector {dense_override!r} "
                        "is absent from the source collection"
                    )
                dense_name, dense_config = selected
            elif len(named) == 1:
                dense_name, dense_config = named[0]
            else:
                raise MigrationError(
                    "source collection has multiple named dense vectors; "
                    "pass --dense-vector-name to select one"
                )

        assert dense_name is not None and dense_config is not None
        raw_dimension = dense_config.get("size")
        if type(raw_dimension) is not int or raw_dimension <= 0:
            raise MigrationError("source dense vector size must be a positive integer")
        dimension = raw_dimension

        sparse_vectors = params.get("sparse_vectors")
        if sparse_vectors is None:
            sparse_vectors = info.get("sparse_vectors")
        sparse_enabled = isinstance(sparse_vectors, Mapping) and bool(sparse_vectors)
        sparse_name: str | None = None
        sparse_override = self._sparse_vector_name_override if honor_overrides else None
        if isinstance(sparse_vectors, Mapping):
            names = [str(name) for name in sparse_vectors]
            if sparse_override:
                if sparse_override not in names:
                    raise MigrationError(
                        f"configured sparse vector {sparse_override!r} "
                        "is absent from the source collection"
                    )
                sparse_name = sparse_override
            elif len(names) == 1:
                sparse_name = names[0]
            elif names:
                raise SparseMigrationError(
                    "source collection has multiple named sparse vectors; "
                    "pass --sparse-vector-name to select one"
                )
        sparse_name = sparse_name or sparse_override or "sparse_vector"

        distance = _canonical_distance(dense_config.get("distance"))
        dense_datatype = _optional_layout_value(
            dense_config.get("datatype"),
            field_name="dense vector datatype",
        )
        sparse_modifier: str | None = None
        if isinstance(sparse_vectors, Mapping) and sparse_name in sparse_vectors:
            sparse_config = sparse_vectors[sparse_name]
            if sparse_config is not None and not isinstance(sparse_config, Mapping):
                raise MigrationError("source sparse vector configuration is invalid")
            if isinstance(sparse_config, Mapping):
                sparse_modifier = _optional_layout_value(
                    sparse_config.get("modifier"),
                    field_name="sparse vector modifier",
                )
        return CollectionLayout(
            dense_vector_name=dense_name,
            sparse_vector_name=sparse_name,
            vector_dimension=dimension,
            distance=distance,
            dense_datatype=dense_datatype,
            sparse_enabled=sparse_enabled,
            sparse_modifier=sparse_modifier,
        )

    def _count(
        self,
        collection: str,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> int:
        response = self._request(
            "POST",
            self._path(collection, "/points/count"),
            {"exact": True, "filter": dict(filter or {})},
        )
        value = _result(response)
        if not isinstance(value, Mapping):
            raise MigrationError(f"invalid Qdrant count response for {collection}")
        count = value.get("count")
        if type(count) is not int or count < 0:
            raise MigrationError(f"invalid Qdrant count for {collection}")
        return count

    def _scroll(
        self,
        collection: str,
        *,
        with_vectors: bool,
        filter: Mapping[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        offset: Any = None
        seen_offsets: set[str] = set()
        while True:
            body: dict[str, Any] = {
                "limit": self.batch_size,
                "with_payload": True,
                "with_vector": with_vectors,
            }
            if filter:
                body["filter"] = dict(filter)
            if offset is not None:
                body["offset"] = offset
            response = self._request(
                "POST",
                self._path(collection, "/points/scroll"),
                body,
            )
            value = _result(response)
            if not isinstance(value, Mapping):
                raise MigrationError(f"invalid Qdrant scroll response for {collection}")
            page = value.get("points")
            if not isinstance(page, list):
                raise MigrationError(f"invalid Qdrant scroll points for {collection}")
            for point in page:
                if isinstance(point, dict):
                    yield point
            if not page:
                return
            next_offset = value.get("next_page_offset")
            if next_offset is None:
                return
            key = repr(next_offset)
            if key in seen_offsets:
                raise MigrationError(f"Qdrant scroll repeated its page offset for {collection}")
            seen_offsets.add(key)
            offset = next_offset

    def _retrieve(
        self,
        collection: str,
        point_ids: list[str],
        *,
        with_vectors: bool,
    ) -> list[dict[str, Any]]:
        if not point_ids:
            return []
        response = self._request(
            "POST",
            self._path(collection, "/points"),
            {
                "ids": point_ids,
                "with_payload": True,
                "with_vector": with_vectors,
            },
        )
        value = _result(response)
        if not isinstance(value, list):
            raise MigrationError(f"invalid Qdrant point lookup response for {collection}")
        return [point for point in value if isinstance(point, dict)]

    def _legacy_metadata(self) -> LegacyMetadata:
        if not self._exists(self.source_metadata_collection):
            raise MigrationError(
                f"legacy metadata collection does not exist: {self.source_metadata_collection}"
            )
        metadata_filter = {
            "must": [
                {
                    "key": "collection_key",
                    "match": {"value": self.source_collection},
                }
            ]
        }
        collection_docs: list[dict[str, Any]] = []
        index_docs: list[dict[str, Any]] = []
        metadata_scanned = 0
        metadata_count = self._count(
            self.source_metadata_collection,
            filter=metadata_filter,
        )
        for point in self._scroll(
            self.source_metadata_collection,
            with_vectors=False,
            filter=metadata_filter,
        ):
            metadata_scanned += 1
            point_id = point.get("id")
            payload = point.get("payload")
            if not isinstance(payload, Mapping):
                continue
            kind = payload.get("kind")
            if kind == "collection":
                expected_point_id = _legacy_collection_metadata_id(self.source_collection)
                if point_id is None or str(point_id) != expected_point_id:
                    raise MigrationError(
                        "legacy collection metadata point-id does not match "
                        f"deterministic encoding: expected={expected_point_id!r} "
                        f"found={point_id!r}"
                    )
                collection_docs.append(dict(payload))
            elif kind == "index":
                index_name = payload.get("index_name")
                if not isinstance(index_name, str) or not index_name.strip():
                    raise MigrationError("legacy index metadata has an empty index name")
                expected_point_id = _legacy_index_metadata_id(
                    self.source_collection,
                    index_name,
                )
                if point_id is None or str(point_id) != expected_point_id:
                    raise MigrationError(
                        f"legacy index metadata point-id does not match deterministic "
                        f"encoding for {index_name!r}: expected={expected_point_id!r} "
                        f"found={point_id!r}"
                    )
                index_docs.append(dict(payload))
            else:
                raise MigrationError(
                    "legacy metadata has an unknown kind for "
                    f"{self.source_collection!r}: {kind!r}"
                )
        if metadata_scanned != metadata_count:
            raise MigrationError(
                "legacy metadata count mismatch: "
                f"exact count={metadata_count} paginated count={metadata_scanned}"
            )
        if len(collection_docs) != 1:
            raise MigrationError(
                f"expected one legacy collection metadata document for {self.source_collection}, "
                f"found {len(collection_docs)}"
            )
        collection_doc = collection_docs[0]
        schema = collection_doc.get("meta")
        if not isinstance(schema, dict):
            raise MigrationError("legacy collection metadata has no schema object")
        _validate_legacy_schema(schema)
        indexes: dict[str, dict[str, Any]] = {}
        for document in index_docs:
            name = document.get("index_name")
            meta = document.get("meta")
            if not isinstance(name, str) or not isinstance(meta, dict):
                raise MigrationError("legacy index metadata is malformed")
            if not name.strip():
                raise MigrationError("legacy index metadata has an empty index name")
            _validate_scalar_index(
                meta.get("ScalarIndex"),
                label=f"legacy index metadata for {name!r}",
            )
            if isinstance(meta.get("VectorIndex"), (list, tuple, set)) or (
                "VectorIndex" in meta and not isinstance(meta.get("VectorIndex"), Mapping)
            ):
                raise MigrationError(
                    f"legacy index metadata VectorIndex is malformed for {name!r}"
                )
            if name in indexes:
                raise MigrationError(f"duplicate legacy index metadata: {name}")
            indexes[name] = dict(meta)
        if not indexes:
            raise MigrationError(
                f"legacy metadata has no index documents for {self.source_collection}"
            )
        return LegacyMetadata(
            schema=dict(schema),
            indexes=indexes,
        )

    @staticmethod
    def _payload(point: Mapping[str, Any]) -> dict[str, Any]:
        value = point.get("payload")
        if not isinstance(value, Mapping):
            raise MigrationError(f"Qdrant point {point.get('id')!r} has no payload")
        return dict(value)

    def _source_vectors(
        self,
        point: Mapping[str, Any],
        layout: CollectionLayout,
    ) -> tuple[list[float] | None, dict[str, list[Any]] | None]:
        vectors = point.get("vector")
        if vectors is None:
            vectors = point.get("vectors")
        if isinstance(vectors, list):
            dense_value: Any = vectors
            sparse_value = None
        elif isinstance(vectors, Mapping):
            dense_value = vectors.get(layout.dense_vector_name)
            if dense_value is None and layout.dense_vector_name == "vector":
                # Qdrant represents an unnamed dense vector alongside named
                # sparse vectors as {"": [...], "sparse_name": {...}}.
                dense_value = vectors.get("")
            sparse_value = vectors.get(layout.sparse_vector_name)
            if sparse_value is None:
                candidates = [
                    value
                    for name, value in vectors.items()
                    if name != layout.dense_vector_name
                    and isinstance(value, Mapping)
                    and "indices" in value
                    and "values" in value
                ]
                if self._sparse_vector_name_override:
                    if candidates:
                        raise SparseMigrationError(
                            f"point {point.get('id')!r} is missing configured sparse vector "
                            f"{layout.sparse_vector_name!r}"
                        )
                elif len(candidates) == 1:
                    sparse_value = candidates[0]
                elif len(candidates) > 1:
                    raise SparseMigrationError(
                        f"point {point.get('id')!r} has multiple sparse vectors"
                    )
        else:
            raise MigrationError(f"Qdrant point {point.get('id')!r} has no vector payload")
        dense = (
            _finite_vector(
                dense_value,
                field_name=f"point {point.get('id')!r} dense vector",
                dimension=layout.vector_dimension,
            )
            if dense_value is not None
            else None
        )
        if sparse_value is None:
            if dense is None:
                raise MigrationError(
                    f"point {point.get('id')!r} has neither a dense nor sparse vector"
                )
            return dense, None
        if not isinstance(sparse_value, Mapping):
            raise SparseMigrationError(f"point {point.get('id')!r} sparse vector is malformed")
        indices = sparse_value.get("indices")
        values = sparse_value.get("values")
        if (
            not isinstance(indices, list)
            or not isinstance(values, list)
            or len(indices) != len(values)
        ):
            raise SparseMigrationError(
                f"point {point.get('id')!r} sparse indices/values are not parallel lists"
            )
        if not indices:
            raise SparseMigrationError(
                f"point {point.get('id')!r} sparse vector has no entries"
            )
        return dense, {"indices": list(indices), "values": list(values)}

    def _transform_point(
        self,
        point: Mapping[str, Any],
        *,
        layout: CollectionLayout,
        schema: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], set[str]]:
        payload = self._payload(point)
        original_id = payload.get(_ORIGINAL_ID_FIELD)
        if original_id is None or str(original_id) == "":
            raise MigrationError(
                f"point {point.get('id')!r} is missing {_ORIGINAL_ID_FIELD} (original id)"
            )
        logical_id = str(original_id)
        source_keys = set(payload)
        payload[_ORIGINAL_ID_FIELD] = logical_id

        if "uri" not in payload:
            raise MigrationError(f"point {point.get('id')!r} is missing uri")
        payload["uri"] = _normalize_uri(payload["uri"], field_name="uri")
        if "parent_uri" in payload and payload["parent_uri"] is not None:
            payload["parent_uri"] = _normalize_uri(
                payload["parent_uri"],
                field_name="parent_uri",
            )
        _normalize_owner_user_id(
            payload,
            uri=payload["uri"],
            point_id=point.get("id"),
            source_keys=source_keys,
        )
        payload["uri_depth"] = _path_depth(payload["uri"])
        payload["scope_roots"] = _scope_roots(payload["uri"])

        field_names = {
            str(field.get("FieldName"))
            for field in schema.get("Fields", [])
            if isinstance(field, Mapping) and field.get("FieldName")
        }
        if "level" in payload or "level" in field_names:
            level = payload.get("level")
            if isinstance(level, bool) or not isinstance(level, (int, float)):
                raise MigrationError(f"point {point.get('id')!r} has no valid level")
            if isinstance(level, float) and (
                not math.isfinite(level) or not level.is_integer()
            ):
                raise MigrationError(f"point {point.get('id')!r} has no valid level")
            level_int = int(level)
            if level_int not in (0, 1, 2):
                raise MigrationError(f"point {point.get('id')!r} has unsupported level {level!r}")
            payload["level"] = level_int
        if "context_type" in payload or "context_type" in field_names:
            context_type = payload.get("context_type")
            if not isinstance(context_type, str) or context_type not in _CONTEXT_TYPES:
                raise MigrationError(
                    f"point {point.get('id')!r} has invalid context_type {context_type!r}"
                )
        for identity_field in ("account_id",):
            if identity_field in payload or identity_field in field_names:
                value = payload.get(identity_field)
                if not isinstance(value, str) or not value.strip():
                    raise MigrationError(
                        f"point {point.get('id')!r} is missing {identity_field}"
                    )

        covered = source_keys | _INTERNAL_PAYLOAD_FIELDS
        if not covered.issubset(payload):
            missing = sorted(covered - set(payload))
            raise MigrationError(
                f"point {point.get('id')!r} lost payload fields during migration: {missing}"
            )

        dense, sparse = self._source_vectors(point, layout)
        vectors: dict[str, Any] = {}
        if dense is not None:
            vectors[layout.dense_vector_name] = dense
        terms: set[str] = set()
        if sparse is not None:
            if not layout.sparse_enabled:
                raise SparseMigrationError(
                    "source points contain sparse vectors but the collection has no "
                    "sparse-vector configuration"
                )
            if not self._sparse_map:
                raise SparseMigrationError(
                    "source sparse vectors require an authoritative old-index-to-term mapping"
                )
            new_indices: list[int] = []
            new_values: list[float] = []
            by_index: dict[int, float] = {}
            target_terms: dict[int, str] = {}
            for raw_index, raw_value in zip(sparse["indices"], sparse["values"], strict=True):
                old_index = _sparse_index(
                    raw_index,
                    field_name=f"point {point.get('id')!r} sparse index",
                )
                if isinstance(raw_value, bool):
                    raise SparseMigrationError(
                        f"point {point.get('id')!r} contains an invalid sparse entry"
                    )
                try:
                    weight = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise SparseMigrationError(
                        f"point {point.get('id')!r} contains an invalid sparse entry"
                    ) from exc
                if not math.isfinite(weight):
                    raise SparseMigrationError(
                        f"point {point.get('id')!r} contains a non-finite sparse weight"
                    )
                term = self._sparse_map.get(old_index)
                if term is None:
                    raise SparseMigrationError(
                        "authoritative sparse mapping is missing old index "
                        f"{old_index} for point {point.get('id')!r}"
                    )
                new_index = stable_sparse_index(term)
                previous_term = target_terms.get(new_index)
                if previous_term is not None and previous_term != term:
                    raise SparseMigrationError(
                        "sparse term collision after migration: "
                        f"index={new_index} terms={previous_term!r},{term!r}"
                    )
                target_terms[new_index] = term
                by_index[new_index] = by_index.get(new_index, 0.0) + weight
                terms.add(term)
            new_indices = sorted(by_index)
            new_values = [by_index[index] for index in new_indices]
            vectors[layout.sparse_vector_name] = {
                "indices": new_indices,
                "values": new_values,
            }
        target_point = {
            "id": to_qdrant_point_id(logical_id),
            "vector": vectors,
            "payload": payload,
        }
        return logical_id, target_point, terms

    def _scan_source(
        self,
        *,
        layout: CollectionLayout,
        schema: Mapping[str, Any],
        capture_points: bool = False,
    ) -> SourceSnapshot:
        source_count = self._count(self.source_collection)
        id_map: dict[str, str] = {}
        source_identities: dict[str, tuple[str, str]] = {}
        source_point_ids: set[tuple[str, str]] = set()
        source_logical_ids: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
        sparse_terms: set[str] = set()
        point_fingerprints: list[str] = []
        acl_incomplete_count = 0
        transformed_points: list[dict[str, Any]] = []
        source_scanned = 0
        for point in self._scroll(self.source_collection, with_vectors=True):
            source_scanned += 1
            raw_point_id = point.get("id")
            if raw_point_id is None:
                raise MigrationError("source collection contains a point without an id")
            source_point_key = (type(raw_point_id).__name__, str(raw_point_id))
            if source_point_key in source_point_ids:
                raise MigrationError(
                    f"source pagination returned duplicate point id {raw_point_id!r}"
                )
            source_point_ids.add(source_point_key)
            raw_payload = self._payload(point)
            raw_original_id = raw_payload.get(_ORIGINAL_ID_FIELD)
            raw_identity = (
                type(raw_original_id).__name__,
                str(raw_original_id),
            )
            if raw_original_id is not None:
                expected_point_id = _legacy_qdrant_point_id(raw_original_id)
                if str(raw_point_id) != str(expected_point_id):
                    raise MigrationError(
                        f"source point-id does not match legacy encoding for "
                        f"{raw_original_id!r}: expected={expected_point_id!r} "
                        f"found={raw_point_id!r}"
                    )
            logical_id, transformed, terms = self._transform_point(
                point,
                layout=layout,
                schema=schema,
            )
            previous_logical = source_logical_ids.get(logical_id)
            if previous_logical is not None:
                if previous_logical[1] != raw_identity:
                    raise MigrationError(
                        f"physical target point-id collision between "
                        f"{previous_logical[0][1]!r} and {logical_id!r}"
                    )
                raise MigrationError(f"duplicate logical source record id {logical_id!r}")
            source_logical_ids[logical_id] = (source_point_key, raw_identity)
            target_id = str(transformed["id"])
            previous_identity = source_identities.get(target_id)
            if previous_identity is not None and previous_identity != raw_identity:
                raise MigrationError(
                    "physical target point-id collision between "
                    f"{previous_identity[1]!r} and {logical_id!r}"
                )
            source_identities[target_id] = raw_identity
            id_map[logical_id] = target_id
            sparse_terms.update(terms)
            fingerprint_payload = dict(transformed["payload"])
            if "owner_user_id" in raw_payload:
                fingerprint_payload["owner_user_id"] = raw_payload["owner_user_id"]
            else:
                fingerprint_payload.pop("owner_user_id", None)
            fingerprint_point = dict(transformed)
            fingerprint_point["payload"] = fingerprint_payload
            point_fingerprints.append(
                _point_fingerprint(
                    source_point_id=raw_point_id,
                    transformed=fingerprint_point,
                )
            )
            payload = transformed["payload"]
            if not _acl_complete(payload):
                acl_incomplete_count += 1
            if capture_points:
                transformed_points.append(transformed)

        if source_scanned != source_count:
            raise MigrationError(
                "source count mismatch: "
                f"exact count={source_count} paginated count={source_scanned}"
            )
        return SourceSnapshot(
            source_count=source_count,
            id_map=id_map,
            sparse_terms=sparse_terms,
            fingerprint=_snapshot_fingerprint(point_fingerprints),
            acl_incomplete_count=acl_incomplete_count,
            points=tuple(transformed_points),
        )

    def _marker_payload(
        self,
        *,
        layout: CollectionLayout,
        metadata: LegacyMetadata,
        sparse_weight: float,
        source_fingerprint: str,
        metadata_fingerprint: str,
        sparse_map_fingerprint: str,
        setup_complete: bool,
        acl_incomplete_count: int,
    ) -> dict[str, Any]:
        return {
            "_openviking_meta_version": _META_VERSION,
            "collection_name": self.target_collection,
            "source_collection": self.source_collection,
            "source_metadata_collection": self.source_metadata_collection,
            "source_fingerprint": source_fingerprint,
            "metadata_fingerprint": metadata_fingerprint,
            "sparse_map_fingerprint": sparse_map_fingerprint,
            "setup_complete": setup_complete,
            "acl_incomplete_count": acl_incomplete_count,
            "schema": metadata.schema,
            "dense_vector_name": layout.dense_vector_name,
            "sparse_vector_name": layout.sparse_vector_name,
            "vector_dim": layout.vector_dimension,
            "distance": layout.distance,
            "dense_datatype": layout.dense_datatype,
            "sparse_enabled": layout.sparse_enabled,
            "sparse_modifier": layout.sparse_modifier,
            "sparse_weight": sparse_weight,
            "indexes": metadata.indexes,
        }

    def _sparse_weight(self, metadata: LegacyMetadata, *, sparse_enabled: bool) -> float:
        values: list[float] = []
        for meta in metadata.indexes.values():
            if meta.get("SparseWeight") is not None:
                try:
                    value = float(meta["SparseWeight"])
                except (TypeError, ValueError) as exc:
                    raise MigrationError("legacy SparseWeight is invalid") from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise MigrationError("legacy SparseWeight must be between 0 and 1")
                values.append(value)
                continue
            vector_index = meta.get("VectorIndex")
            if isinstance(vector_index, Mapping) and vector_index.get(
                "SearchWithSparseLogitAlpha"
            ) is not None:
                try:
                    value = float(vector_index["SearchWithSparseLogitAlpha"])
                except (TypeError, ValueError) as exc:
                    raise MigrationError("legacy sparse alpha is invalid") from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise MigrationError("legacy sparse alpha must be between 0 and 1")
                values.append(value)
        if values and any(value != values[0] for value in values[1:]):
            raise MigrationError("legacy indexes have conflicting sparse weights")
        if values:
            return values[0]
        return 0.5 if sparse_enabled else 0.0

    def _load_current_marker(self) -> dict[str, Any] | None:
        if not self._exists(self.target_metadata_collection):
            return None
        points = self._retrieve(
            self.target_metadata_collection,
            [_META_MARKER_ID],
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].get("payload")
        return dict(payload) if isinstance(payload, Mapping) else None

    def _validate_metadata_layout(self, collection: str) -> None:
        layout = self._layout(
            self._collection_info(collection),
            honor_overrides=False,
        )
        if (
            layout.dense_vector_name != _META_VECTOR_NAME
            or layout.vector_dimension != 1
            or layout.distance != "Dot"
            or layout.sparse_enabled
        ):
            raise MigrationError(
                f"target metadata collection has an incompatible vector layout: {collection}"
            )

    def _validate_existing_target(
        self,
        *,
        target_info: Mapping[str, Any] | None,
        marker: Mapping[str, Any],
        layout: CollectionLayout,
        metadata: LegacyMetadata,
    ) -> None:
        required_fields = {
            "_openviking_meta_version",
            "collection_name",
            "source_collection",
            "source_metadata_collection",
            "source_fingerprint",
            "metadata_fingerprint",
            "sparse_map_fingerprint",
            "setup_complete",
            "acl_incomplete_count",
            "schema",
            "dense_vector_name",
            "sparse_vector_name",
            "vector_dim",
            "distance",
            "dense_datatype",
            "sparse_enabled",
            "sparse_modifier",
            "sparse_weight",
            "indexes",
        }
        missing_fields = sorted(field for field in required_fields if field not in marker)
        if missing_fields:
            raise MigrationError(
                "target current marker is missing required fields: "
                f"{missing_fields!r}"
            )
        if marker.get("_openviking_meta_version") != _META_VERSION:
            raise MigrationError(
                "target metadata collection has no valid current marker "
                f"(expected version {_META_VERSION})"
            )
        for field_name, expected in (
            ("collection_name", self.target_collection),
            ("source_collection", self.source_collection),
            ("source_metadata_collection", self.source_metadata_collection),
        ):
            if marker.get(field_name) != expected:
                raise MigrationError(
                    f"target current marker {field_name} differs: "
                    f"expected={expected!r} target={marker.get(field_name)!r}"
                )
        if not isinstance(marker.get("source_fingerprint"), str) or not marker["source_fingerprint"]:
            raise MigrationError(
                "target current marker has no valid source fingerprint"
            )
        if (
            not isinstance(marker.get("metadata_fingerprint"), str)
            or not marker["metadata_fingerprint"]
        ):
            raise MigrationError("target current marker has no valid metadata fingerprint")
        if (
            not isinstance(marker.get("sparse_map_fingerprint"), str)
            or not marker["sparse_map_fingerprint"]
        ):
            raise MigrationError("target current marker has no valid sparse-map fingerprint")
        if not isinstance(marker.get("setup_complete"), bool):
            raise MigrationError("target current marker has an invalid setup_complete flag")
        acl_incomplete_count = marker.get("acl_incomplete_count")
        if (
            isinstance(acl_incomplete_count, bool)
            or not isinstance(acl_incomplete_count, int)
            or acl_incomplete_count < 0
        ):
            raise MigrationError("target current marker has an invalid ACL-incomplete count")
        if not isinstance(marker.get("schema"), Mapping):
            raise MigrationError("target current marker has no schema")
        dense_vector_name = marker.get("dense_vector_name")
        sparse_vector_name = marker.get("sparse_vector_name")
        vector_dim = marker.get("vector_dim")
        dense_datatype = marker.get("dense_datatype")
        sparse_enabled = marker.get("sparse_enabled")
        sparse_modifier = marker.get("sparse_modifier")
        if (
            not isinstance(dense_vector_name, str)
            or not dense_vector_name
            or not isinstance(sparse_vector_name, str)
            or not sparse_vector_name
            or isinstance(vector_dim, bool)
            or not isinstance(vector_dim, int)
            or vector_dim <= 0
            or (dense_datatype is not None and not isinstance(dense_datatype, str))
            or not isinstance(sparse_enabled, bool)
            or (sparse_modifier is not None and not isinstance(sparse_modifier, str))
        ):
            raise MigrationError("target current marker has an invalid vector layout")
        try:
            marker_distance = _canonical_distance(marker["distance"])
            marker_sparse_weight = float(marker["sparse_weight"])
        except (TypeError, ValueError) as exc:
            raise MigrationError("target current marker has invalid distance or sparse weight") from exc
        if (
            marker_distance != layout.distance
            or not math.isfinite(marker_sparse_weight)
            or not 0.0 <= marker_sparse_weight <= 1.0
        ):
            raise MigrationError("target current marker has an incompatible search policy")
        marker_layout = {
            "dense_vector_name": dense_vector_name,
            "sparse_vector_name": sparse_vector_name,
            "vector_dim": vector_dim,
            "distance": marker_distance,
            "dense_datatype": dense_datatype,
            "sparse_enabled": sparse_enabled,
            "sparse_modifier": sparse_modifier,
        }
        expected_layout = {
            "dense_vector_name": layout.dense_vector_name,
            "sparse_vector_name": layout.sparse_vector_name,
            "vector_dim": layout.vector_dimension,
            "distance": layout.distance,
            "dense_datatype": layout.dense_datatype,
            "sparse_enabled": layout.sparse_enabled,
            "sparse_modifier": layout.sparse_modifier,
        }
        for key, expected in expected_layout.items():
            if marker_layout[key] != expected:
                raise MigrationError(
                    f"target current marker differs for {key}: "
                    f"source={expected!r} target={marker_layout[key]!r}"
                )
        indexes = marker.get("indexes")
        if not isinstance(indexes, Mapping) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(value, Mapping)
            for name, value in indexes.items()
        ):
            raise MigrationError("target current marker has invalid indexes")
        for index_name, index_meta in indexes.items():
            _validate_scalar_index(
                index_meta.get("ScalarIndex"),
                label=f"target current marker index {index_name!r}",
            )
        _validate_schema_subset(metadata.schema, marker["schema"])
        for index_name in metadata.indexes:
            target_index = indexes.get(index_name)
            if not isinstance(target_index, Mapping):
                raise MigrationError(
                    f"target current marker is missing index {index_name!r}"
                )
        if target_info is None:
            target_layout = None
        else:
            target_layout = self._layout(target_info)
            for field_name in (
                "dense_vector_name",
                "sparse_vector_name",
                "vector_dimension",
                "distance",
                "dense_datatype",
                "sparse_enabled",
                "sparse_modifier",
            ):
                expected = getattr(layout, field_name)
                actual = getattr(target_layout, field_name)
                if expected != actual:
                    raise MigrationError(
                        f"target collection layout differs for {field_name}: "
                        f"source={expected!r} target={actual!r}"
                    )
            if marker["setup_complete"]:
                self._validate_payload_indexes(marker["schema"], indexes)

        expected_indexes = set(metadata.indexes)
        actual_indexes = set(indexes)
        if actual_indexes != expected_indexes:
            missing = sorted(expected_indexes - actual_indexes)
            extra = sorted(actual_indexes - expected_indexes)
            raise MigrationError(
                "target current marker index map differs: "
                f"missing={missing!r} extra={extra!r}"
            )
        for index_name, source_index in metadata.indexes.items():
            target_index = indexes[index_name]
            if target_index != source_index:
                raise MigrationError(
                    f"target current marker index {index_name!r} changed"
                )

    def _assert_source_layout(self, expected: CollectionLayout, *, phase: str) -> None:
        actual = self._layout(self._collection_info(self.source_collection))
        if actual != expected:
            raise MigrationError(
                f"source physical layout changed {phase}; "
                "rerun preflight with source writes frozen"
            )

    def _existing_target_points(
        self,
        *,
        with_vectors: bool = False,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        points: dict[str, dict[str, Any]] = {}
        scanned = 0
        for point in self._scroll(self.target_collection, with_vectors=with_vectors):
            scanned += 1
            point_id = point.get("id")
            if point_id is None:
                raise MigrationError("target collection contains a point without an id")
            key = str(point_id)
            if key in points:
                raise MigrationError(f"target pagination returned duplicate point id {key!r}")
            points[key] = point
        return points, scanned

    @staticmethod
    def _expected_payload_indexes(
        schema: Mapping[str, Any],
        indexes: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, str]:
        fields = schema.get("Fields", [])
        field_list = fields if isinstance(fields, list) else []
        expected: dict[str, str] = {
            "uri_depth": "integer",
            "scope_roots": "keyword",
        }
        for metadata in indexes.values():
            scalar = metadata.get("ScalarIndex")
            if isinstance(scalar, Mapping):
                scalar = list(scalar)
            if isinstance(scalar, (list, tuple, set)):
                for field_name in scalar:
                    name = str(field_name)
                    expected[name] = (
                        "integer"
                        if name == "uri_depth"
                        else qdrant_payload_field_schema(name, field_list)
                    )
        return expected

    def _payload_schema(self, info: Mapping[str, Any]) -> Mapping[str, Any]:
        value = info.get("payload_schema")
        if not isinstance(value, Mapping):
            raise MigrationError("Qdrant collection response has no payload_schema")
        return value

    @staticmethod
    def _physical_index_type(value: Mapping[str, Any]) -> str | None:
        raw_type = value.get("data_type") or value.get("type")
        params = value.get("params")
        if raw_type is None and isinstance(params, Mapping):
            raw_type = params.get("type") or params.get("data_type")
        if raw_type is None:
            return None
        normalized = str(raw_type).strip().lower()
        aliases = {
            "int": "integer",
            "int64": "integer",
            "uint": "integer",
            "uint64": "integer",
            "double": "float",
            "bool": "bool",
            "boolean": "bool",
            "datetime": "datetime",
            "date_time": "datetime",
        }
        return aliases.get(normalized, normalized)

    def _validate_payload_indexes(
        self,
        schema: Mapping[str, Any],
        indexes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        physical = self._payload_schema(self._collection_info(self.target_collection))
        for field_name, expected_type in self._expected_payload_indexes(schema, indexes).items():
            value = physical.get(field_name)
            if not isinstance(value, Mapping):
                raise MigrationError(
                    f"target collection is missing payload index {field_name!r}"
                )
            actual_type = self._physical_index_type(value)
            if actual_type != expected_type:
                raise MigrationError(
                    f"target payload index {field_name!r} has incompatible schema: "
                    f"expected={expected_type!r} target={actual_type!r}"
                )

    def _existing_sparse_dictionary(self) -> tuple[dict[str, int], dict[int, str]]:
        if not self._exists(self.target_metadata_collection):
            return {}, {}
        by_term: dict[str, int] = {}
        by_index: dict[int, str] = {}
        for point in self._scroll(self.target_metadata_collection, with_vectors=False):
            point_id = point.get("id")
            if point_id is not None and str(point_id) == _META_MARKER_ID:
                continue
            payload = point.get("payload")
            if not isinstance(payload, Mapping) or payload.get(_SPARSE_TERM_MARKER) is not True:
                raise SparseMigrationError(
                    "target metadata collection contains an unexpected point "
                    f"{point_id!r}"
                )
            term = payload.get("term")
            raw_index = payload.get("index")
            if not isinstance(term, str) or not term.strip():
                raise SparseMigrationError("target sparse dictionary has an invalid term")
            expected_point_id = to_qdrant_point_id(f"openviking:sparse:{term}")
            if point_id is None or str(point_id) != expected_point_id:
                raise SparseMigrationError(
                    "target sparse dictionary point-id collision for term "
                    f"{term!r}: expected={expected_point_id!r} found={point_id!r}"
                )
            index = _sparse_index(
                raw_index,
                field_name=f"target sparse dictionary index for {term!r}",
                minimum=1,
            )
            expected_index = stable_sparse_index(term)
            if index != expected_index:
                raise SparseMigrationError(
                    "target sparse dictionary has an index that does not match "
                    f"the stable term mapping for {term!r}: "
                    f"expected={expected_index} found={index}"
                )
            previous_index = by_term.get(term)
            if previous_index is not None and previous_index != index:
                raise SparseMigrationError(
                    f"target sparse dictionary maps {term!r} to multiple indexes"
                )
            previous_term = by_index.get(index)
            if previous_term is not None and previous_term != term:
                raise SparseMigrationError(
                    "target sparse dictionary collision: "
                    f"index={index} terms={previous_term!r},{term!r}"
                )
            by_term[term] = index
            by_index[index] = term
        return by_term, by_index

    def _assert_sparse_dictionary_complete(self, terms: set[str]) -> None:
        if not terms:
            return
        by_term, _ = self._existing_sparse_dictionary()
        missing = sorted(terms - set(by_term))
        if missing:
            raise SparseMigrationError(
                "target sparse dictionary is missing terms after write: "
                f"{missing!r}"
            )
        for term in terms:
            expected_index = stable_sparse_index(term)
            if by_term[term] != expected_index:
                raise SparseMigrationError(
                    "target sparse dictionary has an incompatible index for "
                    f"{term!r}: expected={expected_index} found={by_term[term]}"
                )

    def _validate_sparse_terms(self, terms: set[str]) -> None:
        by_term, by_index = self._existing_sparse_dictionary()
        planned: dict[int, str] = {}
        for term in sorted(terms):
            index = stable_sparse_index(term)
            previous_term = planned.get(index)
            if previous_term is not None and previous_term != term:
                raise SparseMigrationError(
                    "sparse term collision after migration: "
                    f"index={index} terms={previous_term!r},{term!r}"
                )
            existing_term = by_index.get(index)
            if existing_term is not None and existing_term != term:
                raise SparseMigrationError(
                    "target sparse dictionary collision: "
                    f"index={index} existing_term={existing_term!r} source_term={term!r}"
                )
            existing_index = by_term.get(term)
            if existing_index is not None and existing_index != index:
                raise SparseMigrationError(
                    "target sparse dictionary maps source term to a different index: "
                    f"term={term!r} existing={existing_index} expected={index}"
                )
            planned[index] = term
        if terms and self._exists(self.target_metadata_collection):
            expected_ids = {
                term: to_qdrant_point_id(f"openviking:sparse:{term}") for term in terms
            }
            existing_points = self._retrieve(
                self.target_metadata_collection,
                list(expected_ids.values()),
                with_vectors=False,
            )
            points_by_id = {
                str(point.get("id")): point
                for point in existing_points
                if point.get("id") is not None
            }
            for term, point_id in expected_ids.items():
                point = points_by_id.get(point_id)
                if point is None:
                    continue
                payload = point.get("payload")
                if (
                    not isinstance(payload, Mapping)
                    or not payload.get(_SPARSE_TERM_MARKER)
                    or payload.get("term") != term
                    or payload.get("index") != stable_sparse_index(term)
                ):
                    raise SparseMigrationError(
                        "target sparse dictionary point-id collision for term "
                        f"{term!r}: point={point_id!r}"
                    )

    def _validate_source_metadata_layout(
        self,
        schema: Mapping[str, Any],
        layout: CollectionLayout,
    ) -> None:
        fields = schema.get("Fields", [])
        if not isinstance(fields, list):
            raise MigrationError("legacy collection metadata schema Fields must be a list")
        dense_fields = [
            field
            for field in fields
            if isinstance(field, Mapping)
            and str(field.get("FieldType") or "").strip().lower() == "vector"
        ]
        if len(dense_fields) != 1:
            raise MigrationError(
                "legacy collection metadata must declare exactly one dense vector field"
            )
        dense_field = dense_fields[0]
        raw_dimension = dense_field.get("Dim")
        if (
            isinstance(raw_dimension, bool)
            or not isinstance(raw_dimension, int)
            or raw_dimension != layout.vector_dimension
        ):
            raise MigrationError(
                "legacy dense vector metadata dimension does not match the "
                f"physical source layout: metadata={raw_dimension!r} "
                f"physical={layout.vector_dimension}"
            )
        declared_dense_name = str(dense_field.get("FieldName"))
        if declared_dense_name != layout.dense_vector_name and not (
            self._dense_vector_name_override
            and declared_dense_name == "vector"
            and layout.dense_vector_name == self._dense_vector_name_override
        ):
            raise MigrationError(
                "legacy dense vector metadata name does not match the physical "
                f"source layout: metadata={declared_dense_name!r} "
                f"physical={layout.dense_vector_name!r}"
            )

        sparse_fields = [
            field
            for field in fields
            if isinstance(field, Mapping)
            and str(field.get("FieldType") or "").strip().lower() == "sparse_vector"
        ]
        if not layout.sparse_enabled:
            if sparse_fields:
                raise MigrationError(
                    "legacy metadata declares a sparse vector but the physical "
                    "source layout has no sparse vectors"
                )
            return
        if len(sparse_fields) != 1:
            raise MigrationError(
                "legacy collection metadata must declare exactly one sparse vector "
                "when the physical source layout enables sparse vectors"
            )
        declared_sparse_name = str(sparse_fields[0].get("FieldName"))
        if declared_sparse_name != layout.sparse_vector_name and not (
            self._sparse_vector_name_override
            and declared_sparse_name == "sparse_vector"
            and layout.sparse_vector_name == self._sparse_vector_name_override
        ):
            raise MigrationError(
                "legacy sparse vector metadata name does not match the physical "
                f"source layout: metadata={declared_sparse_name!r} "
                f"physical={layout.sparse_vector_name!r}"
            )

    def preflight(self) -> MigrationPlan:
        """Read and validate source/target state without mutating Qdrant."""

        if not self._exists(self.source_collection):
            raise MigrationError(f"source collection does not exist: {self.source_collection}")
        source_info = self._collection_info(self.source_collection)
        layout = self._layout(source_info)
        metadata = self._legacy_metadata()
        self._validate_source_metadata_layout(metadata.schema, layout)
        metadata_fingerprint = _metadata_fingerprint(metadata)
        sparse_weight = self._sparse_weight(metadata, sparse_enabled=layout.sparse_enabled)
        sparse_map_fingerprint = _sparse_map_fingerprint(self._sparse_map)
        source = self._scan_source(layout=layout, schema=metadata.schema)

        target_exists = self._exists(self.target_collection)
        target_count = 0
        target_points: dict[str, dict[str, Any]] = {}
        marker: dict[str, Any] | None = None
        target_metadata_exists = self._exists(self.target_metadata_collection)
        if target_exists:
            marker = self._load_current_marker()
            if marker is None:
                raise MigrationError(
                    "target collection exists but has no valid current marker; "
                    "refusing to adopt or overwrite it"
                )
            self._validate_metadata_layout(self.target_metadata_collection)
            target_info = self._collection_info(self.target_collection)
            self._validate_existing_target(
                target_info=target_info,
                marker=marker,
                layout=layout,
                metadata=metadata,
            )
            target_count = self._count(self.target_collection)
            target_points, target_scanned = self._existing_target_points()
            if target_scanned != target_count:
                raise MigrationError(
                    "target count mismatch: "
                    f"exact count={target_count} paginated count={target_scanned}"
                )
        elif target_metadata_exists:
            marker = self._load_current_marker()
            if marker is None:
                raise MigrationError(
                    "target metadata collection exists without a valid migration marker; "
                    "refusing to adopt or overwrite it"
                )
            self._validate_metadata_layout(self.target_metadata_collection)
            self._validate_existing_target(
                target_info=None,
                marker=marker,
                layout=layout,
                metadata=metadata,
            )
            if marker.get("setup_complete") is not False:
                raise MigrationError(
                    "target metadata marker is complete but the target collection is missing"
                )

        for logical_id, target_id in source.id_map.items():
            existing = target_points.get(target_id)
            if existing is None:
                continue
            existing_payload = existing.get("payload")
            existing_original = (
                existing_payload.get(_ORIGINAL_ID_FIELD)
                if isinstance(existing_payload, Mapping)
                else None
            )
            if existing_original is None:
                raise MigrationError(
                    f"target point {target_id} is missing {_ORIGINAL_ID_FIELD}; "
                    "refusing to overwrite it"
                )
            if str(existing_original) != logical_id:
                raise MigrationError(
                    f"target point-id collision for {target_id}: "
                    f"existing={existing_original!r} source={logical_id!r}"
                )
        extra_target_ids = set(target_points) - set(source.id_map.values())
        if extra_target_ids:
            raise MigrationError(
                "target collection contains records absent from the source snapshot: "
                f"{sorted(extra_target_ids)!r}"
            )

        if marker is not None:
            if marker.get("source_fingerprint") != source.fingerprint:
                raise MigrationError(
                    "target current marker source fingerprint differs from the source snapshot"
                )
            if marker.get("metadata_fingerprint") != metadata_fingerprint:
                raise MigrationError(
                    "target current marker metadata fingerprint differs from the source metadata"
                )
            if marker.get("sparse_map_fingerprint") != sparse_map_fingerprint:
                raise MigrationError(
                    "target current marker sparse-map fingerprint differs from the sparse map"
                )
        self._validate_sparse_terms(source.sparse_terms)

        return MigrationPlan(
            source_collection=self.source_collection,
            target_collection=self.target_collection,
            source_metadata_collection=self.source_metadata_collection,
            target_metadata_collection=self.target_metadata_collection,
            source_count=source.source_count,
            target_count=target_count,
            target_exists=target_exists,
            dense_vector_name=layout.dense_vector_name,
            sparse_vector_name=layout.sparse_vector_name,
            vector_dimension=layout.vector_dimension,
            distance=layout.distance,
            dense_datatype=layout.dense_datatype,
            sparse_enabled=layout.sparse_enabled,
            sparse_modifier=layout.sparse_modifier,
            sparse_weight=sparse_weight,
            sparse_terms=source.sparse_terms,
            id_map=source.id_map,
            existing_target_ids=set(target_points),
            source_fingerprint=source.fingerprint,
            metadata_fingerprint=metadata_fingerprint,
            sparse_map_fingerprint=sparse_map_fingerprint,
            acl_incomplete_count=source.acl_incomplete_count,
            target_metadata_exists=target_metadata_exists,
        )

    def _create_collection(self, name: str, body: dict[str, Any]) -> None:
        try:
            self._request("PUT", self._path(name), body, params={"wait": "true"})
        except Exception as exc:
            if _status(exc) == 409:
                raise MigrationError(
                    f"target collection appeared during migration: {name}"
                ) from exc
            raise

    def _delete_collection(self, name: str) -> None:
        try:
            self._request("DELETE", self._path(name), params={"timeout": 30})
        except Exception as exc:
            if _status(exc) != 404:
                raise

    def _write_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        self._request(
            "PUT",
            self._path(collection, "/points"),
            {"points": points},
            params={"wait": "true"},
        )

    def _write_marker(self, marker: Mapping[str, Any]) -> None:
        self._write_points(
            self.target_metadata_collection,
            [
                {
                    "id": _META_MARKER_ID,
                    "vector": {_META_VECTOR_NAME: [0.0]},
                    "payload": dict(marker),
                }
            ],
        )

    def _write_indexes(self, schema: Mapping[str, Any], indexes: Mapping[str, Mapping[str, Any]]) -> None:
        for field_name, field_schema in self._expected_payload_indexes(schema, indexes).items():
            try:
                self._request(
                    "PUT",
                    self._path(self.target_collection, "/index"),
                    {
                        "field_name": field_name,
                        "field_schema": field_schema,
                    },
                    params={"wait": "true"},
                )
            except Exception as exc:
                if _status(exc) != 409:
                    raise
        self._validate_payload_indexes(schema, indexes)

    def _write_sparse_dictionary(self, terms: set[str]) -> None:
        if not terms:
            return
        self._validate_sparse_terms(terms)
        existing, _ = self._existing_sparse_dictionary()
        missing = terms - set(existing)
        points = []
        for term in sorted(missing):
            points.append(
                {
                    "id": to_qdrant_point_id(f"openviking:sparse:{term}"),
                    "vector": {_META_VECTOR_NAME: [0.0]},
                    "payload": {
                        _SPARSE_TERM_MARKER: True,
                        "term": term,
                        "index": stable_sparse_index(term),
                    },
                }
            )
        self._write_points(self.target_metadata_collection, points)
        self._assert_sparse_dictionary_complete(terms)

    @staticmethod
    def _assert_source_snapshot(
        expected: MigrationPlan,
        actual: SourceSnapshot,
        *,
        phase: str,
    ) -> None:
        if (
            actual.source_count != expected.source_count
            or actual.id_map != expected.id_map
            or actual.sparse_terms != expected.sparse_terms
            or actual.fingerprint != expected.source_fingerprint
            or actual.acl_incomplete_count != expected.acl_incomplete_count
        ):
            raise MigrationError(
                f"source changed {phase}; rerun preflight with source writes frozen"
            )

    @staticmethod
    def _reviewed_plan_fields(plan: MigrationPlan) -> dict[str, Any]:
        fields = plan.to_dict()
        for name in (
            "target_count",
            "target_exists",
            "existing_target_ids",
            "target_metadata_exists",
        ):
            fields.pop(name, None)
        return fields

    @staticmethod
    def _assert_target_vectors(
        point: Mapping[str, Any],
        expected: Mapping[str, Any],
        *,
        exact: bool = True,
        sparse_dictionary: Mapping[int, str] | None = None,
    ) -> None:
        actual_vectors = point.get("vector")
        if actual_vectors is None:
            actual_vectors = point.get("vectors")
        expected_vectors = expected.get("vector")
        if not isinstance(actual_vectors, Mapping) or not isinstance(
            expected_vectors, Mapping
        ):
            raise MigrationError(
                f"target point {point.get('id')!r} has an invalid vector payload"
            )
        if set(actual_vectors) != set(expected_vectors):
            raise MigrationError(
                f"target point {point.get('id')!r} vector names differ: "
                f"expected={sorted(expected_vectors)!r} "
                f"target={sorted(actual_vectors)!r}"
            )
        for name, expected_value in expected_vectors.items():
            actual_value = actual_vectors.get(name)
            if isinstance(expected_value, list):
                actual_normalized = _finite_vector(
                    actual_value,
                    field_name=f"target point {point.get('id')!r} vector {name!r}",
                    dimension=len(expected_value),
                )
                expected_normalized = _finite_vector(
                    expected_value,
                    field_name=f"expected point vector {name!r}",
                )
                if exact and actual_normalized != expected_normalized:
                    raise MigrationError(
                        f"target point {point.get('id')!r} vector {name!r} differs"
                    )
                continue
            if not isinstance(expected_value, Mapping) or not isinstance(
                actual_value, Mapping
            ):
                raise MigrationError(
                    f"target point {point.get('id')!r} vector {name!r} differs"
                )
            if set(actual_value) != {"indices", "values"}:
                raise MigrationError(
                    f"target point {point.get('id')!r} sparse vector {name!r} is malformed"
                )
            actual_indices = actual_value.get("indices")
            actual_values = actual_value.get("values")
            expected_indices = expected_value.get("indices")
            expected_values = expected_value.get("values")
            if (
                not isinstance(actual_indices, list)
                or not isinstance(actual_values, list)
                or len(actual_indices) != len(actual_values)
                or not isinstance(expected_indices, list)
                or not isinstance(expected_values, list)
                or len(expected_indices) != len(expected_values)
            ):
                raise MigrationError(
                    f"target point {point.get('id')!r} sparse vector {name!r} is malformed"
                )
            actual_index_values = [
                _sparse_index(
                    index,
                    field_name=f"target point {point.get('id')!r} sparse index",
                )
                for index in actual_indices
            ]
            expected_index_values = [
                _sparse_index(
                    index,
                    field_name=f"expected point sparse index {name!r}",
                )
                for index in expected_indices
            ]
            if sparse_dictionary is not None:
                missing_dictionary_indexes = sorted(
                    set(actual_index_values) - set(sparse_dictionary)
                )
                if missing_dictionary_indexes:
                    raise MigrationError(
                        f"target point {point.get('id')!r} sparse vector {name!r} "
                        "contains indexes missing from the target sparse dictionary: "
                        f"{missing_dictionary_indexes!r}"
                    )
            try:
                actual_weight_values = [float(value) for value in actual_values]
                expected_weight_values = [float(value) for value in expected_values]
            except (TypeError, ValueError) as exc:
                raise MigrationError(
                    f"target point {point.get('id')!r} sparse vector {name!r} is malformed"
                ) from exc
            if (
                len(set(actual_index_values)) != len(actual_index_values)
                or not all(math.isfinite(value) for value in actual_weight_values)
                or (
                    exact
                    and (
                        actual_index_values != expected_index_values
                        or actual_weight_values != expected_weight_values
                    )
                )
            ):
                raise MigrationError(
                    f"target point {point.get('id')!r} sparse vector {name!r} differs"
                )

    def _validate_final_target(
        self,
        *,
        source: SourceSnapshot,
        schema: Mapping[str, Any],
        copied_target_ids: set[str],
        allow_acl_fail_open: bool,
    ) -> int:
        target_count = self._count(self.target_collection)
        target_points, target_scanned = self._existing_target_points(with_vectors=True)
        if target_scanned != target_count:
            raise MigrationError(
                "target count mismatch after copy: "
                f"exact count={target_count} paginated count={target_scanned}"
            )
        expected_ids = set(source.id_map.values())
        actual_ids = set(target_points)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extras = sorted(actual_ids - expected_ids)
            raise MigrationError(
                "target records differ after copy: "
                f"missing={missing!r} extras={extras!r}"
            )
        expected_points = {
            str(point["id"]): point
            for point in source.points
        }
        if set(expected_points) != expected_ids:
            raise MigrationError("source snapshot has an inconsistent target id map")
        _, sparse_by_index = self._existing_sparse_dictionary()
        for target_id, expected in expected_points.items():
            actual = target_points[target_id]
            payload = actual.get("payload")
            expected_payload = expected.get("payload")
            if not isinstance(payload, Mapping) or not isinstance(expected_payload, Mapping):
                raise MigrationError(
                    f"target point {target_id!r} has an invalid payload"
                )
            if str(payload.get(_ORIGINAL_ID_FIELD)) != str(
                expected_payload.get(_ORIGINAL_ID_FIELD)
            ):
                raise MigrationError(
                    f"target point {target_id!r} has the wrong original id"
                )
            self._validate_target_payload(
                payload,
                expected_payload,
                schema=schema,
                allow_acl_fail_open=allow_acl_fail_open,
                point_id=target_id,
            )
            copied = target_id in copied_target_ids
            if copied and payload != expected_payload:
                raise MigrationError(
                    f"target point {target_id!r} payload differs from copied source"
                )
            self._assert_target_vectors(
                actual,
                expected,
                exact=copied,
                sparse_dictionary=sparse_by_index,
            )
        self._assert_sparse_dictionary_complete(source.sparse_terms)
        return target_count

    @staticmethod
    def _validate_target_payload(
        payload: Mapping[str, Any],
        expected_payload: Mapping[str, Any],
        *,
        schema: Mapping[str, Any],
        allow_acl_fail_open: bool,
        point_id: str,
    ) -> None:
        """Validate security-sensitive payload fields without clobbering newer data."""

        _assert_security_payload(payload, expected_payload, point_id=point_id)
        uri = payload.get("uri")
        normalized_uri = _normalize_uri(uri, field_name=f"target point {point_id!r} uri")
        if uri != normalized_uri:
            raise MigrationError(f"target point {point_id!r} uri is not canonical")
        uri_depth = payload.get("uri_depth")
        if (
            isinstance(uri_depth, bool)
            or not isinstance(uri_depth, int)
            or uri_depth != _path_depth(normalized_uri)
        ):
            raise MigrationError(f"target point {point_id!r} has invalid uri_depth")
        scope_roots = payload.get("scope_roots")
        if scope_roots != _scope_roots(normalized_uri):
            raise MigrationError(f"target point {point_id!r} has invalid scope_roots")

        schema_fields = {
            str(field.get("FieldName"))
            for field in schema.get("Fields", [])
            if isinstance(field, Mapping) and field.get("FieldName")
        }
        for identity_field in ("account_id",):
            if identity_field in expected_payload or identity_field in schema_fields:
                value = payload.get(identity_field)
                if not isinstance(value, str) or not value.strip():
                    raise MigrationError(
                        f"target point {point_id!r} is missing {identity_field}"
                    )
        expected_owner = owner_fields_for_uri(
            f"viking://{normalized_uri.lstrip('/')}"
        ).get("owner_user_id")
        actual_owner = payload.get("owner_user_id")
        if expected_owner is not None and actual_owner != expected_owner:
            raise MigrationError(
                f"target point {point_id!r} owner_user_id does not match uri"
            )

        if not _acl_complete(expected_payload) and not allow_acl_fail_open:
            raise MigrationError(
                f"target point {point_id!r} lacks complete ACL fields"
            )

    @staticmethod
    def _assert_marker_fingerprints(
        marker: Mapping[str, Any],
        plan: MigrationPlan,
    ) -> None:
        if marker.get("source_fingerprint") != plan.source_fingerprint:
            raise MigrationError(
                "target current marker source fingerprint changed after preflight"
            )
        if marker.get("metadata_fingerprint") != plan.metadata_fingerprint:
            raise MigrationError(
                "target current marker metadata fingerprint changed after preflight"
            )
        if marker.get("sparse_map_fingerprint") != plan.sparse_map_fingerprint:
            raise MigrationError(
                "target current marker sparse-map fingerprint changed after preflight"
            )

    def apply(
        self,
        *,
        confirm: bool = False,
        plan: MigrationPlan | None = None,
        allow_acl_fail_open: bool = False,
    ) -> MigrationResult:
        """Apply a plan; mutation is refused unless ``confirm=True``."""

        if not confirm:
            raise MigrationError("apply requires explicit confirm=True / --confirm")
        if plan is None:
            raise MigrationError("apply requires a reviewed migration plan")
        current_plan = self.preflight()
        if self._reviewed_plan_fields(plan) != self._reviewed_plan_fields(current_plan):
            raise MigrationError(
                "provided migration plan is stale; rerun preflight before apply"
            )
        plan = current_plan
        layout = CollectionLayout(
            dense_vector_name=plan.dense_vector_name,
            sparse_vector_name=plan.sparse_vector_name,
            vector_dimension=plan.vector_dimension,
            distance=plan.distance,
            dense_datatype=plan.dense_datatype,
            sparse_enabled=plan.sparse_enabled,
            sparse_modifier=plan.sparse_modifier,
        )
        metadata = self._legacy_metadata()
        self._validate_source_metadata_layout(metadata.schema, layout)
        metadata_fingerprint = _metadata_fingerprint(metadata)
        if metadata_fingerprint != plan.metadata_fingerprint:
            raise MigrationError(
                "legacy metadata changed after preflight; rerun preflight with writes frozen"
            )
        source = self._scan_source(
            layout=layout,
            schema=metadata.schema,
            capture_points=True,
        )
        self._assert_source_snapshot(plan, source, phase="preflight")
        if plan.acl_incomplete_count and not allow_acl_fail_open:
            raise MigrationError(
                f"{plan.acl_incomplete_count} records lack ACL fields; "
                "refusing cutover without --allow-acl-fail-open"
            )

        target_exists = self._exists(self.target_collection)
        target_metadata_exists = self._exists(self.target_metadata_collection)
        if (
            target_exists != plan.target_exists
            or (
                target_metadata_exists != plan.target_metadata_exists
                and not (
                    not plan.target_metadata_exists
                    and target_metadata_exists
                    and not target_exists
                )
            )
        ):
            raise MigrationError(
                "target state changed after preflight; rerun preflight before apply"
            )

        existing_marker: dict[str, Any] | None = None
        if target_metadata_exists:
            existing_marker = self._load_current_marker()
            if existing_marker is None:
                raise MigrationError("target marker disappeared after preflight")
            self._validate_metadata_layout(self.target_metadata_collection)
            self._validate_existing_target(
                target_info=(
                    self._collection_info(self.target_collection)
                    if target_exists
                    else None
                ),
                marker=existing_marker,
                layout=layout,
                metadata=metadata,
            )
            self._assert_marker_fingerprints(existing_marker, plan)
            if (
                not plan.target_metadata_exists
                and target_metadata_exists
                and existing_marker.get("setup_complete") is not False
            ):
                raise MigrationError(
                    "target metadata appeared after preflight without an incomplete "
                    "migration marker"
                )

        index_schema: Mapping[str, Any] = metadata.schema
        index_metadata: Mapping[str, Mapping[str, Any]] = metadata.indexes
        if existing_marker is not None:
            marker_schema = existing_marker["schema"]
            marker_indexes = existing_marker["indexes"]
            assert isinstance(marker_schema, Mapping)
            assert isinstance(marker_indexes, Mapping)
            index_schema = marker_schema
            index_metadata = marker_indexes

        self._assert_source_layout(layout, phase="target setup")

        if existing_marker is None:
            marker_incomplete = self._marker_payload(
                layout=layout,
                metadata=metadata,
                sparse_weight=plan.sparse_weight,
                source_fingerprint=plan.source_fingerprint,
                metadata_fingerprint=plan.metadata_fingerprint,
                sparse_map_fingerprint=plan.sparse_map_fingerprint,
                setup_complete=False,
                acl_incomplete_count=plan.acl_incomplete_count,
            )
        else:
            # Preserve target-side schema/policy extensions accepted during
            # preflight while toggling only the migration gate.
            marker_incomplete = dict(existing_marker)
            marker_incomplete["setup_complete"] = False
            marker_incomplete["acl_incomplete_count"] = plan.acl_incomplete_count
        marker_complete = dict(marker_incomplete)
        marker_complete["setup_complete"] = True

        target_created = False
        metadata_created = False
        marker_written = False
        try:
            if not target_metadata_exists:
                self._create_collection(
                    self.target_metadata_collection,
                    {"vectors": {_META_VECTOR_NAME: {"size": 1, "distance": "Dot"}}},
                )
                metadata_created = True
                # Reserve the target before creating its data collection. If
                # setup stops here, a later apply can resume from this
                # migration-owned incomplete marker.
                self._write_marker(marker_incomplete)
                marker_written = True

            if target_exists:
                # Keep an interrupted or repaired target unavailable until
                # the complete copy and final readback have succeeded.
                self._write_marker(marker_incomplete)
                marker_written = True
                self._write_indexes(index_schema, index_metadata)
                self._write_sparse_dictionary(source.sparse_terms)
            else:
                # Claim the data collection before writing its marker.  This
                # prevents a concurrent collection creator from inheriting a
                # migration marker after a 409 race.
                self._create_collection(
                    self.target_collection,
                    {
                        "vectors": {
                            layout.dense_vector_name: {
                                "size": layout.vector_dimension,
                                "distance": layout.distance,
                                **(
                                    {"datatype": layout.dense_datatype}
                                    if layout.dense_datatype is not None
                                    else {}
                                ),
                            }
                        },
                        **(
                            {
                                "sparse_vectors": {
                                    layout.sparse_vector_name: {
                                        **(
                                            {"modifier": layout.sparse_modifier}
                                            if layout.sparse_modifier is not None
                                            else {}
                                        )
                                    }
                                }
                            }
                            if layout.sparse_enabled
                            else {}
                        ),
                    },
                )
                target_created = True
                if not marker_written:
                    self._write_marker(marker_incomplete)
                    marker_written = True
                self._write_indexes(index_schema, index_metadata)
                self._write_sparse_dictionary(source.sparse_terms)

            migrated = 0
            skipped = 0
            copied_target_ids: set[str] = set()
            pending: list[dict[str, Any]] = []
            for transformed in source.points:
                pending.append(transformed)
                if len(pending) < self.batch_size:
                    continue
                migrated_batch, skipped_batch, copied_batch = self._apply_batch(pending)
                migrated += migrated_batch
                skipped += skipped_batch
                copied_target_ids.update(copied_batch)
                pending = []
            if pending:
                migrated_batch, skipped_batch, copied_batch = self._apply_batch(pending)
                migrated += migrated_batch
                skipped += skipped_batch
                copied_target_ids.update(copied_batch)

            self._assert_source_layout(layout, phase="final verification")
            final_source = self._scan_source(layout=layout, schema=metadata.schema)
            self._assert_source_snapshot(plan, final_source, phase="apply")
            final_metadata = self._legacy_metadata()
            if _metadata_fingerprint(final_metadata) != plan.metadata_fingerprint:
                raise MigrationError(
                    "legacy metadata changed during apply; rerun preflight with writes frozen"
                )
            target_count = self._validate_final_target(
                source=source,
                schema=metadata.schema,
                copied_target_ids=copied_target_ids,
                allow_acl_fail_open=allow_acl_fail_open,
            )
            self._validate_existing_target(
                target_info=self._collection_info(self.target_collection),
                marker=marker_incomplete,
                layout=layout,
                metadata=metadata,
            )
            final_marker = self._load_current_marker()
            if final_marker is None:
                raise MigrationError("target marker disappeared before completion")
            self._assert_marker_fingerprints(final_marker, plan)
            self._assert_source_layout(layout, phase="completion")
            self._write_marker(marker_complete)
            completed_marker = self._load_current_marker()
            if completed_marker is None or completed_marker.get("setup_complete") is not True:
                raise MigrationError("target completion marker was not persisted")
            self._assert_marker_fingerprints(completed_marker, plan)
        except Exception:
            if not marker_written:
                for collection in (
                    self.target_collection if target_created else None,
                    self.target_metadata_collection if metadata_created else None,
                ):
                    if collection is not None:
                        try:
                            self._delete_collection(collection)
                        except Exception:
                            pass
            raise

        return MigrationResult(
            source_count=plan.source_count,
            migrated_count=migrated,
            skipped_count=skipped,
            target_count=target_count,
            target_collection=self.target_collection,
        )

    def _apply_batch(self, points: list[dict[str, Any]]) -> tuple[int, int, set[str]]:
        existing = {
            str(point.get("id")): point
            for point in self._retrieve(
                self.target_collection,
                [str(point["id"]) for point in points],
                with_vectors=False,
            )
            if point.get("id") is not None
        }
        write: list[dict[str, Any]] = []
        skipped = 0
        for point in points:
            point_id = str(point["id"])
            current = existing.get(point_id)
            if current is None:
                write.append(point)
                continue
            payload = current.get("payload")
            current_id = payload.get(_ORIGINAL_ID_FIELD) if isinstance(payload, Mapping) else None
            source_id = point["payload"][_ORIGINAL_ID_FIELD]
            if current_id is None or str(current_id) != str(source_id):
                raise MigrationError(
                    f"target point-id collision for {point_id}: "
                    f"existing={current_id!r} source={source_id!r}"
                )
            expected_payload = point["payload"]
            if payload == expected_payload:
                skipped += 1
                continue
            if (
                isinstance(payload, Mapping)
                and isinstance(expected_payload, Mapping)
                and payload.get("owner_user_id") is None
            ):
                current_without_owner = dict(payload)
                expected_without_owner = dict(expected_payload)
                current_without_owner.pop("owner_user_id", None)
                expected_without_owner.pop("owner_user_id", None)
                if current_without_owner == expected_without_owner:
                    write.append(point)
                    continue
            skipped += 1
        self._write_points(self.target_collection, write)
        return len(write), skipped, {str(point["id"]) for point in write}


def _load_sparse_map(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read sparse map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError("sparse map JSON must be an object")
    return value


def _load_plan(path: str | None) -> MigrationPlan | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read migration plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError("migration plan JSON must be an object")
    expected_fields = {
        "source_collection",
        "target_collection",
        "source_metadata_collection",
        "target_metadata_collection",
        "source_count",
        "target_count",
        "target_exists",
        "dense_vector_name",
        "sparse_vector_name",
        "vector_dimension",
        "distance",
        "dense_datatype",
        "sparse_enabled",
        "sparse_modifier",
        "sparse_weight",
        "sparse_terms",
        "id_map",
        "existing_target_ids",
        "source_fingerprint",
        "metadata_fingerprint",
        "sparse_map_fingerprint",
        "acl_incomplete_count",
        "target_metadata_exists",
    }
    missing = sorted(expected_fields - set(value))
    extra = sorted(set(value) - expected_fields)
    if missing or extra:
        raise MigrationError(
            f"migration plan fields differ: missing={missing!r} extra={extra!r}"
        )

    def text(name: str, *, non_empty: bool = True) -> str:
        item = value[name]
        if not isinstance(item, str) or (non_empty and not item):
            raise MigrationError(f"migration plan field {name!r} must be a string")
        return item

    def integer(name: str, *, non_negative: bool = False) -> int:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise MigrationError(f"migration plan field {name!r} must be an integer")
        if non_negative and item < 0:
            raise MigrationError(f"migration plan field {name!r} must be non-negative")
        return item

    sparse_terms = value["sparse_terms"]
    if (
        not isinstance(sparse_terms, list)
        or any(not isinstance(term, str) for term in sparse_terms)
        or len(set(sparse_terms)) != len(sparse_terms)
    ):
        raise MigrationError("migration plan sparse_terms must be a list of unique strings")
    existing_target_ids = value["existing_target_ids"]
    if (
        not isinstance(existing_target_ids, list)
        or any(not isinstance(point_id, str) for point_id in existing_target_ids)
        or len(set(existing_target_ids)) != len(existing_target_ids)
    ):
        raise MigrationError(
            "migration plan existing_target_ids must be a list of unique strings"
        )
    id_map = value["id_map"]
    if (
        not isinstance(id_map, dict)
        or any(
            not isinstance(key, str) or not isinstance(point_id, str)
            for key, point_id in id_map.items()
        )
    ):
        raise MigrationError("migration plan id_map must map strings to strings")
    sparse_weight = value["sparse_weight"]
    if (
        isinstance(sparse_weight, bool)
        or not isinstance(sparse_weight, (int, float))
        or not math.isfinite(float(sparse_weight))
    ):
        raise MigrationError("migration plan sparse_weight must be finite")
    for name in ("target_exists", "sparse_enabled", "target_metadata_exists"):
        if not isinstance(value[name], bool):
            raise MigrationError(f"migration plan field {name!r} must be a boolean")
    for name in ("dense_datatype", "sparse_modifier"):
        if value[name] is not None and (
            not isinstance(value[name], str) or not value[name]
        ):
            raise MigrationError(f"migration plan field {name!r} must be a string or null")

    return MigrationPlan(
        source_collection=text("source_collection"),
        target_collection=text("target_collection"),
        source_metadata_collection=text("source_metadata_collection"),
        target_metadata_collection=text("target_metadata_collection"),
        source_count=integer("source_count", non_negative=True),
        target_count=integer("target_count", non_negative=True),
        target_exists=value["target_exists"],
        dense_vector_name=text("dense_vector_name"),
        sparse_vector_name=text("sparse_vector_name"),
        vector_dimension=integer("vector_dimension"),
        distance=text("distance"),
        dense_datatype=value["dense_datatype"],
        sparse_enabled=value["sparse_enabled"],
        sparse_modifier=value["sparse_modifier"],
        sparse_weight=float(sparse_weight),
        sparse_terms=set(sparse_terms),
        id_map=dict(id_map),
        existing_target_ids=set(existing_target_ids),
        source_fingerprint=text("source_fingerprint"),
        metadata_fingerprint=text("metadata_fingerprint"),
        sparse_map_fingerprint=text("sparse_map_fingerprint"),
        acl_incomplete_count=integer("acl_incomplete_count", non_negative=True),
        target_metadata_exists=value["target_metadata_exists"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--source-metadata-collection")
    parser.add_argument("--target-metadata-collection")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dense-vector-name")
    parser.add_argument("--sparse-vector-name")
    parser.add_argument(
        "--sparse-map",
        help="JSON file containing old sparse index -> term (or term -> old index)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="validate and print a read-only plan")
    apply_parser = subparsers.add_parser("apply", help="copy records into the target")
    apply_parser.add_argument(
        "--confirm",
        action="store_true",
        help="required acknowledgement that target Qdrant collections will be written",
    )
    apply_parser.add_argument(
        "--allow-acl-fail-open",
        action="store_true",
        help="acknowledge that records missing ACL fields remain fail-open",
    )
    apply_parser.add_argument(
        "--plan",
        required=True,
        help="reviewed JSON plan produced by the preflight command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.url:
        print("Qdrant URL is required via --url or QDRANT_URL", file=sys.stderr)
        return 2
    try:
        sparse_map = _load_sparse_map(args.sparse_map)
        client = QdrantRestClient(args.url, api_key=args.api_key)
        migration = QdrantMigration(
            client=client,
            source_collection=args.source_collection,
            target_collection=args.target_collection,
            source_metadata_collection=args.source_metadata_collection,
            target_metadata_collection=args.target_metadata_collection,
            batch_size=args.batch_size,
            dense_vector_name=args.dense_vector_name,
            sparse_vector_name=args.sparse_vector_name,
            sparse_map=sparse_map,
        )
        if args.command == "preflight":
            print(json.dumps(migration.preflight().to_dict(), sort_keys=True))
        else:
            reviewed_plan = _load_plan(args.plan)
            print(
                json.dumps(
                    migration.apply(
                        confirm=args.confirm,
                        plan=reviewed_plan,
                        allow_acl_fail_open=args.allow_acl_fail_open,
                    ).to_dict(),
                    sort_keys=True,
                )
            )
    except (MigrationError, QdrantError, ValueError) as exc:
        print(f"qdrant migration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
