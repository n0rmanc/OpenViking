from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from openviking.storage.vectordb.qdrant_sparse import stable_sparse_index
from openviking.storage.vectordb.qdrant_utils import to_qdrant_point_id
from scripts.maintenance.qdrant_migrate import (
    MigrationError,
    QdrantMigration,
    SparseMigrationError,
    _legacy_collection_metadata_id,
    _legacy_index_metadata_id,
    _load_plan,
    _parser,
    _ScanManifest,
    main,
)


class FakeQdrant:
    """Small in-memory REST double that exercises the migration HTTP contract."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.requests: list[tuple[str, str, dict[str, object] | None]] = []
        self.request_params: list[dict[str, object] | None] = []
        self.count_overrides: dict[str, int] = {}

    def add_collection(
        self,
        name: str,
        *,
        vectors: dict[str, object] | object,
        sparse_vectors: dict[str, object] | None = None,
        points: list[dict[str, object]] | None = None,
    ) -> None:
        self.collections[name] = {
            "config": {
                "params": {
                    "vectors": vectors,
                    **({"sparse_vectors": sparse_vectors} if sparse_vectors is not None else {}),
                }
            },
            "points": {str(point["id"]): copy.deepcopy(point) for point in points or []},
            "indexes": {},
            "payload_schema": {},
        }

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append((method, path, copy.deepcopy(body)))
        self.request_params.append(copy.deepcopy(params))
        if method == "GET" and path == "/":
            return {"title": "qdrant", "version": "1.19.1"}
        parts = [unquote(part) for part in urlsplit(path).path.split("/") if part]
        if parts[:1] != ["collections"] or len(parts) < 2:
            raise AssertionError(path)
        name = parts[1]
        collection = self.collections.get(name)
        suffix = parts[2:]

        if method == "GET" and not suffix:
            if collection is None:
                raise _FakeHttpError(404)
            config = copy.deepcopy(collection["config"])
            config["points_count"] = len(collection["points"])
            config["payload_schema"] = copy.deepcopy(collection["payload_schema"])
            config["status"] = "green"
            config["optimizer_status"] = "ok"
            config["update_queue"] = 0
            return {"result": config}

        if method == "PUT" and not suffix:
            if collection is not None:
                raise _FakeHttpError(409)
            self.collections[name] = {
                "config": copy.deepcopy(body or {}),
                "points": {},
                "indexes": {},
                "payload_schema": {},
            }
            return {"result": True}

        if collection is None:
            raise _FakeHttpError(404)

        points: dict[str, dict[str, object]] = collection["points"]
        def filtered_points(request: dict[str, object]) -> list[dict[str, object]]:
            selected = list(points.values())
            point_filter = request.get("filter")
            if not isinstance(point_filter, dict):
                return selected
            must = point_filter.get("must")
            if not isinstance(must, list):
                return selected
            for clause in must:
                if not isinstance(clause, dict) or clause.get("key") != "collection_key":
                    continue
                match = clause.get("match")
                expected = match.get("value") if isinstance(match, dict) else None
                selected = [
                    point
                    for point in selected
                    if isinstance(point.get("payload"), dict)
                    and point["payload"].get("collection_key") == expected
                ]
            return selected
        if method == "DELETE" and not suffix:
            del self.collections[name]
            return {"result": True}

        if suffix == ["points"] and method == "PUT":
            for point in (body or {}).get("points", []):
                points[str(point["id"])] = copy.deepcopy(point)
            return {"result": {"status": "completed"}}

        if suffix == ["points"] and method == "POST":
            result = [
                copy.deepcopy(points[str(point_id)])
                for point_id in (body or {}).get("ids", [])
                if str(point_id) in points
            ]
            return {"result": result}

        if suffix == ["points", "delete"] and method == "POST":
            selector = (body or {}).get("points")
            if not isinstance(selector, list):
                raise AssertionError(body)
            for point_id in selector:
                points.pop(str(point_id), None)
            return {"result": {"status": "completed"}}

        if suffix == ["points", "count"] and method == "POST":
            return {
                "result": {
                    "count": self.count_overrides.get(
                        name,
                        len(filtered_points(body or {})),
                    )
                }
            }

        if suffix == ["points", "scroll"] and method == "POST":
            request = body or {}
            offset = int(request.get("offset") or 0)
            limit = int(request.get("limit") or 1)
            ordered = filtered_points(request)
            page = copy.deepcopy(ordered[offset : offset + limit])
            result: dict[str, object] = {"points": page}
            if offset + len(page) < len(ordered):
                result["next_page_offset"] = offset + len(page)
            return {"result": result}

        if suffix == ["index"] and method == "PUT":
            field_name = str((body or {}).get("field_name"))
            collection["indexes"][field_name] = copy.deepcopy(body)
            collection["payload_schema"][field_name] = {
                "data_type": (body or {}).get("field_schema")
            }
            return {"result": {"status": "completed"}}

        raise AssertionError((method, path, body))


class _FakeHttpError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class _RecordingClient:
    timeout_seconds = 37.0

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, object] | None, dict[str, object] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append((method, path, copy.deepcopy(body), copy.deepcopy(params)))
        if (
            path.count("/") >= 3
            and (
                path.endswith("/points")
                or path.endswith("/points/delete")
                or "/index" in path
            )
        ):
            return {"result": {"status": "completed"}}
        return {"result": True}


class _ReadinessClient:
    def __init__(self, responses: list[dict[str, object]], *, timeout_seconds: float = 1.0):
        self.timeout_seconds = timeout_seconds
        self.responses = list(responses)
        self.requests: list[tuple[str, str, dict[str, object] | None, dict[str, object] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append((method, path, copy.deepcopy(body), copy.deepcopy(params)))
        if not self.responses:
            raise AssertionError("readiness response sequence exhausted")
        return self.responses.pop(0)


def test_migration_target_mutations_use_strong_ordering_and_timeout() -> None:
    client = _RecordingClient()
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="index-data",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="index-data__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=37,
    )

    migration._request(
        "PUT",
        migration._path(migration.target_collection, "/points"),
        {"points": [{"id": "one"}]},
        mutation=True,
    )
    migration._request(
        "POST",
        migration._path(migration.target_collection, "/points/delete"),
        {"points": ["one"]},
        mutation=True,
    )
    migration._request(
        "PUT",
        migration._path(migration.target_collection),
        {"vectors": {"vector": {"size": 2, "distance": "Cosine"}}},
        mutation=True,
    )
    migration._request(
        "PUT",
        migration._path(migration.target_collection, "/index"),
        {"field_name": "account_id", "field_schema": "keyword"},
        mutation=True,
    )
    migration._request(
        "DELETE",
        migration._path(migration.target_collection),
        mutation=True,
    )

    point_params = client.requests[0][3]
    delete_params = client.requests[1][3]
    collection_params = client.requests[2][3]
    index_params = client.requests[3][3]
    collection_delete_params = client.requests[4][3]
    assert point_params == {"wait": "true", "ordering": "strong"}
    assert delete_params == {"wait": "true", "ordering": "strong"}
    assert collection_params == {"timeout": 37}
    assert index_params == {"wait": "true", "timeout": 37}
    assert collection_delete_params == {"timeout": 37}


def test_migration_collection_named_points_keeps_collection_contract() -> None:
    client = _RecordingClient()
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="points",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="points__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=37,
    )

    migration._request(
        "PUT",
        migration._path(migration.target_collection),
        {"vectors": {"vector": {"size": 2, "distance": "Cosine"}}},
        mutation=True,
    )
    migration._request(
        "PUT",
        migration._path(migration.target_collection, "/points"),
        {"points": [{"id": "one"}]},
        mutation=True,
    )

    assert client.requests[0][3] == {"timeout": 37}
    assert client.requests[1][3] == {"wait": "true", "ordering": "strong"}


def test_migration_point_mutation_rejects_acknowledged_result() -> None:
    client = _RecordingClient()
    client.request = lambda *args, **kwargs: {"result": {"status": "acknowledged"}}  # type: ignore[method-assign]
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=37,
    )

    with pytest.raises(MigrationError, match="did not complete"):
        migration._request(
            "PUT",
            migration._path(migration.target_collection, "/points"),
            {"points": [{"id": "one"}]},
            mutation=True,
        )


@pytest.mark.parametrize("response", [{"result": False}, {}])
def test_collection_mutations_require_literal_true_receipts(
    response: dict[str, object],
) -> None:
    client = _RecordingClient()
    client.request = lambda *args, **kwargs: response  # type: ignore[method-assign]
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=37,
    )

    with pytest.raises(MigrationError, match="did not complete"):
        migration._create_collection(
            migration.target_collection,
            {"vectors": {"vector": {"size": 2, "distance": "Cosine"}}},
        )
    with pytest.raises(MigrationError, match="did not complete"):
        migration._delete_collection(migration.target_collection)


def test_migration_rejects_qdrant_versions_below_strong_ordering_floor() -> None:
    client = _ReadinessClient([{"title": "qdrant", "version": "1.9.5"}])
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=1.0,
    )

    with pytest.raises(MigrationError, match="minimum 1.10.0"):
        migration._assert_strong_ordering_support()


@pytest.mark.parametrize("version", ["1.10.0-rc1", "1.10.0-rc1+build.1"])
def test_migration_rejects_qdrant_prerelease_versions(version: str) -> None:
    client = _ReadinessClient([{"title": "qdrant", "version": version}])
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=1.0,
    )

    with pytest.raises(MigrationError, match="unparseable"):
        migration._assert_strong_ordering_support()


def test_migration_accepts_qdrant_build_metadata_on_stable_version() -> None:
    client = _ReadinessClient([{"title": "qdrant", "version": "1.10.0+build.1"}])
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=1.0,
    )

    migration._assert_strong_ordering_support()


def test_migration_readiness_polls_until_green_and_indexes_visible() -> None:
    client = _ReadinessClient(
        [
            {"result": {"status": "yellow", "optimizer_status": "ok"}},
            {
                "result": {
                    "status": "green",
                    "optimizer_status": "ok",
                    "payload_schema": {},
                }
            },
            {
                "result": {
                    "status": "green",
                    "optimizer_status": "ok",
                    "payload_schema": {"account_id": {"data_type": "keyword"}},
                }
            },
        ],
        timeout_seconds=0.2,
    )
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=0.2,
    )

    migration._wait_collection_ready("current", payload_fields={"account_id"})

    assert len(client.requests) == 3


def test_migration_readiness_rejects_red_collection() -> None:
    client = _ReadinessClient(
        [{"result": {"status": "red", "optimizer_status": "ok"}}]
    )
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=1.0,
    )

    with pytest.raises(MigrationError, match="not ready"):
        migration._wait_collection_ready("current")


def test_migration_readiness_times_out_while_collection_is_yellow() -> None:
    client = _ReadinessClient(
        [{"result": {"status": "yellow", "optimizer_status": "ok"}}] * 100,
        timeout_seconds=0.01,
    )
    migration = QdrantMigration(
        client=client,
        source_collection="legacy",
        target_collection="current",
        source_metadata_collection="legacy__meta",
        target_metadata_collection="current__meta",
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=0.01,
    )

    with pytest.raises(MigrationError, match="did not become ready"):
        migration._wait_collection_ready("current")


def _point(
    point_id: object,
    original_id: object,
    *,
    uri: str = "/resources/a.md",
    level: int = 2,
    context_type: str = "resource",
    owner_user_id: str = "alice",
    account_id: str = "acct",
    vector: list[float] | None = None,
    sparse: dict[str, object] | None = None,
) -> dict[str, object]:
    vectors: dict[str, object] = {"vector": vector or [1.0, 0.0]}
    if sparse is not None:
        vectors["sparse_vector"] = sparse
    return {
        "id": point_id,
        "vector": vectors,
        "payload": {
            "_openviking_original_id": original_id,
            "uri": uri,
            "level": level,
            "context_type": context_type,
            "owner_user_id": owner_user_id,
            "account_id": account_id,
            "name": "doc",
        },
    }


def _legacy_fixture(*, sparse: bool = True) -> FakeQdrant:
    qdrant = FakeQdrant()
    vectors: dict[str, object] = {"vector": {"size": 2, "distance": "Cosine"}}
    sparse_vectors = {"sparse_vector": {}} if sparse else None
    fields = [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "uri", "FieldType": "path"},
        {"FieldName": "level", "FieldType": "int64"},
        {"FieldName": "context_type", "FieldType": "string"},
        {"FieldName": "owner_user_id", "FieldType": "string"},
        {"FieldName": "account_id", "FieldType": "string"},
        {"FieldName": "name", "FieldType": "string"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": 2},
    ]
    if sparse:
        fields.append({"FieldName": "sparse_vector", "FieldType": "sparse_vector"})
    source_points = [
        _point(
            1,
            1,
            sparse=(
                {"indices": [111], "values": [0.7]}
                if sparse
                else None
            ),
        ),
        _point(
            "550e8400-e29b-41d4-a716-446655440000",
            "550e8400-e29b-41d4-a716-446655440000",
            uri="/resources/b.md",
            vector=[0.0, 1.0],
            sparse=(
                {"indices": [222], "values": [0.3]}
                if sparse
                else None
            ),
        ),
    ]
    qdrant.add_collection(
        "legacy__context",
        vectors=vectors,
        sparse_vectors=sparse_vectors,
        points=source_points,
    )
    qdrant.add_collection(
        "legacy__context__openviking_meta",
        vectors={"size": 1, "distance": "Cosine"},
        points=[
            {
                "id": _legacy_collection_metadata_id("legacy__context"),
                "vector": [0.0],
                "payload": {
                    "kind": "collection",
                    "collection_key": "legacy__context",
                    "logical_collection_name": "context",
                    "project_name": "legacy",
                    "meta": {
                        "CollectionName": "context",
                        "Fields": fields,
                        "ScalarIndex": ["uri", "level", "context_type", "owner_user_id", "account_id"],
                    },
                },
            },
            {
                "id": _legacy_index_metadata_id("legacy__context", "default"),
                "vector": [0.0],
                "payload": {
                    "kind": "index",
                    "collection_key": "legacy__context",
                    "index_name": "default",
                    "meta": {
                        "IndexName": "default",
                        "VectorIndex": {
                            "IndexType": "hnsw_hybrid",
                            "Distance": "Cosine",
                        },
                        "ScalarIndex": ["uri", "level", "account_id"],
                        "SparseWeight": 0.5,
                    },
                },
            },
        ],
    )
    return qdrant


def _migration(qdrant: FakeQdrant, **kwargs: object) -> QdrantMigration:
    kwargs.setdefault("logical_collection", "legacy/context")
    kwargs.setdefault("migration_id", "mig-1")
    return QdrantMigration(
        client=qdrant,
        source_collection="legacy__context",
        target_collection="current__context",
        source_metadata_collection="legacy__context__openviking_meta",
        target_metadata_collection="current__context__openviking_meta",
        **kwargs,
    )


def _add_current_marker(
    qdrant: FakeQdrant,
    *,
    migration_id: str = "mig-1",
    migration_state: str = "building",
) -> None:
    migration = _migration(qdrant, migration_id=migration_id)
    plan = migration.preflight()
    metadata = migration._legacy_metadata()
    layout = migration._layout(
        migration._collection_info(migration.source_collection),
    )
    marker = migration._marker_payload(
        layout=layout,
        metadata=metadata,
        sparse_weight=plan.sparse_weight,
        source_fingerprint=plan.source_fingerprint,
        metadata_fingerprint=plan.metadata_fingerprint,
        sparse_map_fingerprint=plan.sparse_map_fingerprint,
        setup_complete=migration_state in {"ready", "cutting_over", "active", "retained"},
        acl_incomplete_count=plan.acl_incomplete_count,
        sparse_term_count=plan.sparse_term_count,
        sparse_term_fingerprint=plan.sparse_term_fingerprint,
        migration_state=migration_state,
        source_count=plan.source_count,
        target_count=0,
    )
    marker["migration_id"] = migration_id
    qdrant.add_collection(
        migration.target_metadata_collection,
        vectors={"meta": {"size": 1, "distance": "Dot"}},
        points=[
            {
                "id": to_qdrant_point_id("openviking:metadata"),
                "vector": {"meta": [0.0]},
                "payload": marker,
            }
        ],
    )


def _mark_current_target_building(
    qdrant: FakeQdrant,
    migration: QdrantMigration,
) -> None:
    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["migration_state"] = "building"
    marker["setup_complete"] = False
    # Model an explicit ready-to-building reconciliation window, not an
    # interrupted apply whose durable backfill progress must be preserved.
    marker["last_source_cursor"] = None
    marker["backfill_complete"] = False


def test_preflight_plan_is_compact_and_binds_identity() -> None:
    plan = _migration(
        _legacy_fixture(sparse=False),
        logical_collection="legacy/context",
        migration_id="mig-1",
        timeout_seconds=23,
    ).preflight()

    value = plan.to_dict()

    assert value["logical_collection"] == "legacy/context"
    assert value["migration_id"] == "mig-1"
    assert value["timeout_seconds"] == 23.0
    assert value["target_absent"] is True
    assert value["target_state"] is None
    assert "id_map" not in value
    assert "existing_target_ids" not in value
    assert "sparse_terms" not in value
    assert "url" not in value
    assert "api_key" not in value


def test_foreign_target_marker_is_rejected() -> None:
    qdrant = _legacy_fixture(sparse=False)
    _add_current_marker(qdrant, migration_id="other")

    with pytest.raises(MigrationError, match="migration ID"):
        _migration(qdrant, migration_id="mig-1").preflight()


def test_state_transition_preserves_setup_gate() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _add_current_marker(qdrant)

    assert migration._transition("building")["setup_complete"] is False
    assert migration._transition("ready")["setup_complete"] is True


def test_state_transition_rejects_tampered_marker_layout() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _add_current_marker(qdrant)
    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["vector_dim"] = 999

    with pytest.raises(MigrationError, match="dimension"):
        migration._transition("ready")


def test_preflight_rejects_complete_marker_without_target_collection() -> None:
    qdrant = _legacy_fixture(sparse=False)
    _add_current_marker(qdrant, migration_state="ready")

    with pytest.raises(MigrationError, match="target collection"):
        _migration(qdrant).preflight()


def test_cli_phase_arguments_require_identity_and_timeout() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--url",
                "http://qdrant.invalid",
                "--source-collection",
                "legacy__context",
                "--target-collection",
                "current__context",
                "preflight",
            ]
        )

    args = _parser().parse_args(
        [
            "--url",
            "http://qdrant.invalid",
            "--source-collection",
            "legacy__context",
            "--target-collection",
            "current__context",
            "--logical-collection",
            "legacy/context",
            "--migration-id",
            "mig-1",
            "--timeout-seconds",
            "23",
            "preflight",
        ]
    )
    assert args.logical_collection == "legacy/context"
    assert args.migration_id == "mig-1"
    assert args.timeout_seconds == 23.0

    prepare_args = _parser().parse_args(
        [
            "--url",
            "http://qdrant.invalid",
            "--source-collection",
            "legacy__context",
            "--target-collection",
            "current__context",
            "--logical-collection",
            "legacy/context",
            "--migration-id",
            "mig-1",
            "--timeout-seconds",
            "23",
            "prepare",
            "--plan",
            "plan.json",
            "--confirm",
            "--lock-held",
        ]
    )
    assert prepare_args.command == "prepare"
    assert prepare_args.plan == "plan.json"
    assert prepare_args.lock_held is True


def test_cli_apply_requires_a_reviewed_plan() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--url",
                "http://qdrant.invalid",
                "--source-collection",
                "legacy__context",
                "--target-collection",
                "current__context",
                "--logical-collection",
                "legacy/context",
                "--migration-id",
                "mig-1",
                "--timeout-seconds",
                "23",
                "apply",
                "--confirm",
            ]
        )


def test_prepare_creates_both_collections_before_marker_write() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()

    result = migration.prepare(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    marker_path = migration._path(migration.target_metadata_collection, "/points")
    marker_index = next(
        index
        for index, (method, path, body) in enumerate(qdrant.requests)
        if method == "PUT"
        and path == marker_path
        and body
        and body["points"][0]["id"] == to_qdrant_point_id("openviking:metadata")
    )
    data_create_index = next(
        index
        for index, (method, path, _body) in enumerate(qdrant.requests)
        if method == "PUT" and path == migration._path(migration.target_collection)
    )
    metadata_create_index = next(
        index
        for index, (method, path, _body) in enumerate(qdrant.requests)
        if method == "PUT" and path == migration._path(migration.target_metadata_collection)
    )

    assert data_create_index < marker_index
    assert metadata_create_index < marker_index
    assert result["migration_state"] == "building"
    assert result["setup_complete"] is False


def test_prepare_rejects_source_target_name_collision() -> None:
    qdrant = _legacy_fixture(sparse=False)

    with pytest.raises(ValueError, match="must differ"):
        QdrantMigration(
            client=qdrant,
            source_collection="legacy__context",
            target_collection="legacy__context",
            source_metadata_collection="legacy__context__openviking_meta",
            target_metadata_collection="current__context__openviking_meta",
            logical_collection="legacy/context",
            migration_id="mig-1",
        )


def test_prepare_rejects_foreign_marker_and_shared_metadata_sidecar() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    _add_current_marker(qdrant, migration_id="other")

    with pytest.raises(MigrationError, match="migration ID"):
        migration.prepare(confirm=True, plan=plan, lock_held=True)

    with pytest.raises(ValueError, match="pairwise distinct"):
        QdrantMigration(
            client=qdrant,
            source_collection="legacy__context",
            target_collection="current__context",
            source_metadata_collection="legacy__context__openviking_meta",
            target_metadata_collection="legacy__context__openviking_meta",
            logical_collection="legacy/context",
            migration_id="mig-1",
        )


def test_pre_marker_orphan_cleanup_requires_target_absent_review_and_confirm() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    qdrant.add_collection(
        migration.target_collection,
        vectors={"vector": {"size": 2, "distance": "Cosine"}},
    )
    qdrant.add_collection(
        migration.target_metadata_collection,
        vectors={"meta": {"size": 1, "distance": "Dot"}},
    )

    with pytest.raises(MigrationError, match="confirm"):
        migration._cleanup_pre_marker_orphan(
            reviewed_plan=plan,
            confirm=False,
            lock_held=True,
        )
    with pytest.raises(MigrationError, match="lock"):
        migration._cleanup_pre_marker_orphan(
            reviewed_plan=plan,
            confirm=True,
            lock_held=False,
        )
    wrong_plan = copy.copy(plan)
    wrong_plan.migration_id = "other"
    with pytest.raises(MigrationError, match="migration_id"):
        migration._cleanup_pre_marker_orphan(
            reviewed_plan=wrong_plan,
            confirm=True,
            lock_held=True,
        )

    migration._cleanup_pre_marker_orphan(
        reviewed_plan=plan,
        confirm=True,
        lock_held=True,
    )

    assert migration.target_collection not in qdrant.collections
    assert migration.target_metadata_collection not in qdrant.collections


def test_apply_checks_frozen_source_before_pre_marker_orphan_cleanup() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    layout = migration._layout(
        migration._collection_info(migration.source_collection),
    )
    target_body = migration._target_collection_body(layout)
    qdrant.add_collection(
        migration.target_collection,
        vectors=target_body["vectors"],
    )
    qdrant.add_collection(
        migration.target_metadata_collection,
        vectors={"meta": {"size": 1, "distance": "Dot"}},
    )
    qdrant.collections[migration.source_collection]["points"]["1"]["payload"][
        "name"
    ] = "changed-after-review"

    with pytest.raises(MigrationError, match="stale|source changed"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    assert migration.target_collection in qdrant.collections
    assert migration.target_metadata_collection in qdrant.collections


def test_prepare_race_re_reads_409_and_accepts_only_same_migration_marker() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    original_request = qdrant.request
    raced = False

    def race_on_data_create(method, path, body=None, *, params=None):
        nonlocal raced
        if (
            not raced
            and method == "PUT"
            and path == migration._path(migration.target_collection)
        ):
            raced = True
            original_request(method, path, body, params=params)
            metadata = migration._legacy_metadata()
            layout = migration._layout(
                migration._collection_info(migration.source_collection),
            )
            qdrant.add_collection(
                migration.target_metadata_collection,
                vectors={"meta": {"size": 1, "distance": "Dot"}},
                points=[
                    {
                        "id": to_qdrant_point_id("openviking:metadata"),
                        "vector": {"meta": [0.0]},
                        "payload": migration._marker_payload(
                            layout=layout,
                            metadata=metadata,
                            sparse_weight=plan.sparse_weight,
                            source_fingerprint=plan.source_fingerprint,
                            metadata_fingerprint=plan.metadata_fingerprint,
                            sparse_map_fingerprint=plan.sparse_map_fingerprint,
                            setup_complete=False,
                            acl_incomplete_count=plan.acl_incomplete_count,
                            sparse_term_count=plan.sparse_term_count,
                            sparse_term_fingerprint=plan.sparse_term_fingerprint,
                            source_count=plan.source_count,
                            target_count=0,
                        ),
                    }
                ],
            )
            raise _FakeHttpError(409)
        return original_request(method, path, body, params=params)

    qdrant.request = race_on_data_create  # type: ignore[method-assign]

    result = migration.prepare(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result["migration_id"] == "mig-1"
    assert result["migration_state"] == "building"


def test_prepare_rejects_rolled_back_creation_race() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()

    original_request = qdrant.request
    raced = False

    def race_to_rolled_back_marker(method, path, body=None, *, params=None):
        nonlocal raced
        if (
            not raced
            and method == "PUT"
            and path == migration._path(migration.target_collection)
        ):
            raced = True
            original_request(method, path, body, params=params)
            metadata = migration._legacy_metadata()
            layout = migration._layout(
                migration._collection_info(migration.source_collection),
            )
            qdrant.add_collection(
                migration.target_metadata_collection,
                vectors={"meta": {"size": 1, "distance": "Dot"}},
                points=[
                    {
                        "id": to_qdrant_point_id("openviking:metadata"),
                        "vector": {"meta": [0.0]},
                        "payload": migration._marker_payload(
                            layout=layout,
                            metadata=metadata,
                            sparse_weight=plan.sparse_weight,
                            source_fingerprint=plan.source_fingerprint,
                            metadata_fingerprint=plan.metadata_fingerprint,
                            sparse_map_fingerprint=plan.sparse_map_fingerprint,
                            setup_complete=False,
                            migration_state="rolled_back",
                            acl_incomplete_count=plan.acl_incomplete_count,
                            sparse_term_count=plan.sparse_term_count,
                            sparse_term_fingerprint=plan.sparse_term_fingerprint,
                            source_count=plan.source_count,
                            target_count=0,
                        ),
                    }
                ],
            )
            raise _FakeHttpError(409)
        return original_request(method, path, body, params=params)

    qdrant.request = race_to_rolled_back_marker  # type: ignore[method-assign]

    with pytest.raises(MigrationError, match="rolled_back"):
        migration.prepare(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )
    assert raced is True
    assert migration.target_collection in qdrant.collections
    assert (
        qdrant.collections[migration.target_metadata_collection]["points"][
            to_qdrant_point_id("openviking:metadata")
        ]["payload"]["migration_state"]
        == "rolled_back"
    )


def test_sparse_dictionary_write_is_chunked_and_verified() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(
        qdrant,
        sparse_map={111: "hello", 222: "world"},
        batch_size=1,
    )
    plan = migration.preflight()

    migration.prepare(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    dictionary_writes = [
        body["points"]
        for method, path, body in qdrant.requests
        if method == "PUT"
        and path == migration._path(migration.target_metadata_collection, "/points")
        and body
        and any(
            point.get("payload", {}).get("_openviking_sparse_term") is True
            for point in body["points"]
        )
    ]
    assert [len(points) for points in dictionary_writes] == [1, 1]
    assert len(
        [
            point
            for point in qdrant.collections[migration.target_metadata_collection]["points"].values()
            if point.get("payload", {}).get("_openviking_sparse_term") is True
        ]
    ) == 2


def _apply(migration: QdrantMigration, **kwargs: object):
    kwargs.setdefault("plan", migration.preflight())
    kwargs.setdefault("lock_held", True)
    return migration.apply(**kwargs)


def test_preflight_aggregates_metadata_and_remaps_ids_without_writes() -> None:
    qdrant = _legacy_fixture()
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})

    plan = migration.preflight()

    assert plan.source_count == 2
    assert plan.dense_vector_name == "vector"
    assert plan.vector_dimension == 2
    assert plan.sparse_term_count == 2
    assert plan.target_absent is True
    assert all(method in {"GET", "POST"} for method, _, _ in qdrant.requests)


@pytest.mark.parametrize("kind", ["collection", "index"])
def test_legacy_metadata_point_ids_must_match_deterministic_encoding(kind: str) -> None:
    qdrant = _legacy_fixture(sparse=False)
    expected_id = (
        _legacy_collection_metadata_id("legacy__context")
        if kind == "collection"
        else _legacy_index_metadata_id("legacy__context", "default")
    )
    point = qdrant.collections["legacy__context__openviking_meta"]["points"].pop(expected_id)
    point["id"] = "replaced-metadata-id"
    qdrant.collections["legacy__context__openviking_meta"]["points"][
        "replaced-metadata-id"
    ] = point

    with pytest.raises(MigrationError, match="deterministic encoding"):
        _migration(qdrant).preflight()


@pytest.mark.parametrize("kind", [None, "unexpected"])
def test_legacy_metadata_unknown_kind_fails_closed(kind: str | None) -> None:
    qdrant = _legacy_fixture(sparse=False)
    point = {
        "id": "unexpected-metadata",
        "vector": [0.0],
        "payload": {
            "collection_key": "legacy__context",
            **({"kind": kind} if kind is not None else {}),
        },
    }
    qdrant.collections["legacy__context__openviking_meta"]["points"][
        point["id"]
    ] = point

    with pytest.raises(MigrationError, match="unknown kind"):
        _migration(qdrant).preflight()


def test_apply_creates_current_marker_indexes_and_data_but_never_changes_source() -> None:
    qdrant = _legacy_fixture()
    source_before = copy.deepcopy(qdrant.collections["legacy__context"])
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"}, batch_size=1)

    result = _apply(migration, confirm=True, allow_acl_fail_open=True)

    assert result.migrated_count == 2
    assert result.skipped_count == 0
    assert qdrant.collections["legacy__context"] == source_before
    target = qdrant.collections["current__context"]
    target_meta = qdrant.collections["current__context__openviking_meta"]
    assert len(target["points"]) == 2
    marker = target_meta["points"][to_qdrant_point_id("openviking:metadata")]
    assert marker["payload"]["_openviking_meta_version"] == 1
    assert marker["payload"]["collection_name"] == "current__context"
    assert "default" in marker["payload"]["indexes"]
    assert result.target_count == 2


def test_apply_reconciles_existing_target_records_from_source() -> None:
    qdrant = _legacy_fixture()
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    target = qdrant.collections["current__context"]["points"]
    existing_id = to_qdrant_point_id("1")
    target[existing_id]["payload"]["name"] = "newer-target-value"
    _mark_current_target_building(qdrant, migration)
    data_writes_before = len(
        [
            request
            for request in qdrant.requests
            if request[0] == "PUT" and request[1].endswith("/current__context/points")
        ]
    )

    result = _apply(migration, confirm=True, allow_acl_fail_open=True)

    assert result.migrated_count == 1
    assert result.skipped_count == 1
    assert target[existing_id]["payload"]["name"] == "doc"
    assert (
        len(
            [
                request
                for request in qdrant.requests
                if request[0] == "PUT" and request[1].endswith("/current__context/points")
            ]
        )
        == data_writes_before + 1
    )


def test_reviewed_plan_can_be_reused_after_target_creation() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()

    first = _apply(migration, confirm=True, plan=plan, allow_acl_fail_open=True)
    _mark_current_target_building(qdrant, migration)
    second = _apply(migration, confirm=True, plan=plan, allow_acl_fail_open=True)

    assert first.migrated_count == 2
    assert second.migrated_count == 0
    assert second.skipped_count == 2


def test_incomplete_marker_resumes_after_data_setup_crash(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    original_create_collection = migration._create_collection

    def fail_data_collection(name, body):
        if name == migration.target_collection:
            raise RuntimeError("simulated setup crash")
        return original_create_collection(name, body)

    monkeypatch.setattr(migration, "_create_collection", fail_data_collection)
    with pytest.raises(RuntimeError, match="simulated setup crash"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    assert migration.target_metadata_collection not in qdrant.collections
    assert migration.target_collection not in qdrant.collections

    monkeypatch.setattr(migration, "_create_collection", original_create_collection)
    result = migration.apply(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result.migrated_count == 2
    assert result.target_count == 2


def test_resume_reconciles_newer_target_vectors() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    existing_id = to_qdrant_point_id("1")
    qdrant.collections["current__context"]["points"][existing_id]["vector"]["vector"] = [
        9.0,
        9.0,
    ]
    _mark_current_target_building(qdrant, migration)

    result = _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    assert result.migrated_count == 1
    assert result.skipped_count == 1
    assert qdrant.collections["current__context"]["points"][existing_id]["vector"]["vector"] == [
        1.0,
        0.0,
    ]


def test_resume_repairs_malformed_existing_target_vectors() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    existing_id = to_qdrant_point_id("1")
    qdrant.collections["current__context"]["points"][existing_id]["vector"].pop("vector")
    _mark_current_target_building(qdrant, migration)

    result = _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    assert result.migrated_count == 1
    assert (
        qdrant.collections["current__context"]["points"][existing_id]["vector"]["vector"]
        == [1.0, 0.0]
    )


def test_resume_repairs_sparse_indexes_missing_from_dictionary() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    existing_id = to_qdrant_point_id("1")
    qdrant.collections["current__context"]["points"][existing_id]["vector"][
        "sparse_vector"
    ]["indices"] = [7]
    _mark_current_target_building(qdrant, migration)

    result = _apply(
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}),
        confirm=True,
        allow_acl_fail_open=True,
    )

    assert result.migrated_count == 1
    assert (
        qdrant.collections["current__context"]["points"][existing_id]["vector"][
            "sparse_vector"
        ]["indices"]
        == [stable_sparse_index("hello")]
    )


def test_resume_repairs_missing_acl_on_existing_complete_target() -> None:
    qdrant = _legacy_fixture(sparse=False)
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["payload"].update(
            {
                "acl_enabled": False,
                "acl_direct_grants": [],
                "acl_inherited_grants": [],
            }
        )
    migration = _migration(qdrant)
    _apply(migration, confirm=True)
    existing_id = to_qdrant_point_id("1")
    qdrant.collections["current__context"]["points"][existing_id]["payload"].pop(
        "acl_enabled"
    )
    _mark_current_target_building(qdrant, migration)

    result = _apply(_migration(qdrant), confirm=True)

    assert result.migrated_count == 1
    assert "acl_enabled" in qdrant.collections["current__context"]["points"][
        existing_id
    ]["payload"]


def test_missing_original_id_fails_before_target_creation() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"]["1"]["payload"].pop(
        "_openviking_original_id"
    )

    with pytest.raises(MigrationError, match="original id"):
        _migration(qdrant).preflight()

    assert "current__context" not in qdrant.collections


def test_numeric_and_string_ids_that_map_to_one_point_fail_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"][to_qdrant_point_id("1")] = _point(
        to_qdrant_point_id("1"),
        "1",
        uri="/resources/c.md",
    )

    with pytest.raises(MigrationError, match="collision"):
        _migration(qdrant).preflight()


def test_duplicate_logical_source_ids_fail_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"][to_qdrant_point_id("1")] = _point(
        to_qdrant_point_id("1"),
        "1",
        uri="/resources/c.md",
    )

    with pytest.raises(MigrationError, match="collision"):
        _migration(qdrant).preflight()


def test_source_count_must_match_pagination() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.count_overrides["legacy__context"] = 3

    with pytest.raises(MigrationError, match="source count"):
        _migration(qdrant).preflight()


def _prepare_backfill(
    qdrant: FakeQdrant,
    *,
    batch_size: int = 1,
) -> tuple[QdrantMigration, object]:
    migration = _migration(qdrant, batch_size=batch_size)
    plan = migration.preflight()
    migration.prepare(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    return migration, plan


def _prepare_reconcile(
    qdrant: FakeQdrant,
    *,
    batch_size: int = 1,
) -> tuple[QdrantMigration, object]:
    migration, plan = _prepare_backfill(qdrant, batch_size=batch_size)
    migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    return migration, plan


def test_reconcile_upserts_source_payload_and_vector_changes() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    source = qdrant.collections["legacy__context"]["points"]["1"]
    source["payload"]["name"] = "changed"
    source["vector"]["vector"] = [0.0, 1.0]

    result = migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    target = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]
    assert target["payload"]["name"] == "changed"
    assert target["vector"]["vector"] == [0.0, 1.0]
    assert result["source_count"] == 2
    assert result["migration_state"] == "building"
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["migration_state"] == "building"
    assert marker["setup_complete"] is False


def test_reconcile_deletes_target_extras() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    extra = _point(to_qdrant_point_id("extra"), "extra", uri="/resources/extra.md")
    qdrant.collections["current__context"]["points"][extra["id"]] = extra

    migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert extra["id"] not in qdrant.collections["current__context"]["points"]


def test_reconcile_uses_sqlite_manifest_not_an_unbounded_id_set(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    paths: list[str] = []
    lookups: list[str] = []

    class TrackingScanManifest(_ScanManifest):
        def __init__(self) -> None:
            super().__init__()
            paths.append(self._path)

        def has_source_target(self, target_id: str) -> bool:
            lookups.append(target_id)
            return super().has_source_target(target_id)

    monkeypatch.setattr(
        "scripts.maintenance.qdrant_migrate._ScanManifest",
        TrackingScanManifest,
    )
    migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert lookups
    assert paths
    assert all(not Path(path).exists() for path in paths)


def test_reconcile_requires_external_lock_even_with_barrier() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["migration_state"] = "cutting_over"
    marker["setup_complete"] = True
    before_points = copy.deepcopy(qdrant.collections["current__context"]["points"])
    before_requests = len(qdrant.requests)

    with pytest.raises(MigrationError, match="lock"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            barrier_held=True,
            allow_acl_fail_open=True,
            lock_held=False,
        )

    assert qdrant.collections["current__context"]["points"] == before_points
    assert len(qdrant.requests) == before_requests


def test_reconcile_rechecks_fingerprint_candidates_with_direct_payload_vector_compare(
    monkeypatch,
) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    target = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]
    target["payload"]["name"] = "stale"
    monkeypatch.setattr(
        "scripts.maintenance.qdrant_migrate._point_fingerprint",
        lambda **_kwargs: "same",
    )

    migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert (
        qdrant.collections["current__context"]["points"][
            to_qdrant_point_id("1")
        ]["payload"]["name"]
        == "doc"
    )


def test_reconcile_compares_canonical_float32_vector_values() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"]["1"]["vector"]["vector"] = [
        0.1,
        0.2,
    ]
    migration, plan = _prepare_reconcile(qdrant)
    target = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]
    target["vector"]["vector"] = [0.10000000149011612, 0.20000000298023224]
    result = migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result["migrated_count"] == 0
    assert target["vector"]["vector"] == [0.10000000149011612, 0.20000000298023224]


def test_reconcile_fails_closed_on_metadata_or_sparse_map_drift(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    original = migration._upsert_target_batch

    def mutate_map(points):
        migration._sparse_map[7] = "changed"
        return original(points)

    monkeypatch.setattr(migration, "_upsert_target_batch", mutate_map)
    with pytest.raises(MigrationError, match="sparse map"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["migration_state"] == "failed"


def test_reconcile_fails_closed_on_metadata_drift(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    original = migration._reconcile_round
    rounds = 0

    def mutate_metadata_after_round(*, layout, schema, metadata, state):
        nonlocal rounds
        snapshot = original(
            layout=layout,
            schema=schema,
            metadata=metadata,
            state=state,
        )
        rounds += 1
        if rounds == 1:
            metadata_point = qdrant.collections[
                "legacy__context__openviking_meta"
            ]["points"][_legacy_collection_metadata_id("legacy__context")]
            metadata_point["payload"]["meta"]["CollectionName"] = "changed"
        return snapshot

    monkeypatch.setattr(migration, "_reconcile_round", mutate_metadata_after_round)
    with pytest.raises(MigrationError, match="metadata"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["migration_state"] == "failed"


def test_reconcile_fails_after_three_non_converging_rounds(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    original = migration._reconcile_round
    calls = 0

    def changing_source(*, layout, schema, metadata, state):
        nonlocal calls
        calls += 1
        snapshot = original(
            layout=layout,
            schema=schema,
            metadata=metadata,
            state=state,
        )
        source = qdrant.collections["legacy__context"]["points"]["1"]
        source["payload"]["name"] = f"changed-{calls}"
        return snapshot

    monkeypatch.setattr(migration, "_reconcile_round", changing_source)
    with pytest.raises(MigrationError, match="round 3"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )
    assert calls == 3


def test_reconcile_publishes_only_final_stable_source_snapshot(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    original = migration._reconcile_round
    snapshots = []

    def mutate_once(*, layout, schema, metadata, state):
        snapshot = original(
            layout=layout,
            schema=schema,
            metadata=metadata,
            state=state,
        )
        snapshots.append(snapshot)
        if len(snapshots) == 1:
            source = qdrant.collections["legacy__context"]["points"]["1"]
            source["payload"]["name"] = "final"
        return snapshot

    monkeypatch.setattr(migration, "_reconcile_round", mutate_once)
    result = migration.reconcile(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert len(snapshots) == 3
    assert snapshots[0].fingerprint != snapshots[-1].fingerprint
    assert marker["source_fingerprint"] == snapshots[-1].fingerprint
    assert marker["migration_state"] == "building"
    assert result["rounds"] == 3
    assert (
        qdrant.collections["current__context"]["points"][
            to_qdrant_point_id("1")
        ]["payload"]["name"]
        == "final"
    )


@pytest.mark.parametrize("tamper", ["foreign", "ready"])
def test_reconcile_rechecks_final_marker_before_publication(
    monkeypatch,
    tamper: str,
) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    original_round = migration._reconcile_round
    original_load = migration._load_current_marker
    rounds = 0
    tampered = False
    writes_at_tamper: list[int] = []

    def mutation_count() -> int:
        return sum(
            method == "PUT" and path.endswith("/points")
            or method == "POST" and path.endswith("/points/delete")
            for method, path, _body in qdrant.requests
        )

    def track_round(*, layout, schema, metadata, state):
        nonlocal rounds
        snapshot = original_round(
            layout=layout,
            schema=schema,
            metadata=metadata,
            state=state,
        )
        rounds += 1
        return snapshot

    def tamper_final_marker():
        nonlocal tampered
        if rounds >= 2 and not tampered:
            tampered = True
            marker = qdrant.collections[
                "current__context__openviking_meta"
            ]["points"][to_qdrant_point_id("openviking:metadata")]["payload"]
            if tamper == "foreign":
                marker["migration_id"] = "foreign"
            else:
                marker["migration_state"] = "ready"
                marker["setup_complete"] = True
            writes_at_tamper.append(mutation_count())
        return original_load()

    monkeypatch.setattr(migration, "_reconcile_round", track_round)
    monkeypatch.setattr(migration, "_load_current_marker", tamper_final_marker)
    with pytest.raises(MigrationError, match="migration ID|state changed"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert writes_at_tamper
    assert mutation_count() == writes_at_tamper[0]
    if tamper == "foreign":
        assert marker["migration_id"] == "foreign"
    else:
        assert marker["migration_state"] == "ready"
        assert marker["setup_complete"] is True


def test_cutover_reconcile_preserves_cutting_over_and_setup_gate() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["migration_state"] = "cutting_over"
    marker["setup_complete"] = True

    migration.reconcile(
        confirm=True,
        plan=plan,
        barrier_held=True,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert marker["migration_state"] == "cutting_over"
    assert marker["setup_complete"] is True


def test_interrupted_reconcile_rebuilds_and_deletes_manifest(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_reconcile(qdrant)
    paths: list[str] = []

    class TrackingScanManifest(_ScanManifest):
        def __init__(self) -> None:
            super().__init__()
            paths.append(self._path)

    monkeypatch.setattr(
        "scripts.maintenance.qdrant_migrate._ScanManifest",
        TrackingScanManifest,
    )
    monkeypatch.setattr(
        migration,
        "_upsert_target_batch",
        lambda _points: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        migration.reconcile(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )
    assert paths
    assert all(not Path(path).exists() for path in paths)


def test_count_requests_strong_consistency() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)

    assert migration._count(migration.source_collection) == 2

    count_params = [
        params
        for (method, path, _body), params in zip(
            qdrant.requests,
            qdrant.request_params,
            strict=True,
        )
        if method == "POST" and path.endswith("/points/count")
    ]
    assert count_params == [{"consistency": "all"}]


def test_backfill_persists_integer_cursor_after_each_batch() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    written_markers: list[dict[str, object]] = []
    original_write_marker = migration._write_marker

    def record_marker(marker):
        written_markers.append(copy.deepcopy(marker))
        return original_write_marker(marker)

    migration._write_marker = record_marker  # type: ignore[method-assign]

    result = migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result["backfill_complete"] is True
    assert [marker["last_source_cursor"] for marker in written_markers] == [1, None]
    assert written_markers[0]["backfill_complete"] is False
    assert written_markers[-1]["backfill_complete"] is True


def test_backfill_persists_string_cursor_without_coercion(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_points = list(
        qdrant.collections[migration.source_collection]["points"].values()
    )
    original_scroll_page = migration._scroll_page
    cursor = "550e8400-e29b-41d4-a716-446655440001"
    pages = {
        None: ([copy.deepcopy(source_points[0])], cursor),
        cursor: ([copy.deepcopy(source_points[1])], None),
    }

    def scroll_page(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        assert with_vectors is True
        return pages[offset]

    monkeypatch.setattr(migration, "_scroll_page", scroll_page)

    result = migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result["backfill_complete"] is True
    assert result["last_source_cursor"] is None
    assert qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]["last_source_cursor"] is None


def test_backfill_rejects_malformed_or_repeated_cursor(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_point = next(
        iter(qdrant.collections[migration.source_collection]["points"].values())
    )
    original_scroll_page = migration._scroll_page

    def malformed(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        return [copy.deepcopy(source_point)], {"not": "an offset"}

    monkeypatch.setattr(migration, "_scroll_page", malformed)
    with pytest.raises(MigrationError, match="offset"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["last_source_cursor"] = 0
    marker["backfill_complete"] = False

    def repeated(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        assert offset == 0
        return [copy.deepcopy(source_point)], 0

    monkeypatch.setattr(migration, "_scroll_page", repeated)
    with pytest.raises(MigrationError, match="repeated"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )


def test_backfill_rejects_invalid_uuid_cursor(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_point = next(
        iter(qdrant.collections[migration.source_collection]["points"].values())
    )
    original_scroll_page = migration._scroll_page

    def malformed_uuid(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        return [copy.deepcopy(source_point)], "not-a-qdrant-uuid"

    monkeypatch.setattr(migration, "_scroll_page", malformed_uuid)
    with pytest.raises(MigrationError, match="UUID"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )


def test_backfill_rejects_completed_marker_with_cursor() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["backfill_complete"] = True
    marker["last_source_cursor"] = 1

    with pytest.raises(MigrationError, match="completion"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )


def test_failed_batch_can_be_retried_without_source_mutation(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_before = copy.deepcopy(qdrant.collections[migration.source_collection])
    original_write_points = migration._write_points
    failed = False

    def fail_once(collection, points):
        nonlocal failed
        if collection == migration.target_collection and not failed:
            failed = True
            raise MigrationError("target write failed")
        return original_write_points(collection, points)

    monkeypatch.setattr(migration, "_write_points", fail_once)
    with pytest.raises(MigrationError, match="target write failed"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["last_source_cursor"] is None
    assert marker["backfill_complete"] is False
    assert qdrant.collections[migration.source_collection] == source_before

    result = migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    assert result["backfill_complete"] is True


def test_backfill_holds_one_page_and_one_write_batch(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_points = list(
        qdrant.collections[migration.source_collection]["points"].values()
    )
    pages = {
        None: ([copy.deepcopy(source_points[0])], 1),
        1: ([copy.deepcopy(source_points[1])], None),
    }
    original_scroll_page = migration._scroll_page
    page_sizes: list[int] = []
    batch_sizes: list[int] = []
    original_upsert = migration._upsert_target_batch

    def scroll_page(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        page = pages[offset]
        page_sizes.append(len(page[0]))
        return page

    def upsert(points):
        batch_sizes.append(len(points))
        return original_upsert(points)

    monkeypatch.setattr(migration, "_scroll_page", scroll_page)
    monkeypatch.setattr(migration, "_upsert_target_batch", upsert)
    migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert page_sizes == [1, 1]
    assert batch_sizes == [1, 1]


def test_completed_backfill_resume_does_not_scan_source(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    original_scroll_page = migration._scroll_page

    def reject_source_scan(collection, *, offset, with_vectors, filter=None):
        if collection == migration.source_collection:
            raise AssertionError("completed backfill scanned the source")
        return original_scroll_page(
            collection,
            offset=offset,
            with_vectors=with_vectors,
            filter=filter,
        )

    monkeypatch.setattr(migration, "_scroll_page", reject_source_scan)
    result = migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result["backfill_complete"] is True
    assert result["migrated_count"] == 0


def test_marker_failure_after_target_write_retries_same_page(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    original_write_marker = migration._write_marker
    failed = False

    def fail_once(marker):
        nonlocal failed
        if marker["last_source_cursor"] == 1 and not failed:
            failed = True
            raise MigrationError("marker write failed")
        return original_write_marker(marker)

    monkeypatch.setattr(migration, "_write_marker", fail_once)
    with pytest.raises(MigrationError, match="marker write failed"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["last_source_cursor"] is None
    assert marker["backfill_complete"] is False
    assert to_qdrant_point_id("1") in qdrant.collections[
        migration.target_collection
    ]["points"]

    result = migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    assert result["backfill_complete"] is True


def test_cross_page_duplicate_source_point_fails_closed(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration, plan = _prepare_backfill(qdrant, batch_size=1)
    source_point = next(
        iter(qdrant.collections[migration.source_collection]["points"].values())
    )
    original_scroll_page = migration._scroll_page

    def duplicate_page(collection, *, offset, with_vectors, filter=None):
        if collection != migration.source_collection:
            return original_scroll_page(
                collection,
                offset=offset,
                with_vectors=with_vectors,
                filter=filter,
            )
        if offset is None:
            return [copy.deepcopy(source_point)], 1
        return [copy.deepcopy(source_point)], None

    monkeypatch.setattr(migration, "_scroll_page", duplicate_page)
    with pytest.raises(MigrationError, match="duplicate point id"):
        migration.backfill(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["last_source_cursor"] == 1
    assert marker["backfill_complete"] is False


def test_backfill_preserves_canonical_observations_across_batch_boundaries() -> None:
    qdrant = _legacy_fixture(sparse=True)
    second = next(
        point
        for point in qdrant.collections["legacy__context"]["points"].values()
        if point["payload"]["_openviking_original_id"]
        == "550e8400-e29b-41d4-a716-446655440000"
    )
    second["vector"]["sparse_vector"] = {"indices": [111], "values": [0.3]}
    migration = _migration(
        qdrant,
        sparse_map={111: "hello"},
        batch_size=1,
    )
    plan = migration.preflight()
    migration.prepare(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )
    marker_before = copy.deepcopy(
        qdrant.collections[migration.target_metadata_collection]["points"][
            to_qdrant_point_id("openviking:metadata")
        ]["payload"]
    )

    migration.backfill(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    marker_after = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    for field_name in (
        "source_count",
        "source_fingerprint",
        "acl_incomplete_count",
        "sparse_term_count",
        "sparse_term_fingerprint",
    ):
        assert marker_after[field_name] == marker_before[field_name] == plan.to_dict()[
            field_name
        ]


def test_apply_resumes_from_persisted_backfill_cursor(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant, batch_size=1)
    plan = migration.preflight()
    original_upsert = migration._upsert_target_batch
    calls = 0

    def fail_second_batch(points):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MigrationError("simulated failed apply batch")
        return original_upsert(points)

    monkeypatch.setattr(migration, "_upsert_target_batch", fail_second_batch)
    with pytest.raises(MigrationError, match="simulated failed apply batch"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["last_source_cursor"] == 1
    assert marker["backfill_complete"] is False

    monkeypatch.setattr(migration, "_upsert_target_batch", original_upsert)
    result = migration.apply(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result.migrated_count == 1
    assert result.target_count == 2


def test_target_count_must_match_pagination() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    qdrant.count_overrides["current__context"] = 3

    with pytest.raises(MigrationError, match="target count"):
        _migration(qdrant).preflight()


def test_sparse_hash_collisions_fail_closed(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=True)
    monkeypatch.setattr(
        "scripts.maintenance.qdrant_migrate.stable_sparse_index",
        lambda term: 7,
    )

    with pytest.raises(SparseMigrationError, match="collision"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_existing_sparse_dictionary_collisions_fail_closed() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    metadata_points = qdrant.collections["current__context__openviking_meta"]["points"]
    hello_id = to_qdrant_point_id("openviking:sparse:hello")
    metadata_points[hello_id]["payload"]["term"] = "different"

    with pytest.raises(SparseMigrationError, match="collision"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_existing_sparse_dictionary_point_id_collision_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    metadata_points = qdrant.collections["current__context__openviking_meta"]["points"]
    hello_id = to_qdrant_point_id("openviking:sparse:hello")
    metadata_points[hello_id]["payload"].update(
        {"term": "different", "index": 999}
    )

    with pytest.raises(SparseMigrationError, match="point-id"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_sparse_dictionary_write_is_verified_before_completion(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    original_write_points = migration._write_points

    def drop_dictionary_write(collection, points):
        if collection == migration.target_metadata_collection and any(
            point.get("payload", {}).get("_openviking_sparse_term") is True
            for point in points
        ):
            return
        return original_write_points(collection, points)

    monkeypatch.setattr(migration, "_write_points", drop_dictionary_write)

    with pytest.raises(SparseMigrationError, match="missing terms after write"):
        _apply(migration, confirm=True, allow_acl_fail_open=True)

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["setup_complete"] is False


def test_existing_target_allows_newer_schema_fields_and_sparse_policy() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["schema"]["Fields"].append(
        {"FieldName": "acl_enabled", "FieldType": "bool"}
    )
    marker["sparse_weight"] = 0.9

    plan = _migration(qdrant).preflight()

    assert plan.target_absent is False


def test_complete_marker_indexes_must_exist_physically() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["indexes"]["bogus"] = {"ScalarIndex": ["not_physical"]}

    with pytest.raises(MigrationError, match="missing payload index.*not_physical"):
        _migration(qdrant).preflight()


def test_marker_scalar_index_must_have_valid_shape() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["indexes"]["bogus"] = {"ScalarIndex": "not-a-list"}

    with pytest.raises(MigrationError, match="ScalarIndex is malformed"):
        _migration(qdrant).preflight()


def test_marker_index_map_must_match_legacy_metadata() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["indexes"]["extra"] = {"VectorIndex": {"IndexType": "hnsw"}}

    with pytest.raises(MigrationError, match="index map differs.*extra"):
        _migration(qdrant).preflight()


def test_marker_index_metadata_must_match_legacy_metadata() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["indexes"]["default"]["Description"] = "tampered"

    with pytest.raises(MigrationError, match="index 'default' changed"):
        _migration(qdrant).preflight()


@pytest.mark.parametrize("size", [2.5, True, "2"])
def test_dense_vector_size_must_be_a_positive_integer(size: object) -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["config"]["params"]["vectors"]["vector"][
        "size"
    ] = size

    with pytest.raises(MigrationError, match="positive integer"):
        _migration(qdrant).preflight()


def test_apply_rejects_ready_target_before_copy() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["operator_extension"] = {"retention": "audit"}
    marker_before = copy.deepcopy(marker)
    target_before = copy.deepcopy(qdrant.collections["current__context"]["points"])
    data_writes_before = len(
        [
            request
            for request in qdrant.requests
            if request[0] == "PUT" and request[1].endswith("/current__context/points")
        ]
    )

    with pytest.raises(MigrationError, match="ready"):
        _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    assert marker == marker_before
    assert qdrant.collections["current__context"]["points"] == target_before
    assert (
        len(
            [
                request
                for request in qdrant.requests
                if request[0] == "PUT"
                and request[1].endswith("/current__context/points")
            ]
        )
        == data_writes_before
    )


def test_apply_rechecks_marker_state_after_prepare_before_copy(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    original_prepare = migration.prepare

    def prepare_then_ready(**kwargs):
        marker = original_prepare(**kwargs)
        migration._transition("ready")
        return marker

    monkeypatch.setattr(migration, "prepare", prepare_then_ready)

    with pytest.raises(MigrationError, match="ready"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["migration_state"] == "ready"
    assert marker["setup_complete"] is True
    assert qdrant.collections[migration.target_collection]["points"] == {}


def test_existing_target_rejects_changed_source_field_type() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    level = next(field for field in marker["schema"]["Fields"] if field["FieldName"] == "level")
    level["FieldType"] = "string"

    with pytest.raises(MigrationError, match="schema"):
        _migration(qdrant).preflight()


def test_existing_target_requires_a_marker_owned_by_target_collection() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.add_collection(
        "current__context",
        vectors={"vector": {"size": 2, "distance": "Cosine"}},
    )

    with pytest.raises(MigrationError, match="current marker"):
        _migration(qdrant).preflight()


def test_sparse_data_without_authoritative_mapping_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=True)

    with pytest.raises(SparseMigrationError, match="authoritative"):
        _migration(qdrant).preflight()


def test_conflicting_sparse_weights_fail_closed() -> None:
    qdrant = _legacy_fixture(sparse=True)
    secondary_id = _legacy_index_metadata_id("legacy__context", "secondary")
    qdrant.collections["legacy__context__openviking_meta"]["points"][secondary_id] = {
        "id": secondary_id,
        "vector": [0.0],
        "payload": {
            "kind": "index",
            "collection_key": "legacy__context",
            "index_name": "secondary",
            "meta": {
                "IndexName": "secondary",
                "VectorIndex": {"IndexType": "hnsw_hybrid", "Distance": "Cosine"},
                "SparseWeight": 0.7,
            },
        },
    }

    with pytest.raises(MigrationError, match="conflicting sparse weights"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_uri_sidecars_are_recomputed_and_payload_is_not_dropped() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"]["1"]["payload"].update(
        {
            "uri": "viking://resources/nested/a.md",
            "parent_uri": "viking://resources/nested",
            "scope_roots": ["/wrong"],
            "uri_depth": 99,
            "tags": ["keep", "this"],
        }
    )
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    payload = qdrant.collections["current__context"]["points"][to_qdrant_point_id("1")][
        "payload"
    ]

    assert payload["uri"] == "/resources/nested/a.md"
    assert payload["parent_uri"] == "/resources/nested"
    assert payload["uri_depth"] == 3
    assert payload["scope_roots"] == ["/", "/resources", "/resources/nested", "/resources/nested/a.md"]
    assert payload["tags"] == ["keep", "this"]


def test_ownerless_uri_does_not_require_owner_user_id() -> None:
    qdrant = _legacy_fixture(sparse=False)
    payload = qdrant.collections["legacy__context"]["points"]["1"]["payload"]
    payload["uri"] = "/user"
    payload.pop("owner_user_id")

    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)

    target_payload = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]["payload"]
    assert "owner_user_id" not in target_payload
    _mark_current_target_building(qdrant, migration)
    result = _apply(migration, confirm=True, allow_acl_fail_open=True)
    assert result.migrated_count == 0


def test_missing_owner_user_id_is_backfilled_from_user_uri() -> None:
    qdrant = _legacy_fixture(sparse=False)
    payload = qdrant.collections["legacy__context"]["points"]["1"]["payload"]
    payload["uri"] = "/user/alice/memories/a.md"
    payload.pop("owner_user_id")

    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)

    target_payload = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]["payload"]
    assert target_payload["owner_user_id"] == "alice"


def test_owner_user_id_mismatch_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    payload = qdrant.collections["legacy__context"]["points"]["1"]["payload"]
    payload["uri"] = "/user/alice/memories/a.md"
    payload["owner_user_id"] = "bob"

    with pytest.raises(MigrationError, match="owner_user_id"):
        _migration(qdrant).preflight()


def test_owner_normalization_resumes_legacy_target(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    payload = qdrant.collections["legacy__context"]["points"]["1"]["payload"]
    payload["uri"] = "/user/alice/memories/a.md"
    payload.pop("owner_user_id")

    migration = _migration(qdrant)
    monkeypatch.setattr(
        "scripts.maintenance.qdrant_migrate._normalize_owner_user_id",
        lambda payload, *, uri, point_id, source_keys: None,
    )
    monkeypatch.setattr(
        QdrantMigration,
        "_validate_target_payload",
        staticmethod(lambda *args, **kwargs: None),
    )
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    monkeypatch.undo()

    _mark_current_target_building(qdrant, migration)
    plan = migration.preflight()
    result = migration.apply(
        confirm=True,
        plan=plan,
        allow_acl_fail_open=True,
        lock_held=True,
    )

    assert result.migrated_count == 1
    assert qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]["payload"]["owner_user_id"] == "alice"


def test_apply_requires_explicit_confirmation() -> None:
    qdrant = _legacy_fixture(sparse=False)

    with pytest.raises(MigrationError, match="confirm"):
        _migration(qdrant).apply()

    assert "current__context" not in qdrant.collections


def test_apply_requires_external_lock_before_requests() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    qdrant.requests.clear()

    with pytest.raises(MigrationError, match="lock"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
        )

    assert qdrant.requests == []
    assert "current__context" not in qdrant.collections


def test_apply_rejects_cutting_over_target_state() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker["migration_state"] = "cutting_over"
    marker["setup_complete"] = True
    target_before = copy.deepcopy(qdrant.collections[migration.target_collection])

    plan = _migration(qdrant).preflight()
    with pytest.raises(MigrationError, match="cutting_over"):
        _migration(qdrant).apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    assert qdrant.collections[migration.target_collection] == target_before


def test_target_metadata_collection_collision_is_rejected() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.add_collection(
        "current__context__openviking_meta",
        vectors={"meta": {"size": 1, "distance": "Dot"}},
    )

    with pytest.raises(MigrationError, match="metadata collection"):
        _migration(qdrant).preflight()


def test_missing_legacy_index_metadata_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context__openviking_meta"]["points"].pop(
        _legacy_index_metadata_id("legacy__context", "default")
    )

    with pytest.raises(MigrationError, match="no index documents"):
        _migration(qdrant).preflight()


def test_cli_json_plan_is_serializable() -> None:
    qdrant = _legacy_fixture(sparse=False)
    plan = _migration(qdrant).preflight()

    encoded = json.dumps(plan.to_dict(), sort_keys=True)

    assert '"source_count": 2' in encoded


def test_cli_apply_requires_reviewed_plan() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--url",
                "http://qdrant.invalid",
                "--source-collection",
                "legacy__context",
                "--target-collection",
                "current__context",
                "apply",
                "--confirm",
            ]
        )

    assert exc_info.value.code == 2


def test_reviewed_plan_json_round_trips(tmp_path) -> None:
    qdrant = _legacy_fixture(sparse=True)
    plan = _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    loaded = _load_plan(str(path))

    assert loaded is not None
    assert loaded.to_dict() == plan.to_dict()


def test_default_legacy_metadata_collection_is_global() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["__openviking_meta"] = qdrant.collections.pop(
        "legacy__context__openviking_meta"
    )

    migration = QdrantMigration(
        client=qdrant,
        source_collection="legacy__context",
        target_collection="current__context",
        logical_collection="legacy/context",
        migration_id="mig-1",
    )

    assert migration.source_metadata_collection == "__openviking_meta"
    assert migration.preflight().source_count == 2


def test_cli_reports_invalid_sparse_map_without_traceback(tmp_path, capsys) -> None:
    sparse_map = tmp_path / "sparse.json"
    sparse_map.write_text("[]", encoding="utf-8")

    result = main(
        [
            "--url",
            "http://qdrant.invalid",
            "--source-collection",
            "legacy__context",
            "--target-collection",
            "current__context",
            "--logical-collection",
            "legacy/context",
            "--migration-id",
            "mig-1",
            "--timeout-seconds",
            "10",
            "--sparse-map",
            str(sparse_map),
            "preflight",
        ]
    )

    assert result == 2
    assert "qdrant migration failed" in capsys.readouterr().err


def test_acl_incomplete_records_require_explicit_acknowledgement() -> None:
    qdrant = _legacy_fixture(sparse=False)

    with pytest.raises(MigrationError, match="ACL"):
        _apply(_migration(qdrant), confirm=True)

    assert "current__context" not in qdrant.collections


def test_source_mutation_between_preflight_and_apply_is_rejected(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    original_scan = migration._scan_source
    calls = 0

    def scan_source(*, layout, schema, manifest=None, point_callback=None):
        nonlocal calls
        snapshot = original_scan(
            layout=layout,
            schema=schema,
            manifest=manifest,
            point_callback=point_callback,
        )
        calls += 1
        if calls == 1:
            points = qdrant.collections["legacy__context"]["points"]
            points.pop("1")
            points["3"] = _point(3, 3, uri="/resources/c.md")
        return snapshot

    monkeypatch.setattr(migration, "_scan_source", scan_source)

    with pytest.raises(MigrationError, match="source changed"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    assert "current__context" not in qdrant.collections


def test_apply_rejects_a_stale_plan() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    plan = migration.preflight()
    qdrant.collections["legacy__context"]["points"]["1"]["payload"]["name"] = "changed"

    with pytest.raises(MigrationError, match="stale"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )

    assert "current__context" not in qdrant.collections


def test_incomplete_marker_is_rejected_before_copy() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    marker.pop("setup_complete")

    with pytest.raises(MigrationError, match="required fields"):
        _migration(qdrant).preflight()


def test_incompatible_target_metadata_layout_is_rejected() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    qdrant.collections["current__context__openviking_meta"]["config"]["vectors"] = {
        "meta": {"size": 2, "distance": "Dot"}
    }

    with pytest.raises(MigrationError, match="incompatible vector layout"):
        _migration(qdrant).preflight()


def test_source_vector_datatype_and_sparse_modifier_are_preserved() -> None:
    qdrant = _legacy_fixture(sparse=True)
    source_params = qdrant.collections["legacy__context"]["config"]["params"]
    source_params["vectors"]["vector"]["datatype"] = "float16"
    source_params["sparse_vectors"]["sparse_vector"]["modifier"] = "idf"

    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    plan = migration.preflight()

    assert plan.dense_datatype == "float16"
    assert plan.sparse_modifier == "idf"
    _apply(migration, confirm=True, allow_acl_fail_open=True)

    target_params = qdrant.collections["current__context"]["config"]["vectors"]
    assert target_params["vector"]["datatype"] == "float16"
    assert (
        qdrant.collections["current__context"]["config"]["sparse_vectors"][
            "sparse_vector"
        ]["modifier"]
        == "idf"
    )


def test_partial_index_setup_is_repaired_on_resume(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    original_write_indexes = migration._write_indexes
    failed = False

    def fail_once(schema, indexes):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected index failure")
        return original_write_indexes(schema, indexes)

    monkeypatch.setattr(migration, "_write_indexes", fail_once)
    with pytest.raises(RuntimeError, match="index failure"):
        _apply(migration, confirm=True, allow_acl_fail_open=True)

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["setup_complete"] is False
    assert qdrant.collections["current__context"]["indexes"] == {}

    result = _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    assert result.target_count == 2
    assert qdrant.collections["current__context"]["indexes"]
    resumed_marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert resumed_marker["setup_complete"] is True


def test_existing_target_records_absent_from_source_remain_for_reconcile() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    _mark_current_target_building(qdrant, migration)
    extra_id = to_qdrant_point_id("extra")
    qdrant.collections["current__context"]["points"][extra_id] = _point(
        extra_id,
        "extra",
        uri="/resources/extra.md",
    )

    plan = _migration(qdrant).preflight()

    assert plan.target_absent is False
    with pytest.raises(MigrationError, match="extras"):
        _migration(qdrant).apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )


def test_existing_target_extra_requires_deterministic_original_id() -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    qdrant.collections["current__context"]["points"]["extra"] = _point(
        "extra",
        "extra",
        uri="/resources/extra.md",
    )

    with pytest.raises(MigrationError, match="deterministic"):
        _migration(qdrant).preflight()


def test_sparse_only_source_record_is_preserved() -> None:
    qdrant = _legacy_fixture(sparse=True)
    qdrant.collections["legacy__context"]["points"]["1"]["vector"].pop("vector")
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})

    _apply(migration, confirm=True, allow_acl_fail_open=True)

    target_point = qdrant.collections["current__context"]["points"][
        to_qdrant_point_id("1")
    ]
    assert "vector" not in target_point["vector"]
    assert target_point["vector"]["sparse_vector"]["indices"]


def test_multiple_named_sparse_vectors_require_selection() -> None:
    qdrant = _legacy_fixture(sparse=True)
    qdrant.collections["legacy__context"]["config"]["params"]["sparse_vectors"] = {
        "sparse_vector": {},
        "other_sparse": {},
    }

    with pytest.raises(SparseMigrationError, match="multiple named sparse"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_unsupported_manhattan_distance_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["config"]["params"]["vectors"]["vector"][
        "distance"
    ] = "Manhattan"

    with pytest.raises(MigrationError, match="unsupported Qdrant distance"):
        _migration(qdrant).preflight()


def test_malformed_legacy_schema_fields_fail_during_preflight() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context__openviking_meta"]["points"][
        _legacy_collection_metadata_id("legacy__context")
    ][
        "payload"
    ]["meta"]["Fields"] = ["malformed"]

    with pytest.raises(MigrationError, match="malformed field"):
        _migration(qdrant).preflight()


def test_malformed_legacy_index_metadata_fails_during_preflight() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context__openviking_meta"]["points"][
        _legacy_index_metadata_id("legacy__context", "default")
    ][
        "payload"
    ]["meta"]["ScalarIndex"] = "malformed"

    with pytest.raises(MigrationError, match="ScalarIndex is malformed"):
        _migration(qdrant).preflight()


def test_metadata_count_must_match_filtered_pagination() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.count_overrides["legacy__context__openviking_meta"] = 3

    with pytest.raises(MigrationError, match="metadata count"):
        _migration(qdrant).preflight()


def test_acl_fields_must_have_valid_types_and_grants() -> None:
    qdrant = _legacy_fixture(sparse=False)
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["payload"].update(
            {
                "acl_enabled": True,
                "acl_direct_grants": ["not-an-acl-token"],
                "acl_inherited_grants": [],
            }
        )

    plan = _migration(qdrant).preflight()

    assert plan.acl_incomplete_count == 2
    with pytest.raises(MigrationError, match="ACL"):
        _apply(_migration(qdrant), confirm=True)


def test_acl_disabled_with_grants_is_incomplete() -> None:
    qdrant = _legacy_fixture(sparse=False)
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["payload"].update(
            {
                "acl_enabled": False,
                "acl_direct_grants": ["1:user:alice"],
                "acl_inherited_grants": [],
            }
        )

    plan = _migration(qdrant).preflight()

    assert plan.acl_incomplete_count == 2
    with pytest.raises(MigrationError, match="ACL"):
        _apply(_migration(qdrant), confirm=True)


def test_valid_empty_acl_fields_are_complete() -> None:
    qdrant = _legacy_fixture(sparse=False)
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["payload"].update(
            {
                "acl_enabled": False,
                "acl_direct_grants": [],
                "acl_inherited_grants": [],
            }
        )

    result = _apply(_migration(qdrant), confirm=True)

    assert result.target_count == 2


def test_target_metadata_rejects_foreign_points() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    qdrant.collections["current__context__openviking_meta"]["points"]["foreign"] = {
        "id": "foreign",
        "vector": {"meta": [0.0]},
        "payload": {"term": "foreign", "index": 7},
    }

    with pytest.raises(SparseMigrationError, match="unexpected point"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_target_sparse_dictionary_requires_stable_index_for_all_terms() -> None:
    qdrant = _legacy_fixture(sparse=True)
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})
    _apply(migration, confirm=True, allow_acl_fail_open=True)
    term = "foreign"
    qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id(f"openviking:sparse:{term}")
    ] = {
        "id": to_qdrant_point_id(f"openviking:sparse:{term}"),
        "vector": {"meta": [0.0]},
        "payload": {
            "_openviking_sparse_term": True,
            "term": term,
            "index": 999,
        },
    }

    with pytest.raises(SparseMigrationError, match="stable term mapping"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_marker_fingerprint_change_after_preflight_is_rejected(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)
    migration = _migration(qdrant)
    _mark_current_target_building(qdrant, migration)
    plan = migration.preflight()
    original_write_indexes = migration._write_indexes

    def write_indexes(schema, indexes):
        original_write_indexes(schema, indexes)
        qdrant.collections[migration.target_metadata_collection]["points"][
            to_qdrant_point_id("openviking:metadata")
        ]["payload"]["source_fingerprint"] = "changed-after-preflight"

    monkeypatch.setattr(migration, "_write_indexes", write_indexes)

    with pytest.raises(MigrationError, match="fingerprint changed"):
        migration.apply(
            confirm=True,
            plan=plan,
            allow_acl_fail_open=True,
            lock_held=True,
        )


def test_index_409_without_physical_index_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    original_request = qdrant.request

    def reject_indexes(method, path, body=None, *, params=None):
        if method == "PUT" and path.endswith("/current__context/index"):
            raise _FakeHttpError(409)
        return original_request(method, path, body, params=params)

    qdrant.request = reject_indexes  # type: ignore[method-assign]

    with pytest.raises(MigrationError, match="missing payload index"):
        _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    marker = qdrant.collections["current__context__openviking_meta"]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["setup_complete"] is False


def test_target_creation_race_preserves_competing_metadata_marker() -> None:
    qdrant = _legacy_fixture(sparse=False)
    original_request = qdrant.request

    def race_on_target(method, path, body=None, *, params=None):
        if method == "PUT" and path.endswith("/current__context"):
            original_request(method, path, body, params=params)
            raise _FakeHttpError(409)
        return original_request(method, path, body, params=params)

    qdrant.request = race_on_target  # type: ignore[method-assign]

    with pytest.raises(MigrationError, match="without an owned migration marker"):
        _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)

    assert "current__context" in qdrant.collections
    assert "current__context__openviking_meta" not in qdrant.collections


def test_completion_marker_write_is_verified(monkeypatch) -> None:
    qdrant = _legacy_fixture(sparse=False)
    migration = _migration(qdrant)
    original_write_points = migration._write_points

    def drop_completion_marker(collection, points):
        if collection == migration.target_metadata_collection and any(
            point.get("payload", {}).get("setup_complete") is True
            for point in points
        ):
            return
        return original_write_points(collection, points)

    monkeypatch.setattr(migration, "_write_points", drop_completion_marker)

    with pytest.raises(MigrationError, match="completion marker"):
        _apply(migration, confirm=True, allow_acl_fail_open=True)

    marker = qdrant.collections[migration.target_metadata_collection]["points"][
        to_qdrant_point_id("openviking:metadata")
    ]["payload"]
    assert marker["setup_complete"] is False


def test_final_target_identity_readback_rejects_dropped_points() -> None:
    qdrant = _legacy_fixture(sparse=False)
    original_request = qdrant.request

    def drop_one_data_point(method, path, body=None, *, params=None):
        if method == "PUT" and path.endswith("/current__context/points"):
            body = copy.deepcopy(body)
            body["points"] = body["points"][:1]
        return original_request(method, path, body, params=params)

    qdrant.request = drop_one_data_point  # type: ignore[method-assign]

    with pytest.raises(MigrationError, match="target records differ"):
        _apply(_migration(qdrant), confirm=True, allow_acl_fail_open=True)


def test_empty_sparse_only_source_record_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=True)
    point = qdrant.collections["legacy__context"]["points"]["1"]
    point["vector"].pop("vector")
    point["vector"]["sparse_vector"] = {"indices": [], "values": []}

    with pytest.raises(SparseMigrationError, match="no entries"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_dense_vector_override_survives_resume_preflight() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["config"]["params"]["vectors"] = {
        "embedding": {"size": 2, "distance": "Cosine"}
    }
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["vector"]["embedding"] = point["vector"].pop("vector")
    migration = _migration(qdrant, dense_vector_name="embedding")

    _apply(migration, confirm=True, allow_acl_fail_open=True)

    assert migration.preflight().dense_vector_name == "embedding"


def test_unnamed_dense_vector_rejects_a_name_override() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["config"]["params"]["vectors"] = {
        "size": 2,
        "distance": "Cosine",
    }
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["vector"][""] = point["vector"].pop("vector")

    with pytest.raises(MigrationError, match="unnamed source dense vector"):
        _migration(qdrant, dense_vector_name="embedding").preflight()


def test_multiple_named_dense_vectors_require_selection() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["config"]["params"]["vectors"] = {
        "vector": {"size": 2, "distance": "Cosine"},
        "image": {"size": 2, "distance": "Cosine"},
    }
    for point in qdrant.collections["legacy__context"]["points"].values():
        point["vector"]["image"] = [0.0, 1.0]

    with pytest.raises(MigrationError, match="multiple named dense vectors"):
        _migration(qdrant).preflight()


def test_sparse_map_rejects_non_string_terms_and_fractional_indexes() -> None:
    with pytest.raises(SparseMigrationError, match="non-empty string"):
        _migration(_legacy_fixture(sparse=False), sparse_map={"111": None})
    qdrant = _legacy_fixture(sparse=True)
    qdrant.collections["legacy__context"]["points"]["1"]["vector"]["sparse_vector"][
        "indices"
    ] = [111.9]
    with pytest.raises(SparseMigrationError, match="integer"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_sparse_map_accepts_numeric_looking_reverse_terms() -> None:
    qdrant = _legacy_fixture(sparse=True)

    plan = _migration(qdrant, sparse_map={"111": 111, "world": 222}).preflight()

    assert plan.sparse_term_count == 2


def test_fractional_level_fails_closed() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context"]["points"]["1"]["payload"]["level"] = 1.9

    with pytest.raises(MigrationError, match="valid level"):
        _migration(qdrant).preflight()


def test_unnamed_dense_vector_with_named_sparse_vector_is_read() -> None:
    qdrant = _legacy_fixture(sparse=True)
    for point in qdrant.collections["legacy__context"]["points"].values():
        vectors = point["vector"]
        point["vector"] = {
            "": vectors.pop("vector"),
            **vectors,
        }
    migration = _migration(qdrant, sparse_map={111: "hello", 222: "world"})

    _apply(migration, confirm=True, allow_acl_fail_open=True)

    target = qdrant.collections["current__context"]["points"]
    assert target[to_qdrant_point_id("1")]["vector"]["vector"] == [1.0, 0.0]


def test_source_metadata_dimension_must_match_physical_layout() -> None:
    qdrant = _legacy_fixture(sparse=False)
    vector_field = next(
        field
        for field in qdrant.collections["legacy__context__openviking_meta"]["points"][
            _legacy_collection_metadata_id("legacy__context")
        ]["payload"]["meta"]["Fields"]
        if field["FieldName"] == "vector"
    )
    vector_field["Dim"] = 3

    with pytest.raises(MigrationError, match="dimension"):
        _migration(qdrant).preflight()


def test_source_metadata_sparse_declaration_must_match_physical_layout() -> None:
    qdrant = _legacy_fixture(sparse=False)
    qdrant.collections["legacy__context__openviking_meta"]["points"][
        _legacy_collection_metadata_id("legacy__context")
    ][
        "payload"
    ]["meta"]["Fields"].append(
        {"FieldName": "sparse_vector", "FieldType": "sparse_vector"}
    )

    with pytest.raises(MigrationError, match="no sparse vectors"):
        _migration(qdrant).preflight()


def test_source_metadata_sparse_name_must_match_physical_layout() -> None:
    qdrant = _legacy_fixture(sparse=True)
    sparse_field = next(
        field
        for field in qdrant.collections["legacy__context__openviking_meta"]["points"][
            _legacy_collection_metadata_id("legacy__context")
        ]["payload"]["meta"]["Fields"]
        if field["FieldName"] == "sparse_vector"
    )
    sparse_field["FieldName"] = "other_sparse"

    with pytest.raises(MigrationError, match="sparse vector metadata name"):
        _migration(qdrant, sparse_map={111: "hello", 222: "world"}).preflight()


def test_all_migration_collection_names_must_be_distinct() -> None:
    qdrant = _legacy_fixture(sparse=False)

    with pytest.raises(ValueError, match="pairwise distinct"):
        QdrantMigration(
            client=qdrant,
            source_collection="legacy__context",
            target_collection="current__context",
            source_metadata_collection="legacy__context__openviking_meta",
            target_metadata_collection="current__context",
            logical_collection="legacy/context",
            migration_id="mig-1",
        )
