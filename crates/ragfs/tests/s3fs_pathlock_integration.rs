#![cfg(feature = "s3")]

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;

use ragfs::core::{ConfigValue, FileSystem, PluginConfig, WriteFlag};
use ragfs::lock::{MemoryPathLockProvider, PathLockConfig, PathLockManager};
use ragfs::plugins::s3fs::S3FileSystem;

fn test_config() -> Option<PluginConfig> {
    let required = |name: &str| env::var(name).ok().filter(|value| !value.is_empty());
    let endpoint = required("RAGFS_S3_TEST_ENDPOINT")?;
    let bucket = required("RAGFS_S3_TEST_BUCKET")?;
    let access_key_id = required("RAGFS_S3_TEST_ACCESS_KEY_ID")?;
    let secret_access_key = required("RAGFS_S3_TEST_SECRET_ACCESS_KEY")?;

    let mut params = HashMap::new();
    params.insert("bucket".to_string(), ConfigValue::String(bucket));
    params.insert("endpoint".to_string(), ConfigValue::String(endpoint));
    params.insert(
        "region".to_string(),
        ConfigValue::String(
            env::var("RAGFS_S3_TEST_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
        ),
    );
    params.insert(
        "access_key_id".to_string(),
        ConfigValue::String(access_key_id),
    );
    params.insert(
        "secret_access_key".to_string(),
        ConfigValue::String(secret_access_key),
    );
    params.insert("use_path_style".to_string(), ConfigValue::Bool(true));
    params.insert(
        "directory_marker_mode".to_string(),
        ConfigValue::String("none".to_string()),
    );
    params.insert("cache_enabled".to_string(), ConfigValue::Bool(true));
    params.insert("stat_cache_ttl".to_string(), ConfigValue::Int(600));
    params.insert(
        "prefix".to_string(),
        ConfigValue::String(format!("_it/{}", uuid::Uuid::new_v4().simple())),
    );

    Some(PluginConfig::single_backend("s3fs", "/s3", params))
}

#[tokio::test]
async fn pathlock_accepts_s3_implicit_directory_after_negative_stat_cache() {
    let Some(config) = test_config() else {
        eprintln!(
            "skipping S3FS PathLock integration test; set \
             RAGFS_S3_TEST_ENDPOINT, RAGFS_S3_TEST_BUCKET, \
             RAGFS_S3_TEST_ACCESS_KEY_ID, and RAGFS_S3_TEST_SECRET_ACCESS_KEY"
        );
        return;
    };

    let fs = Arc::new(
        S3FileSystem::new(&config)
            .await
            .expect("construct S3 filesystem"),
    );
    let parent = "/archive_040";

    assert!(fs.stat(parent).await.is_err(), "parent starts absent");
    fs.write(
        "/archive_040/messages.jsonl",
        br#"{"role":"user"}"#,
        0,
        WriteFlag::Create,
    )
    .await
    .expect("create child object");
    assert!(
        fs.stat(parent).await.is_err(),
        "negative parent stat remains cached before mkdir"
    );

    let manager = PathLockManager::new(
        fs.clone(),
        Arc::new(MemoryPathLockProvider::new()),
        PathLockConfig::default(),
    );
    let lease = manager
        .acquire_exact("/archive_040/.abstract.md", Duration::ZERO, None)
        .await
        .expect("PathLock should accept an existing implicit S3 directory");

    manager
        .release(&lease)
        .await
        .expect("release PathLock lease");
    fs.remove_all(parent).await.expect("clean up test prefix");
}
