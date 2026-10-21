use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::Digest;
use soroban_cost_estimator::cache;

/// Serialize cache tests because `std::env::set_var` is not thread-safe.
static HOME_MUTEX: Mutex<()> = Mutex::new(());

/// Number of worker threads used by the concurrency tests.
const CONCURRENT_THREADS: usize = 8;
/// Number of cache entries each worker thread writes/reads.
const ENTRIES_PER_THREAD: usize = 25;

/// Run a test with HOME pointing to a temporary directory so cache
/// operations don't touch the real user's home.
///
/// Uses a unique temp directory per call to avoid races on the same dir.
/// Uses a global mutex to serialize env-var manipulation.
fn with_temp_home<F>(test: F)
where
    F: FnOnce(&PathBuf) + std::panic::UnwindSafe,
{
    let guard = HOME_MUTEX.lock().expect("cache test mutex");

    // Generate a unique suffix so parallel tests don't share the same dir
    let suffix: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!(
        "soroban_cache_test_{}_{}",
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp home");

    let old_home = std::env::var_os("HOME");
    let old_userprofile = std::env::var_os("USERPROFILE");
    // SAFETY: serialized by HOME_MUTEX, no other thread reads these env vars
    // during this block.
    unsafe {
        std::env::set_var("HOME", &tmp);
        // The library prefers $HOME on Unix and $USERPROFILE on Windows when
        // resolving the data dir, so set both to keep the cache inside the
        // temp dir on every platform.
        std::env::set_var("USERPROFILE", &tmp);
    }

    // Run the test; catch panics so we can clean up regardless
    let result = std::panic::catch_unwind(|| {
        // Verify the cache dir resolves inside the temp dir
        let data_dir = soroban_cost_estimator::paths::data_dir().expect("data dir");
        assert!(
            data_dir.starts_with(&tmp),
            "data dir should point to temp dir: {} vs {}",
            data_dir.display(),
            tmp.display()
        );
        test(&tmp);
    });

    // SAFETY: serialized by HOME_MUTEX, no other thread reads these env vars
    // during this block.
    unsafe {
        if let Some(old) = old_home {
            std::env::set_var("HOME", old);
        } else {
            std::env::remove_var("HOME");
        }
        if let Some(old) = old_userprofile {
            std::env::set_var("USERPROFILE", old);
        } else {
            std::env::remove_var("USERPROFILE");
        }
    }

    // Clean up temp dir
    let _ = std::fs::remove_dir_all(&tmp);

    // Drop the guard BEFORE resume_unwind to avoid poisoning the mutex
    drop(guard);

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn test_save_and_load_estimate() {
    with_temp_home(|_tmp| {
        // Save an estimate
        cache::save_estimate(
            "abc123",
            "my_func",
            &["arg1".to_string(), "arg2".to_string()],
            "testnet",
            42,
            1_000_000,
            200_000,
            50_000,
            None,
            true,
        )
        .expect("save estimate");

        // Load it back
        let loaded = cache::load_estimate(
            "abc123",
            "my_func",
            &["arg1".to_string(), "arg2".to_string()],
        )
        .expect("load estimate")
        .expect("estimate should exist");

        assert_eq!(loaded.wasm_hash, "abc123");
        assert_eq!(loaded.function, "my_func");
        assert_eq!(loaded.network, "testnet");
        assert_eq!(loaded.ledger, 42);
        assert_eq!(loaded.total_stroops, 1_000_000);
        assert_eq!(loaded.cpu_instructions, 200_000);
        assert_eq!(loaded.memory_bytes, 50_000);
    });
}

#[test]
fn test_load_nonexistent_estimate() {
    with_temp_home(|_tmp| {
        let result = cache::load_estimate("nope", "no_func", &[]).expect("load nonexistent");
        assert!(result.is_none(), "nonexistent estimate should return None");
    });
}

#[test]
fn test_different_args_produce_different_cache_keys() {
    with_temp_home(|_tmp| {
        // Save with one set of args
        cache::save_estimate(
            "hash1",
            "fn1",
            &["a".to_string()],
            "testnet",
            1,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("save with args [a]");

        // Save with different args
        cache::save_estimate(
            "hash1",
            "fn1",
            &["b".to_string()],
            "testnet",
            2,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("save with args [b]");

        // Load with first args → should get ledger 1
        let r1 = cache::load_estimate("hash1", "fn1", &["a".to_string()])
            .expect("load [a]")
            .expect("should exist");
        assert_eq!(r1.ledger, 1);

        // Load with second args → should get ledger 2
        let r2 = cache::load_estimate("hash1", "fn1", &["b".to_string()])
            .expect("load [b]")
            .expect("should exist");
        assert_eq!(r2.ledger, 2);
    });
}

#[test]
fn test_list_cached_estimates_filters_by_network() {
    with_temp_home(|_tmp| {
        // Save estimates for two networks (different functions so they don't collide)
        cache::save_estimate("h1", "f_testnet", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("testnet save");
        cache::save_estimate(
            "h1",
            "f_mainnet",
            &[],
            "mainnet",
            2,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("mainnet save");

        let testnet_estimates = cache::list_cached_estimates("testnet").expect("list testnet");
        assert_eq!(testnet_estimates.len(), 1, "should have 1 testnet estimate");
        assert_eq!(testnet_estimates[0].ledger, 1);

        let mainnet_estimates = cache::list_cached_estimates("mainnet").expect("list mainnet");
        assert_eq!(mainnet_estimates.len(), 1, "should have 1 mainnet estimate");
        assert_eq!(mainnet_estimates[0].ledger, 2);

        // Unknown network → empty
        let futurenet = cache::list_cached_estimates("futurenet").expect("list futurenet");
        assert!(futurenet.is_empty(), "futurenet should have no estimates");
    });
}

#[test]
fn test_find_stale_estimates() {
    with_temp_home(|_tmp| {
        // Save at ledger 5
        cache::save_estimate("h1", "f1", &[], "testnet", 5, 100, 10, 5, None, true)
            .expect("save at 5");
        // Save at ledger 10
        cache::save_estimate("h1", "f2", &[], "testnet", 10, 200, 20, 10, None, true)
            .expect("save at 10");
        // Save at ledger 15
        cache::save_estimate("h1", "f3", &[], "testnet", 15, 300, 30, 15, None, true)
            .expect("save at 15");

        let all = cache::list_cached_estimates("testnet").expect("list all");
        assert_eq!(all.len(), 3, "should have 3 estimates");

        // Current ledger = 12 → stale = ones at 5 and 10
        let stale = cache::find_stale_estimates(&all, 12);
        assert_eq!(stale.len(), 2, "should find 2 stale at ledger 12");
        let stale_names: Vec<&str> = stale.iter().map(|e| e.function.as_str()).collect();
        assert!(stale_names.contains(&"f1"));
        assert!(stale_names.contains(&"f2"));
        assert!(!stale_names.contains(&"f3"));

        // Current ledger = 5 → only the one at 5 is NOT stale
        let stale = cache::find_stale_estimates(&all, 5);
        assert_eq!(stale.len(), 0, "none should be stale at ledger 5");

        // Current ledger = 20 → all are stale
        let stale = cache::find_stale_estimates(&all, 20);
        assert_eq!(stale.len(), 3, "all should be stale at ledger 20");
    });
}

#[test]
fn test_cache_is_empty_initially() {
    with_temp_home(|_tmp| {
        let estimates = cache::list_cached_estimates("testnet").expect("list on empty cache");
        assert!(estimates.is_empty(), "fresh cache should be empty");
    });
}

#[test]
fn test_overwrite_existing_estimate() {
    with_temp_home(|_tmp| {
        // Save at ledger 10
        cache::save_estimate(
            "h1",
            "f1",
            &["x".to_string()],
            "testnet",
            10,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("first save");

        // Overwrite at ledger 20
        cache::save_estimate(
            "h1",
            "f1",
            &["x".to_string()],
            "testnet",
            20,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("overwrite");

        // Load → should get ledger 20
        let loaded = cache::load_estimate("h1", "f1", &["x".to_string()])
            .expect("load")
            .expect("should exist");
        assert_eq!(loaded.ledger, 20);
        assert_eq!(loaded.total_stroops, 200);
    });
}

#[test]
fn test_verify_cache_empty() {
    with_temp_home(|_tmp| {
        let statuses = cache::verify_cache().expect("verify on empty cache");
        assert!(statuses.is_empty(), "fresh cache should have no entries");
    });
}

#[test]
fn test_verify_cache_all_valid() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save f1");
        cache::save_estimate("h2", "f2", &[], "mainnet", 2, 200, 20, 10, None, true)
            .expect("save f2");

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 2, "should report both entries");
        assert!(
            statuses.iter().all(|s| s.valid),
            "entries written by save_estimate should all be valid: {statuses:?}"
        );
    });
}

#[test]
fn test_verify_cache_detects_corrupted_entries() {
    with_temp_home(|tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save f1");

        // Corrupt entry: a row written by a newer tool (future schema version)
        // cannot be safely migrated forward, so it is flagged as invalid —
        // the SQLite equivalent of a malformed/unreadable JSON file.
        insert_raw_row(
            tmp,
            cache::CACHE_SCHEMA_VERSION + 1,
            "garbage",
            "garbage_func",
            &[],
            "testnet",
            1,
            "2026-01-01T00:00:00Z",
        );

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 2, "should report both entries");

        let corrupt: Vec<&cache::CacheEntryStatus> = statuses.iter().filter(|s| !s.valid).collect();
        assert_eq!(corrupt.len(), 1, "future-version entry should be flagged");
        assert!(
            corrupt[0].filename.starts_with("garbage"),
            "future-version entry should be reported: {}",
            corrupt[0].filename
        );
    });
}

#[test]
fn test_verify_cache_ignores_non_json_files() {
    with_temp_home(|tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save f1");

        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        std::fs::write(dir.join("notes.txt"), "not a cache entry").expect("write txt");

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 1, "only .json files should be checked");
        assert!(statuses[0].valid);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Cross-network cache isolation
// ─────────────────────────────────────────────────────────────────────────

/// Saving the same key (wasm_hash + function + args) on two different
/// networks overwrites the previous entry.  This documents the current
/// behaviour so any future fix can be detected by these tests.
#[test]
fn test_same_key_different_networks_overwrites_previous() {
    with_temp_home(|_tmp| {
        // First write: testnet, ledger 10
        cache::save_estimate(
            "hash1",
            "func1",
            &["arg".to_string()],
            "testnet",
            10,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("save testnet");

        let loaded = cache::load_estimate("hash1", "func1", &["arg".to_string()])
            .expect("load")
            .expect("should exist");
        assert_eq!(loaded.network, "testnet");
        assert_eq!(loaded.ledger, 10);

        // Second write: mainnet, same key, ledger 20
        cache::save_estimate(
            "hash1",
            "func1",
            &["arg".to_string()],
            "mainnet",
            20,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("save mainnet");

        // The testnet entry is gone — the file was overwritten.
        let loaded = cache::load_estimate("hash1", "func1", &["arg".to_string()])
            .expect("load")
            .expect("should still exist");
        assert_eq!(
            loaded.network, "mainnet",
            "mainnet should have overwritten testnet"
        );
        assert_eq!(loaded.ledger, 20);

        // list_cached_estimates confirms the leak.
        let testnet = cache::list_cached_estimates("testnet").expect("list");
        assert!(
            testnet.is_empty(),
            "testnet should have no entries after overwrite"
        );
        let mainnet = cache::list_cached_estimates("mainnet").expect("list");
        assert_eq!(mainnet.len(), 1);
    });
}

/// When different networks use distinct wasm_hash + function + args keys,
/// each network's estimates are fully isolated.
#[test]
fn test_different_keys_different_networks_are_isolated() {
    with_temp_home(|_tmp| {
        cache::save_estimate(
            "hashA",
            "funcA",
            &["a1".to_string()],
            "testnet",
            1,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("testnet A");
        cache::save_estimate(
            "hashB",
            "funcB",
            &["b1".to_string()],
            "mainnet",
            2,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("mainnet B");
        cache::save_estimate(
            "hashC",
            "funcC",
            &["c1".to_string()],
            "futurenet",
            3,
            300,
            30,
            15,
            None,
            true,
        )
        .expect("futurenet C");

        let tn = cache::list_cached_estimates("testnet").expect("list testnet");
        assert_eq!(tn.len(), 1);
        assert_eq!(tn[0].function, "funcA");
        assert_eq!(tn[0].network, "testnet");

        let mn = cache::list_cached_estimates("mainnet").expect("list mainnet");
        assert_eq!(mn.len(), 1);
        assert_eq!(mn[0].function, "funcB");
        assert_eq!(mn[0].network, "mainnet");

        let fn_ = cache::list_cached_estimates("futurenet").expect("list futurenet");
        assert_eq!(fn_.len(), 1);
        assert_eq!(fn_[0].function, "funcC");
        assert_eq!(fn_[0].network, "futurenet");
    });
}

/// load_estimate does not filter by network — it returns whatever the file
/// contains.  This test documents that cross-network calls return the
/// *stored* network, even if the caller intended a different one.
#[test]
fn test_load_estimate_returns_stored_network_not_caller_network() {
    with_temp_home(|_tmp| {
        // Save on testnet
        cache::save_estimate(
            "hash",
            "fn",
            &["x".to_string()],
            "testnet",
            10,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("save");

        // load_estimate has no network parameter — it returns whatever was saved.
        let loaded = cache::load_estimate("hash", "fn", &["x".to_string()])
            .expect("load")
            .expect("should exist");
        assert_eq!(loaded.network, "testnet");
    });
}

/// Multiple networks with the same wasm_hash and function but different args
/// should not leak — the args hash isolates them.
#[test]
fn test_same_wasm_function_different_args_different_networks_isolated() {
    with_temp_home(|_tmp| {
        cache::save_estimate(
            "hash",
            "func",
            &["arg-tn".to_string()],
            "testnet",
            1,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("testnet save");
        cache::save_estimate(
            "hash",
            "func",
            &["arg-mn".to_string()],
            "mainnet",
            2,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("mainnet save");

        let tn = cache::load_estimate("hash", "func", &["arg-tn".to_string()])
            .expect("load tn")
            .expect("should exist");
        assert_eq!(tn.network, "testnet");
        assert_eq!(tn.ledger, 1);

        let mn = cache::load_estimate("hash", "func", &["arg-mn".to_string()])
            .expect("load mn")
            .expect("should exist");
        assert_eq!(mn.network, "mainnet");
        assert_eq!(mn.ledger, 2);

        // Both still appear under their respective network lists.
        assert_eq!(cache::list_cached_estimates("testnet").unwrap().len(), 1);
        assert_eq!(cache::list_cached_estimates("mainnet").unwrap().len(), 1);
    });
}

/// find_stale_estimates must not mix networks — stale entries from one
/// network must not appear when querying another.
#[test]
fn test_find_stale_estimates_does_not_mix_networks() {
    with_temp_home(|_tmp| {
        // testnet: ledger 5 (stale at ledger 10)
        cache::save_estimate("h", "f-tn", &[], "testnet", 5, 100, 10, 5, None, true).expect("tn");
        // mainnet: ledger 12 (NOT stale at ledger 10)
        cache::save_estimate("h", "f-mn", &[], "mainnet", 12, 200, 20, 10, None, true).expect("mn");

        let tn_all = cache::list_cached_estimates("testnet").expect("list tn");
        let tn_stale = cache::find_stale_estimates(&tn_all, 10);
        assert_eq!(tn_stale.len(), 1, "testnet ledger 5 should be stale at 10");
        assert_eq!(tn_stale[0].network, "testnet");

        let mn_all = cache::list_cached_estimates("mainnet").expect("list mn");
        let mn_stale = cache::find_stale_estimates(&mn_all, 10);
        assert!(
            mn_stale.is_empty(),
            "mainnet ledger 12 should not be stale at 10"
        );
    });
}

/// verify_cache must report entries from every network as valid, and not
/// leak network information across files.
#[test]
fn test_verify_cache_across_networks_all_valid() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true).expect("tn");
        cache::save_estimate("h2", "f2", &[], "mainnet", 2, 200, 20, 10, None, true).expect("mn");
        cache::save_estimate("h3", "f3", &[], "futurenet", 3, 300, 30, 15, None, true).expect("fn");

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 3, "all three entries should be verified");
        assert!(
            statuses.iter().all(|s| s.valid),
            "all entries should be valid"
        );
    });
}

/// Concurrent saves from two different networks to the same key must leave
/// a valid entry behind — no torn writes, no corruption.
#[test]
fn test_concurrent_cross_network_same_key_no_corruption() {
    with_temp_home(|_tmp| {
        let args = vec!["shared".to_string()];
        let tn_args = args.clone();
        let mn_args = args.clone();

        let tn = std::thread::spawn(move || {
            cache::save_estimate(
                "shared-hash",
                "shared-func",
                &tn_args,
                "testnet",
                1,
                100,
                10,
                5,
                None,
                true,
            )
            .expect("concurrent testnet save");
        });
        let mn = std::thread::spawn(move || {
            cache::save_estimate(
                "shared-hash",
                "shared-func",
                &mn_args,
                "mainnet",
                2,
                200,
                20,
                10,
                None,
                true,
            )
            .expect("concurrent mainnet save");
        });

        tn.join().expect("testnet thread panicked");
        mn.join().expect("mainnet thread panicked");

        // The surviving entry must be valid JSON.
        let loaded = cache::load_estimate("shared-hash", "shared-func", &args)
            .expect("load")
            .expect("shared key should exist");
        assert!(
            loaded.network == "testnet" || loaded.network == "mainnet",
            "surviving entry must be from one of the two networks: {loaded:?}"
        );

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 1, "one entry for the shared key");
        assert!(statuses[0].valid, "entry must be valid: {statuses:?}");
    });
}

/// Load after cross-network overwrite must return the latest writer's data,
/// not the first writer's. This is the "leak" scenario: a testnet cache
/// entry is silently replaced by a mainnet write.
#[test]
fn test_load_after_cross_network_overwrite_returns_latest_writer() {
    with_temp_home(|_tmp| {
        cache::save_estimate(
            "h",
            "f",
            &["x".to_string()],
            "testnet",
            100,
            1000,
            100,
            50,
            None,
            true,
        )
        .expect("save testnet");

        let before = cache::load_estimate("h", "f", &["x".to_string()])
            .unwrap()
            .unwrap();
        assert_eq!(before.network, "testnet");
        assert_eq!(before.ledger, 100);

        // Overwrite with mainnet
        cache::save_estimate(
            "h",
            "f",
            &["x".to_string()],
            "mainnet",
            200,
            2000,
            200,
            100,
            None,
            true,
        )
        .expect("save mainnet");

        let after = cache::load_estimate("h", "f", &["x".to_string()])
            .unwrap()
            .unwrap();
        assert_eq!(
            after.network, "mainnet",
            "should return mainnet after overwrite"
        );
        assert_eq!(after.ledger, 200);

        // Verify the testnet list is now empty for this key.
        let tn = cache::list_cached_estimates("testnet").unwrap();
        assert!(
            tn.is_empty(),
            "testnet list should be empty after mainnet overwrite"
        );
    });
}

/// Concurrent `save_estimate`/`load_estimate` calls on distinct cache keys
/// must not corrupt the cache.
///
/// Each thread owns a unique `(wasm_hash, function)` pair and writes/reads
/// `ENTRIES_PER_THREAD` estimates with unique args, so no two threads ever
/// touch the same cache file. After every thread finishes, every entry must
/// still be present, loadable, and parseable.
#[test]
fn test_concurrent_save_and_load_estimates() {
    with_temp_home(|_tmp| {
        let handles: Vec<_> = (0..CONCURRENT_THREADS)
            .map(|t| {
                std::thread::spawn(move || {
                    let wasm_hash = format!("hash-{t}");
                    let function = format!("func-{t}");
                    for j in 0..ENTRIES_PER_THREAD {
                        let args = vec![format!("arg-{t}-{j}")];
                        cache::save_estimate(
                            &wasm_hash,
                            &function,
                            &args,
                            "testnet",
                            j as u32,
                            1_000 + j as i64,
                            10_000 + j as u64,
                            1_000 + j as u64,
                            None,
                            true,
                        )
                        .expect("concurrent save");

                        // Load back immediately; only this thread wrote this key.
                        let loaded = cache::load_estimate(&wasm_hash, &function, &args)
                            .expect("concurrent load")
                            .expect("estimate saved by this thread should load");
                        assert_eq!(loaded.ledger, j as u32);
                        assert_eq!(loaded.total_stroops, 1_000 + j as i64);
                        assert_eq!(loaded.cpu_instructions, 10_000 + j as u64);
                        assert_eq!(loaded.memory_bytes, 1_000 + j as u64);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("concurrent save/load thread panicked");
        }

        // Every concurrently written entry must still be present and intact.
        let estimates = cache::list_cached_estimates("testnet").expect("list after concurrent");
        assert_eq!(
            estimates.len(),
            CONCURRENT_THREADS * ENTRIES_PER_THREAD,
            "all concurrently written estimates should be present"
        );
        let statuses = cache::verify_cache().expect("verify after concurrent");
        assert_eq!(statuses.len(), CONCURRENT_THREADS * ENTRIES_PER_THREAD);
        assert!(
            statuses.iter().all(|s| s.valid),
            "concurrent save/load must not corrupt entries: {statuses:?}"
        );
    });
}

/// Concurrent `load_estimate` calls must not corrupt the cache.
///
/// Seed a known set of entries, then hammer the cache with reads from many
/// threads at once. Every entry must load back with its exact values and the
/// cache must still verify as fully valid afterwards.
#[test]
fn test_concurrent_load_estimates() {
    with_temp_home(|_tmp| {
        // Seed the cache sequentially so every entry exists before the reads.
        for t in 0..CONCURRENT_THREADS {
            let wasm_hash = format!("hash-{t}");
            let function = format!("func-{t}");
            for j in 0..ENTRIES_PER_THREAD {
                cache::save_estimate(
                    &wasm_hash,
                    &function,
                    &[format!("arg-{t}-{j}")],
                    "testnet",
                    j as u32,
                    1_000 + j as i64,
                    10_000 + j as u64,
                    1_000 + j as u64,
                    None,
                    true,
                )
                .expect("seed save");
            }
        }

        let handles: Vec<_> = (0..CONCURRENT_THREADS)
            .map(|t| {
                std::thread::spawn(move || {
                    for j in 0..ENTRIES_PER_THREAD {
                        let loaded = cache::load_estimate(
                            &format!("hash-{t}"),
                            &format!("func-{t}"),
                            &[format!("arg-{t}-{j}")],
                        )
                        .expect("concurrent load")
                        .expect("seeded estimate should load");
                        assert_eq!(loaded.ledger, j as u32);
                        assert_eq!(loaded.total_stroops, 1_000 + j as i64);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("concurrent load thread panicked");
        }

        let statuses = cache::verify_cache().expect("verify after concurrent loads");
        assert_eq!(statuses.len(), CONCURRENT_THREADS * ENTRIES_PER_THREAD);
        assert!(
            statuses.iter().all(|s| s.valid),
            "concurrent loads must not corrupt entries: {statuses:?}"
        );
    });
}

/// Insert a raw row directly into the SQLite cache database, bypassing
/// `save_estimate` so tests can exercise the migration path directly.
///
/// The `args_hash` column is computed the same way the library does
/// (SHA-256 over the concatenated arg strings).
fn insert_raw_row(
    home: &Path,
    version: u32,
    wasm_hash: &str,
    function: &str,
    args: &[&str],
    network: &str,
    ledger: u32,
    timestamp: &str,
) {
    let mut hasher = sha2::Sha256::new();
    for arg in args {
        hasher.update(arg.as_bytes());
    }
    let args_hash = hex::encode(hasher.finalize());

    let dir = home.join(".soroban-cost-estimator");
    std::fs::create_dir_all(&dir).expect("create data dir");
    let db = dir.join("cache.db");
    let conn = rusqlite::Connection::open(&db).expect("open cache db");
    // Ensure the schema exists in this exact database before writing rows
    // directly.
    cache::ensure_cache_schema(&conn).expect("ensure cache schema");

    conn.execute(
        "INSERT OR REPLACE INTO estimates \
         (version, wasm_hash, function, args_hash, network, ledger, total_stroops, cpu_instructions, memory_bytes, timestamp) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        rusqlite::params![
            version as i64,
            wasm_hash,
            function,
            args_hash,
            network,
            ledger as i64,
            100i64,
            10i64,
            5i64,
            timestamp,
        ],
    )
    .expect("insert raw row");
}

/// Write a raw cache entry with an explicit schema `version` (or the initial
/// schema when `version: None`), bypassing `save_estimate`.
fn write_raw_entry(
    tmp: &Path,
    wasm_hash: &str,
    function: &str,
    args: &[&str],
    version: Option<u32>,
    ledger: u32,
) {
    let version = version.unwrap_or(cache::INITIAL_SCHEMA_VERSION);
    insert_raw_row(
        tmp,
        version,
        wasm_hash,
        function,
        args,
        "testnet",
        ledger,
        "2026-01-01T00:00:00Z",
    );
}

/// The current schema version constant exposed by the library.
///
/// Kept in sync with `cache::CACHE_SCHEMA_VERSION`. If the library bumps
/// the schema, these tests must be revisited.
fn current_schema_version() -> u32 {
    cache::CACHE_SCHEMA_VERSION
}

// ─────────────────────────────────────────────────────────────────────────
// clear_cache (issue #24)
// ─────────────────────────────────────────────────────────────────────────

/// `clear_cache` deletes every entry recorded for the requested network and
/// returns the number of rows removed, leaving other networks untouched.
#[test]
fn test_clear_cache_removes_requested_network_only() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save testnet f1");
        cache::save_estimate("h2", "f2", &[], "testnet", 2, 200, 20, 10, None, true)
            .expect("save testnet f2");
        cache::save_estimate("h3", "f3", &[], "mainnet", 3, 300, 30, 15, None, true)
            .expect("save mainnet f3");

        let removed = cache::clear_cache("testnet").expect("clear testnet");
        assert_eq!(
            removed, 2,
            "clearing testnet should delete its two entries, got {removed}"
        );

        let testnet = cache::list_cached_estimates("testnet").expect("list testnet");
        assert!(testnet.is_empty(), "testnet should be empty after clearing");
        let mainnet = cache::list_cached_estimates("mainnet").expect("list mainnet");
        assert_eq!(mainnet.len(), 1, "mainnet entries must be untouched");
        assert_eq!(mainnet[0].function, "f3");
        assert_eq!(mainnet[0].network, "mainnet");

        // A second clear on the already-empty network deletes nothing.
        let removed = cache::clear_cache("testnet").expect("clear testnet again");
        assert_eq!(removed, 0, "second clear should delete nothing");
    });
}

/// Clearing a network with no entries (or a brand-new cache) is a no-op that
/// reports zero and does not error.
#[test]
fn test_clear_cache_on_empty_cache_returns_zero() {
    with_temp_home(|_tmp| {
        let removed = cache::clear_cache("futurenet").expect("clear empty cache");
        assert_eq!(removed, 0, "an empty cache should report zero cleared");
    });
}

/// Clearing one network must never leak into a different network's entries.
#[test]
fn test_clear_cache_never_touches_other_networks() {
    with_temp_home(|_tmp| {
        cache::save_estimate("hA", "fA", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save testnet");
        cache::save_estimate("hB", "fB", &[], "mainnet", 2, 200, 20, 10, None, true)
            .expect("save mainnet");
        cache::save_estimate("hC", "fC", &[], "futurenet", 3, 300, 30, 15, None, true)
            .expect("save futurenet");

        let removed = cache::clear_cache("mainnet").expect("clear mainnet");
        assert_eq!(removed, 1, "only the mainnet entry should be deleted");

        assert_eq!(cache::list_cached_estimates("testnet").unwrap().len(), 1);
        assert_eq!(cache::list_cached_estimates("mainnet").unwrap().len(), 0);
        assert_eq!(cache::list_cached_estimates("futurenet").unwrap().len(), 1);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Schema versioning & migration
// ─────────────────────────────────────────────────────────────────────────

/// Entries saved by `save_estimate` carry the current schema version, and
/// loading them returns the same version.
#[test]
fn test_saved_entries_are_current_schema_version() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 3, 100, 10, 5, None, true).expect("save");
        let loaded = cache::load_estimate("h1", "f1", &[])
            .expect("load")
            .expect("entry should exist");
        assert_eq!(loaded.version, current_schema_version());
    });
}

/// A legacy entry (no `version` key) loads successfully and is treated as
/// the initial schema version, which then equals the current schema.
#[test]
fn test_load_legacy_entry_without_version_field() {
    with_temp_home(|tmp| {
        // No `version` key, like entries written before versioning
        // was introduced.
        write_raw_entry(tmp, "legacy", "old_func", &["a"], None, 7);
        let loaded = cache::load_estimate("legacy", "old_func", &["a".to_string()])
            .expect("legacy entry should load")
            .expect("entry should exist");
        assert_eq!(loaded.version, current_schema_version());
        assert_eq!(loaded.wasm_hash, "legacy");
        assert_eq!(loaded.ledger, 7);
    });
}

/// An entry that already carries the current version passes through
/// `migrate_to_latest` unchanged (fields and version intact).
#[test]
fn test_migrate_to_latest_current_version_is_identity() {
    with_temp_home(|_tmp| {
        let entry = cache::CachedEstimate {
            version: current_schema_version(),
            wasm_hash: "abc".to_string(),
            function: "f".to_string(),
            args_hash: "def".to_string(),
            network: "testnet".to_string(),
            ledger: 1,
            total_stroops: 100,
            cpu_instructions: 10,
            memory_bytes: 5,
            timestamp: "t".to_string(),
            execution_duration_ms: 42,
            success: true,
            network_passphrase: None,
            core_version: None,
            io: None,
        };
        let migrated = cache::migrate_to_latest(entry.clone()).expect("migrate");
        assert_eq!(migrated.version, current_schema_version());
        assert_eq!(migrated.ledger, 1);
    });
}

/// An entry with a version *newer* than the current schema is rejected by
/// `migrate_to_latest` rather than silently misread.
#[test]
fn test_migrate_to_latest_rejects_future_version() {
    with_temp_home(|_tmp| {
        let entry = cache::CachedEstimate {
            version: current_schema_version() + 1,
            wasm_hash: "abc".to_string(),
            function: "f".to_string(),
            args_hash: "def".to_string(),
            network: "testnet".to_string(),
            ledger: 1,
            total_stroops: 100,
            cpu_instructions: 10,
            memory_bytes: 5,
            timestamp: "t".to_string(),
            execution_duration_ms: 0,
            success: true,
            network_passphrase: None,
            core_version: None,
            io: None,
        };
        let err = cache::migrate_to_latest(entry).expect_err("future version must be rejected");
        assert!(err.to_string().contains("newer"), "unhelpful error: {err}");
    });
}

/// `load_estimate` surfaces the error for an entry written by a newer tool,
/// instead of returning a misleading success.
#[test]
fn test_load_rejects_future_version_entry() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "future",
            "new_func",
            &["b"],
            Some(current_schema_version() + 1),
            1,
        );
        let result = cache::load_estimate("future", "new_func", &["b".to_string()]);
        assert!(
            result.is_err(),
            "future-version entries must fail to load, got {result:?}"
        );
    });
}

/// `verify_cache` flags future-version entries as not valid and records the
/// detected version, even though their JSON parses cleanly.
#[test]
fn test_verify_cache_flags_future_version_entries() {
    with_temp_home(|tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save valid");
        write_raw_entry(
            tmp,
            "future",
            "new_func",
            &["b"],
            Some(current_schema_version() + 1),
            1,
        );

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 2, "both .json files should be reported");

        let future = statuses
            .iter()
            .find(|s| s.filename.starts_with("future"))
            .expect("future entry should be reported");
        assert!(!future.valid, "future-version entry must be flagged");

        let good = statuses
            .iter()
            .find(|s| s.filename.starts_with("h1"))
            .expect("valid entry should be reported");
        assert!(good.valid, "current-version entry must stay valid");
        assert_eq!(good.version, Some(current_schema_version()));
    });
}

/// Legacy entries (no version key) are reported as valid by `verify_cache`,
/// with their detected version defaulting to the initial schema.
#[test]
fn test_verify_cache_accepts_legacy_entries() {
    with_temp_home(|tmp| {
        write_raw_entry(tmp, "legacy", "old_func", &["a"], None, 7);
        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 1);
        assert!(
            statuses[0].valid,
            "legacy entry should verify as valid: {statuses:?}"
        );
        assert_eq!(statuses[0].version, Some(cache::INITIAL_SCHEMA_VERSION));
    });
}

/// Concurrent saves to the *same* cache key must leave a valid entry behind.
///
/// Two threads race to write the same `(wasm_hash, function, args)` key with
/// different ledgers. Whichever write lands last wins, but the surviving file
/// must parse as a valid `CachedEstimate` (no torn writes) and the cache must
/// verify cleanly.
#[test]
fn test_concurrent_same_key_saves_leave_valid_entry() {
    with_temp_home(|_tmp| {
        let args = vec!["shared".to_string()];
        let handles: Vec<_> = (0..CONCURRENT_THREADS)
            .map(|t| {
                let args = args.clone();
                std::thread::spawn(move || {
                    cache::save_estimate(
                        "shared-hash",
                        "shared-func",
                        &args,
                        "testnet",
                        t as u32,
                        1_000 + t as i64,
                        10_000,
                        1_000,
                        None,
                        true,
                    )
                    .expect("concurrent same-key save");
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("concurrent same-key thread panicked");
        }

        // The surviving entry must be one of the written variants.
        let loaded = cache::load_estimate("shared-hash", "shared-func", &args)
            .expect("load shared key")
            .expect("shared key should exist after concurrent saves");
        assert_eq!(loaded.wasm_hash, "shared-hash");
        assert_eq!(loaded.function, "shared-func");
        assert!(
            loaded.ledger < CONCURRENT_THREADS as u32,
            "ledger must be one of the written variants: {loaded:?}"
        );

        let statuses = cache::verify_cache().expect("verify after same-key saves");
        assert_eq!(statuses.len(), 1, "one entry for the shared key");
        assert!(
            statuses[0].valid,
            "shared-key entry must stay valid: {statuses:?}"
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// TTL (time-to-live) freshness
// ─────────────────────────────────────────────────────────────────────────

/// Write a raw cache entry with an explicit `timestamp`, bypassing
/// `save_estimate` so tests can control how old an entry is for TTL checks.
fn write_raw_entry_with_timestamp(
    tmp: &Path,
    wasm_hash: &str,
    function: &str,
    args: &[&str],
    timestamp: &str,
) {
    insert_raw_row(
        tmp,
        cache::CACHE_SCHEMA_VERSION,
        wasm_hash,
        function,
        args,
        "testnet",
        7,
        timestamp,
    );
}

/// An entry timestamped "now" is fresh under a TTL of one hour.
#[test]
fn test_is_cache_entry_fresh_within_ttl() {
    with_temp_home(|tmp| {
        let now = chrono::Utc::now().to_rfc3339();
        write_raw_entry_with_timestamp(tmp, "h1", "f1", &["a"], &now);

        let entry = cache::load_estimate("h1", "f1", &["a".to_string()])
            .expect("load")
            .expect("entry should exist");
        assert!(
            cache::is_cache_entry_fresh(&entry, std::time::Duration::from_secs(3600)),
            "an entry written now should be fresh under a 1h TTL"
        );
    });
}

/// An entry older than the TTL is not fresh.
#[test]
fn test_is_cache_entry_fresh_expired() {
    with_temp_home(|tmp| {
        let two_hours_ago = (chrono::Utc::now() - chrono::TimeDelta::hours(2)).to_rfc3339();
        write_raw_entry_with_timestamp(tmp, "h1", "f1", &["a"], &two_hours_ago);

        let entry = cache::load_estimate("h1", "f1", &["a".to_string()])
            .expect("load")
            .expect("entry should exist");
        assert!(
            !cache::is_cache_entry_fresh(&entry, std::time::Duration::from_secs(3600)),
            "an entry written 2h ago should be stale under a 1h TTL"
        );
    });
}

/// An entry whose timestamp cannot be parsed is never considered fresh:
/// an unverifiable age must not be trusted.
#[test]
fn test_is_cache_entry_fresh_unparseable_timestamp() {
    with_temp_home(|tmp| {
        write_raw_entry_with_timestamp(tmp, "h1", "f1", &["a"], "not-a-date");

        let entry = cache::load_estimate("h1", "f1", &["a".to_string()])
            .expect("load")
            .expect("entry should exist");
        assert!(
            !cache::is_cache_entry_fresh(&entry, std::time::Duration::from_secs(3600)),
            "an entry with an unparseable timestamp must not count as fresh"
        );
    });
}

/// No entry at all means "re-simulate": `load_fresh_estimate` returns None.
#[test]
fn test_load_fresh_estimate_missing_entry() {
    with_temp_home(|_tmp| {
        let fresh = cache::load_fresh_estimate(
            "nope",
            "no_func",
            &[],
            std::time::Duration::from_secs(3600),
        )
        .expect("load fresh on empty cache");
        assert!(fresh.is_none(), "a missing entry must yield None");
    });
}

/// A fresh entry is returned intact by `load_fresh_estimate`.
#[test]
fn test_load_fresh_estimate_returns_fresh_entry() {
    with_temp_home(|tmp| {
        let now = chrono::Utc::now().to_rfc3339();
        write_raw_entry_with_timestamp(tmp, "h1", "f1", &["a"], &now);

        let fresh = cache::load_fresh_estimate(
            "h1",
            "f1",
            &["a".to_string()],
            std::time::Duration::from_secs(3600),
        )
        .expect("load fresh")
        .expect("fresh entry should be returned");
        assert_eq!(fresh.wasm_hash, "h1");
        assert_eq!(fresh.ledger, 7);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Estimate history (#321)
// ─────────────────────────────────────────────────────────────────────────

/// Every save appends to the history log, newest first, and the log is
/// scoped to the requested `(wasm_hash, function)` pair.
#[test]
fn test_load_estimate_history_newest_first_and_limited() {
    with_temp_home(|_tmp| {
        for (ledger, fee) in [(10u32, 100i64), (20, 200), (30, 300)] {
            cache::save_estimate("h", "f", &[], "testnet", ledger, fee, 1, 1, None, true)
                .expect("save");
        }
        // A different function must not leak into this function's history.
        cache::save_estimate("h", "other", &[], "testnet", 40, 400, 1, 1, None, true)
            .expect("save other");

        let all = cache::load_estimate_history("h", "f", 5).expect("history");
        assert_eq!(all.len(), 3, "three runs of `f` should be recorded");
        assert_eq!(all[0].total_stroops, 300, "newest run first");
        assert_eq!(all[1].total_stroops, 200);
        assert_eq!(all[2].total_stroops, 100);
        assert_eq!(all[0].ledger, 30);
        assert_eq!(all[0].function, "f");

        let limited = cache::load_estimate_history("h", "f", 2).expect("history");
        assert_eq!(limited.len(), 2, "limit must cap the result");
        assert_eq!(limited[0].total_stroops, 300);
        assert_eq!(limited[1].total_stroops, 200);

        let zero = cache::load_estimate_history("h", "f", 0).expect("history");
        assert!(zero.is_empty(), "a zero limit yields no rows");

        let missing = cache::load_estimate_history("h", "missing", 5).expect("history");
        assert!(missing.is_empty(), "unknown function has no history");
    });
}

/// A single save records exactly one history entry (distinct from the
/// upserting `estimates` table).
#[test]
fn test_save_estimate_appends_single_history_entry() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h", "f", &[], "testnet", 1, 100, 10, 5, None, true).expect("save");
        let history = cache::load_estimate_history("h", "f", 5).expect("history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].cpu_instructions, 10);
        assert_eq!(history[0].memory_bytes, 5);
        assert_eq!(history[0].network, "testnet");
    });
}

/// An expired entry is treated as a miss even though the file exists: the
/// caller must re-simulate.
#[test]
fn test_load_fresh_estimate_expired_returns_none() {
    with_temp_home(|tmp| {
        let two_hours_ago = (chrono::Utc::now() - chrono::TimeDelta::hours(2)).to_rfc3339();
        write_raw_entry_with_timestamp(tmp, "h1", "f1", &["a"], &two_hours_ago);

        // The entry exists and loads fine...
        let loaded = cache::load_estimate("h1", "f1", &["a".to_string()])
            .expect("load")
            .expect("entry should exist");
        assert_eq!(loaded.ledger, 7);

        // ...but it is not fresh under a 1h TTL.
        let fresh = cache::load_fresh_estimate(
            "h1",
            "f1",
            &["a".to_string()],
            std::time::Duration::from_secs(3600),
        )
        .expect("load fresh");
        assert!(fresh.is_none(), "an expired entry must yield None");
    });
}

/// The export envelope carries a schema version, an export timestamp, and
/// the estimate records — everything a backup needs to be self-describing.
#[test]
fn test_export_cache_envelope_shape() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true).expect("save");

        let export = cache::export_cache(None).expect("export");
        assert_eq!(
            export.schema_version,
            cache::CACHE_EXPORT_SCHEMA_VERSION,
            "envelope should stamp the export schema version"
        );
        assert!(
            chrono::DateTime::parse_from_rfc3339(&export.exported_at).is_ok(),
            "exported_at should be RFC-3339: {}",
            export.exported_at
        );
        assert!(export.network.is_none(), "unfiltered export has no network");
        assert_eq!(export.estimates.len(), 1, "should export the seeded entry");
        assert_eq!(export.estimates[0].function, "f1");
    });
}

/// A network filter restricts the export to that network's estimates.
#[test]
fn test_export_cache_filters_by_network() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f_testnet", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("testnet save");
        cache::save_estimate(
            "h1",
            "f_mainnet",
            &[],
            "mainnet",
            2,
            200,
            20,
            10,
            None,
            true,
        )
        .expect("mainnet save");

        let testnet = cache::export_cache(Some("testnet")).expect("export testnet");
        assert_eq!(testnet.network, Some("testnet".to_string()));
        assert_eq!(testnet.estimates.len(), 1, "should export only testnet");
        assert_eq!(testnet.estimates[0].function, "f_testnet");

        let all = cache::export_cache(None).expect("export all");
        assert!(all.network.is_none());
        assert_eq!(
            all.estimates.len(),
            2,
            "unfiltered export should carry both"
        );

        let empty = cache::export_cache(Some("futurenet")).expect("export futurenet");
        assert_eq!(empty.network, Some("futurenet".to_string()));
        assert!(
            empty.estimates.is_empty(),
            "unknown network exports nothing"
        );
    });
}

/// The envelope round-trips through JSON — the shape backups are stored in.
#[test]
fn test_export_cache_json_round_trip() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true).expect("save");

        let export = cache::export_cache(Some("testnet")).expect("export");
        let json = serde_json::to_string_pretty(&export).expect("serialize");
        let parsed: cache::CacheExport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.schema_version, cache::CACHE_EXPORT_SCHEMA_VERSION);
        assert_eq!(parsed.network, Some("testnet".to_string()));
        assert_eq!(parsed.estimates.len(), 1);
        assert_eq!(parsed.estimates[0].total_stroops, 100);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────
// SQLite storage backend: indexes & legacy JSON migration (#334)
// ─────────────────────────────────────────────────────────────────────────

/// The SHA-256 hex of the concatenated args, matching the library's cache
/// key derivation.
fn args_hash_of(args: &[&str]) -> String {
    let mut hasher = sha2::Sha256::new();
    for arg in args {
        hasher.update(arg.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Path of the data directory under a temp HOME.
fn data_dir(tmp: &Path) -> PathBuf {
    tmp.join(".soroban-cost-estimator")
}

/// Write a legacy (pre-SQLite) cache entry JSON file into
/// `~/.soroban-cost-estimator/cache/`.
fn write_legacy_cache_file(tmp: &Path, name: &str, contents: &str) {
    let dir = data_dir(tmp).join("cache");
    std::fs::create_dir_all(&dir).expect("create legacy cache dir");
    std::fs::write(dir.join(name), contents).expect("write legacy cache file");
}

/// SQLite index names present on the `estimates` table.
fn estimate_index_names(conn: &rusqlite::Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'estimates'")
        .expect("prepare index query");
    stmt.query_map([], |row| row.get::<_, String>(0))
        .expect("query indexes")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect indexes")
}

#[test]
fn test_sqlite_schema_creates_lookup_indexes() {
    with_temp_home(|tmp| {
        let dir = data_dir(tmp);
        std::fs::create_dir_all(&dir).expect("create data dir");
        let conn = rusqlite::Connection::open(dir.join("cache.db")).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        let names = estimate_index_names(&conn);
        for expected in [
            "idx_estimates_lookup",
            "idx_estimates_network_created",
            "idx_estimates_key_net_created",
            "idx_estimates_last_accessed",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "expected index {expected} on estimates; found {names:?}"
            );
        }
    });
}

#[test]
fn test_legacy_json_cache_is_imported_transparently() {
    with_temp_home(|tmp| {
        let legacy = format!(
            r#"{{
  "wasm_hash": "abc",
  "function": "f",
  "args_hash": "{}",
  "network": "testnet",
  "ledger": 7,
  "total_stroops": 1234,
  "cpu_instructions": 10,
  "memory_bytes": 5,
  "timestamp": "2026-01-01T00:00:00Z"
}}"#,
            args_hash_of(&[])
        );
        write_legacy_cache_file(tmp, "abc-f.json", &legacy);

        // No explicit migration call: opening the cache does it transparently.
        let loaded = cache::load_estimate("abc", "f", &[])
            .expect("load migrated entry")
            .expect("legacy entry should have been imported");
        assert_eq!(loaded.total_stroops, 1234);
        assert_eq!(loaded.ledger, 7);
        assert_eq!(loaded.version, cache::CACHE_SCHEMA_VERSION);
        assert_eq!(loaded.execution_duration_ms, 0);
        assert_eq!(loaded.duration_ms(), None);
        assert!(loaded.success);

        // The legacy directory is retired once fully imported.
        assert!(
            !data_dir(tmp).join("cache").exists(),
            "legacy JSON cache dir should be removed after a full import"
        );
    });
}

#[test]
fn test_legacy_json_cache_keeps_unparseable_files() {
    with_temp_home(|tmp| {
        let valid = format!(
            r#"{{"wasm_hash":"abc","function":"f","args_hash":"{}","network":"testnet","ledger":1,"total_stroops":1,"cpu_instructions":1,"memory_bytes":1,"timestamp":"2026-01-01T00:00:00Z"}}"#,
            args_hash_of(&[])
        );
        write_legacy_cache_file(tmp, "good.json", &valid);
        write_legacy_cache_file(tmp, "bad.json", "{ not valid json");

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert_eq!(estimates.len(), 1, "one valid entry should be imported");

        let legacy_dir = data_dir(tmp).join("cache");
        assert!(
            legacy_dir.join("bad.json.rejected").exists(),
            "unparseable files must be kept under a .rejected extension"
        );
        assert!(
            !legacy_dir.join("bad.json").exists(),
            "the original unparseable file should no longer be retried"
        );
    });
}

#[test]
fn test_legacy_json_entry_from_future_version_is_kept() {
    with_temp_home(|tmp| {
        let future = format!(
            r#"{{"schema_version":{},"wasm_hash":"abc","function":"f","args_hash":"{}","network":"testnet","ledger":1,"total_stroops":1,"cpu_instructions":1,"memory_bytes":1,"timestamp":"2026-01-01T00:00:00Z"}}"#,
            cache::CACHE_SCHEMA_VERSION + 5,
            args_hash_of(&[])
        );
        write_legacy_cache_file(tmp, "future.json", &future);

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert!(estimates.is_empty(), "future entries must not be imported");
        assert!(
            data_dir(tmp).join("cache").join("future.json").exists(),
            "a future-schema legacy file must be preserved, not deleted"
        );
    });
}

#[test]
fn test_pre_v3_database_gains_last_accessed_column() {
    with_temp_home(|tmp| {
        let dir = data_dir(tmp);
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");

        // Recreate the exact pre-v3 table shape: no `last_accessed` column
        // and none of the secondary indexes.
        {
            let conn = rusqlite::Connection::open(&db).expect("open cache db");
            conn.execute_batch(
                "CREATE TABLE estimates (\n                     version INTEGER NOT NULL,\n                     wasm_hash TEXT NOT NULL,\n                     function TEXT NOT NULL,\n                     args_hash TEXT NOT NULL,\n                     network TEXT NOT NULL,\n                     ledger INTEGER NOT NULL,\n                     total_stroops INTEGER NOT NULL,\n                     cpu_instructions INTEGER NOT NULL,\n                     memory_bytes INTEGER NOT NULL,\n                     timestamp TEXT NOT NULL,\n                     duration_ms INTEGER,\n                     success INTEGER NOT NULL DEFAULT 1,\n                     PRIMARY KEY (wasm_hash, function, args_hash)\n                 );",
            )
            .expect("create legacy table");
            conn.execute(
                "INSERT INTO estimates (version, wasm_hash, function, args_hash, network, ledger, total_stroops, cpu_instructions, memory_bytes, timestamp) \
                 VALUES (2, 'h', 'f', ?1, 'testnet', 1, 100, 10, 5, '2026-01-01T00:00:00Z')",
                [args_hash_of(&[])],
            )
            .expect("insert legacy row");
        }

        // Opening through the library migrates the table in place.
        let loaded = cache::load_estimate("h", "f", &[])
            .expect("load")
            .expect("row should survive the table migration");
        assert_eq!(loaded.version, cache::CACHE_SCHEMA_VERSION);
        assert_eq!(loaded.total_stroops, 100);

        let conn = rusqlite::Connection::open(&db).expect("reopen cache db");
        let has_column = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(estimates)")
                .expect("pragma");
            let names: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .expect("query columns")
                .collect::<Result<_, _>>()
                .expect("collect columns");
            names.iter().any(|n| n == "last_accessed")
        };
        assert!(has_column, "pre-v3 cache should gain last_accessed");
        assert!(
            !estimate_index_names(&conn).is_empty(),
            "pre-v3 cache should gain the lookup indexes"
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Schema versioning & migration (#341)
// ─────────────────────────────────────────────────────────────────────────

/// An *unversioned* legacy entry (v0) is migrated to the current schema and
/// gets the conservative defaults for fields it predates.
#[test]
fn test_v0_unversioned_entry_migrates_to_current_schema() {
    with_temp_home(|tmp| {
        write_raw_entry(tmp, "legacy0", "old_func", &["a"], Some(0), 3);

        let loaded = cache::load_estimate("legacy0", "old_func", &["a".to_string()])
            .expect("v0 entry should load")
            .expect("entry should exist");
        assert_eq!(loaded.version, cache::CACHE_SCHEMA_VERSION);
        assert_eq!(loaded.ledger, 3);
        assert_eq!(loaded.execution_duration_ms, 0, "v0 has no duration data");
        assert_eq!(loaded.duration_ms(), None, "v0 has no duration data");
        assert!(loaded.success, "v0 entries were always successful");
    });
}

/// A v2 entry moves to v3 without losing the fields v2 introduced.
#[test]
fn test_v2_entry_migrates_to_v3_preserving_fields() {
    let entry = cache::CachedEstimate {
        version: cache::DURATION_SCHEMA_VERSION,
        wasm_hash: "abc".to_string(),
        function: "f".to_string(),
        args_hash: "def".to_string(),
        network: "testnet".to_string(),
        ledger: 9,
        total_stroops: 4_200,
        cpu_instructions: 111,
        memory_bytes: 22,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        execution_duration_ms: 1_234,
        success: false,
        network_passphrase: None,
        core_version: None,
        io: None,
    };
    let migrated = cache::migrate_to_latest(entry).expect("migrate v2 to v3");
    assert_eq!(migrated.version, cache::CACHE_SCHEMA_VERSION);
    assert_eq!(migrated.execution_duration_ms, 1_234);
    assert_eq!(migrated.duration_ms(), Some(1_234));
    assert!(!migrated.success, "a recorded failure must be preserved");
    assert_eq!(migrated.total_stroops, 4_200);
}

/// Serialized entries carry the `schema_version` key, and the pre-rename
/// `version` key is still accepted on read.
#[test]
fn test_schema_version_serialization_key() {
    let entry = cache::CachedEstimate {
        version: cache::CACHE_SCHEMA_VERSION,
        wasm_hash: "abc".to_string(),
        function: "f".to_string(),
        args_hash: "def".to_string(),
        network: "testnet".to_string(),
        ledger: 1,
        total_stroops: 1,
        cpu_instructions: 1,
        memory_bytes: 1,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        execution_duration_ms: 0,
        success: true,
        network_passphrase: None,
        core_version: None,
        io: None,
    };
    let json = serde_json::to_string(&entry).expect("serialize");
    assert!(
        json.contains("\"schema_version\""),
        "serialized cache entries must carry schema_version: {json}"
    );

    let with_legacy_key = json.replace("\"schema_version\"", "\"version\"");
    let parsed: cache::CachedEstimate =
        serde_json::from_str(&with_legacy_key).expect("legacy `version` key must deserialize");
    assert_eq!(parsed.version, cache::CACHE_SCHEMA_VERSION);
}

/// An entry with no `version`/`schema_version` key at all (the true v0 JSON
/// shape) deserializes to the initial schema version and migrates forward.
#[test]
fn test_unversioned_json_deserializes_and_migrates() {
    let json = r#"{
        "wasm_hash": "abc",
        "function": "f",
        "args_hash": "def",
        "network": "testnet",
        "ledger": 5,
        "total_stroops": 10,
        "cpu_instructions": 1,
        "memory_bytes": 1,
        "timestamp": "2026-01-01T00:00:00Z"
    }"#;
    let parsed: cache::CachedEstimate = serde_json::from_str(json).expect("v0 JSON");
    assert_eq!(parsed.version, cache::INITIAL_SCHEMA_VERSION);

    let migrated = cache::migrate_to_latest(parsed).expect("migrate");
    assert_eq!(migrated.version, cache::CACHE_SCHEMA_VERSION);
}

/// The upgrade hint names the entry's version so the user knows what to do.
#[test]
fn test_future_schema_error_names_version_and_remedy() {
    let entry = cache::CachedEstimate {
        version: cache::CACHE_SCHEMA_VERSION + 1,
        wasm_hash: "abc".to_string(),
        function: "f".to_string(),
        args_hash: "def".to_string(),
        network: "testnet".to_string(),
        ledger: 1,
        total_stroops: 1,
        cpu_instructions: 1,
        memory_bytes: 1,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        execution_duration_ms: 0,
        success: true,
        network_passphrase: None,
        core_version: None,
        io: None,
    };
    let err = cache::migrate_to_latest(entry).expect_err("future version must be rejected");
    let message = err.to_string();
    assert!(message.contains("newer"), "{message}");
    assert!(
        message.contains("upgrade"),
        "should include an upgrade hint: {message}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// LRU eviction (#333)
// ─────────────────────────────────────────────────────────────────────────

/// [`cache::CacheLimits`] with only an entry quota (byte quota disabled).
fn entry_limits(max_entries: usize) -> cache::CacheLimits {
    cache::CacheLimits {
        max_bytes: 0,
        max_entries,
    }
}

/// Save one estimate with explicit limits (keeps eviction tests independent
/// of the process-wide CLI configuration).
fn save_with(
    limits: cache::CacheLimits,
    wasm_hash: &str,
    function: &str,
    ledger: u32,
) -> soroban_cost_estimator::error::AppResult<()> {
    cache::save_estimate_with_limits(
        wasm_hash,
        function,
        &[],
        "testnet",
        ledger,
        1_000,
        100,
        50,
        None,
        None,
        true,
        limits,
    )
}

#[test]
fn test_eviction_by_entry_quota_keeps_cache_bounded() {
    with_temp_home(|_tmp| {
        let limits = entry_limits(5);
        for i in 0..10 {
            save_with(limits, "h", &format!("f{i}"), i).expect("save");
        }

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert!(
            estimates.len() <= 5,
            "entry quota must be respected, got {} entries",
            estimates.len()
        );
        assert!(!estimates.is_empty(), "eviction must not empty the cache");
        assert!(
            estimates.iter().any(|e| e.function == "f9"),
            "the newest entry must survive eviction: {estimates:?}"
        );
        assert!(
            !estimates.iter().any(|e| e.function == "f0"),
            "the oldest entry must be evicted first: {estimates:?}"
        );
    });
}

#[test]
fn test_eviction_protects_most_recently_accessed_entry() {
    with_temp_home(|_tmp| {
        let limits = entry_limits(3);
        let pause = || std::thread::sleep(std::time::Duration::from_millis(2));

        save_with(limits, "h", "f0", 0).expect("save f0");
        pause();
        save_with(limits, "h", "f1", 1).expect("save f1");
        pause();
        save_with(limits, "h", "f2", 2).expect("save f2");
        pause();

        // Reading f0 makes it the most recently accessed entry even though
        // it is the oldest by creation time.
        cache::load_estimate("h", "f0", &[])
            .expect("load f0")
            .expect("f0 exists");
        pause();

        // Pushing past the quota evicts f1 and f2 (LRU order), not f0.
        save_with(limits, "h", "f3", 3).expect("save f3");

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert!(
            estimates.iter().any(|e| e.function == "f0"),
            "recently accessed f0 must survive: {estimates:?}"
        );
        assert!(
            estimates.iter().any(|e| e.function == "f3"),
            "the newest entry must survive: {estimates:?}"
        );
        assert!(
            !estimates.iter().any(|e| e.function == "f1"),
            "the least-recently-accessed entry must be evicted: {estimates:?}"
        );
    });
}

#[test]
fn test_eviction_by_byte_quota_stays_under_limit() {
    with_temp_home(|_tmp| {
        // A quota comfortably above SQLite's structural floor (page 1 plus a
        // root page per table/index) but well below the size of 600 rows.
        let max_bytes = 96 * 1024;
        let limits = cache::CacheLimits {
            max_bytes,
            max_entries: 0,
        };
        const SAVED: u32 = 600;

        for i in 0..SAVED {
            save_with(limits, &format!("hash-{i:04}"), "f", i).expect("save");
        }

        let stats = cache::cache_stats().expect("stats");
        assert!(
            stats.total_entries < SAVED as usize,
            "byte quota should have evicted entries, kept {}",
            stats.total_entries
        );
        assert!(
            stats.live_bytes <= max_bytes,
            "live cache size {} must stay under the {max_bytes} byte quota",
            stats.live_bytes,
        );
    });
}

/// A byte quota smaller than the database's structural floor must not wipe
/// the cache: eviction stops once deleting rows stops reducing the size.
#[test]
fn test_unattainable_byte_quota_does_not_empty_cache() {
    with_temp_home(|_tmp| {
        // 1 byte is below any SQLite database's floor.
        let limits = cache::CacheLimits {
            max_bytes: 1,
            max_entries: 0,
        };
        for i in 0..20 {
            save_with(limits, &format!("hash-{i:04}"), "f", i).expect("save");
        }

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert!(
            !estimates.is_empty(),
            "an unattainable byte quota must not evict every entry"
        );
    });
}

#[test]
fn test_unbounded_limits_disable_eviction() {
    with_temp_home(|_tmp| {
        for i in 0..50 {
            save_with(cache::CacheLimits::UNBOUNDED, "h", &format!("f{i}"), i).expect("save");
        }
        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert_eq!(estimates.len(), 50, "unbounded limits must never evict");
    });
}

#[test]
fn test_evict_lru_is_noop_under_default_limits() {
    with_temp_home(|_tmp| {
        for i in 0..5 {
            cache::save_estimate(
                "h",
                &format!("f{i}"),
                &[],
                "testnet",
                i,
                100,
                10,
                5,
                None,
                true,
            )
            .expect("save");
        }
        let evicted = cache::evict_lru().expect("prune");
        assert_eq!(evicted, 0, "a small cache is within the default quota");
    });
}

#[test]
fn test_concurrent_saves_respect_entry_quota() {
    with_temp_home(|_tmp| {
        let limits = entry_limits(10);
        let handles: Vec<_> = (0..4)
            .map(|t| {
                std::thread::spawn(move || {
                    for j in 0..20 {
                        save_with(limits, &format!("hash-{t}"), &format!("f{j}"), j)
                            .expect("concurrent save");
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("concurrent save thread panicked");
        }

        let estimates = cache::list_cached_estimates("testnet").expect("list");
        assert!(
            estimates.len() <= 10,
            "quota must hold under concurrent writers, got {}",
            estimates.len()
        );
        let statuses = cache::verify_cache().expect("verify after concurrent eviction");
        assert!(
            statuses.iter().all(|s| s.valid),
            "concurrent eviction must not corrupt entries: {statuses:?}"
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// cache_stats (issue #281)
// ─────────────────────────────────────────────────────────────────────────

/// `cache_stats` on a fresh cache reports zero entries, no timestamps, an
/// empty per-network breakdown, and no disk error.
#[test]
fn test_cache_stats_empty_cache() {
    with_temp_home(|_tmp| {
        let stats = cache::cache_stats().expect("stats on empty cache");
        assert_eq!(stats.total_entries, 0, "fresh cache should have 0 entries");
        assert!(
            stats.oldest_entry.is_none(),
            "no oldest entry on an empty cache"
        );
        assert!(
            stats.newest_entry.is_none(),
            "no newest entry on an empty cache"
        );
        assert!(
            stats.per_network.is_empty(),
            "empty cache should have no per-network breakdown"
        );
    });
}

/// `cache_stats` aggregates every cached estimate across networks: total
/// count, oldest/newest timestamps, per-network breakdown, and disk usage.
#[test]
fn test_cache_stats_reports_populated_cache() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h1", "f1", &[], "testnet", 1, 100, 10, 5, None, true)
            .expect("save testnet f1");
        cache::save_estimate("h2", "f2", &[], "mainnet", 2, 200, 20, 10, None, true)
            .expect("save mainnet f2");

        let stats = cache::cache_stats().expect("stats on populated cache");
        assert_eq!(stats.total_entries, 2, "both entries should be counted");
        assert!(stats.disk_bytes > 0, "the SQLite file should occupy space");
        assert!(
            stats.oldest_entry.is_some(),
            "an oldest timestamp should be reported"
        );
        assert!(
            stats.newest_entry.is_some(),
            "a newest timestamp should be reported"
        );

        let count = |network: &str| {
            stats
                .per_network
                .iter()
                .find(|(name, _)| name == network)
                .map(|(_, count)| *count)
        };
        assert_eq!(
            count("mainnet"),
            Some(1),
            "mainnet breakdown should count 1"
        );
        assert_eq!(
            count("testnet"),
            Some(1),
            "testnet breakdown should count 1"
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// query_cache and CacheFilter tests (issue #335)
// ─────────────────────────────────────────────────────────────────────────

fn seed_query_test_entry(
    home: &Path,
    wasm_hash: &str,
    function: &str,
    args: &[&str],
    network: &str,
    total_stroops: i64,
    cpu_instructions: u64,
    timestamp: &str,
) {
    let dir = home.join(".soroban-cost-estimator");
    std::fs::create_dir_all(&dir).expect("create data dir");
    let db = dir.join("cache.db");
    let conn = rusqlite::Connection::open(&db).expect("open cache db");
    cache::ensure_cache_schema(&conn).expect("ensure cache schema");

    let mut hasher = sha2::Sha256::new();
    for a in args {
        hasher.update(a.as_bytes());
    }
    let args_hash = hex::encode(hasher.finalize());

    conn.execute(
        "INSERT OR REPLACE INTO estimates \
         (version, wasm_hash, function, args_hash, network, ledger, total_stroops, cpu_instructions, memory_bytes, timestamp, duration_ms, success) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            cache::CACHE_SCHEMA_VERSION as i64,
            wasm_hash,
            function,
            args_hash,
            network,
            100i64,
            total_stroops,
            cpu_instructions as i64,
            1024i64,
            timestamp,
            Some(50i64),
            1i64,
        ],
    )
    .expect("insert test estimate");
}

#[test]
fn test_query_cache_wasm_hash_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111111111111111111111111111111111111111111111111111111111111111",
            "func_a",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "2222222222222222222222222222222222222222222222222222222222222222",
            "func_b",
            &[],
            "testnet",
            200_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );

        // Exact match
        let filter = cache::CacheFilter {
            wasm_hash: Some(
                "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            ),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].function, "func_a");

        // Case-insensitive prefix match
        let filter_prefix = cache::CacheFilter {
            wasm_hash: Some("2222".to_string()),
            ..Default::default()
        };
        let res_prefix = cache::query_cache(&filter_prefix).expect("query");
        assert_eq!(res_prefix.len(), 1);
        assert_eq!(res_prefix[0].function, "func_b");

        // Non-matching hash returns empty
        let filter_none = cache::CacheFilter {
            wasm_hash: Some(
                "3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            ),
            ..Default::default()
        };
        let res_none = cache::query_cache(&filter_none).expect("query");
        assert!(res_none.is_empty());
    });
}

#[test]
fn test_query_cache_function_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
            "transfer",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000",
            "approve",
            &[],
            "testnet",
            200_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );

        let filter = cache::CacheFilter {
            function: Some("transfer".to_string()),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].function, "transfer");

        // Exact match semantics: substring should NOT match
        let filter_sub = cache::CacheFilter {
            function: Some("trans".to_string()),
            ..Default::default()
        };
        let res_sub = cache::query_cache(&filter_sub).expect("query");
        assert!(res_sub.is_empty());
    });
}

#[test]
fn test_query_cache_network_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
            "fn1",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000",
            "fn2",
            &[],
            "mainnet",
            200_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );

        let filter = cache::CacheFilter {
            network: Some("testnet".to_string()),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].network, "testnet");

        let filter_main = cache::CacheFilter {
            network: Some("mainnet".to_string()),
            ..Default::default()
        };
        let res_main = cache::query_cache(&filter_main).expect("query");
        assert_eq!(res_main.len(), 1);
        assert_eq!(res_main[0].network, "mainnet");
    });
}

#[test]
fn test_query_cache_min_fee_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "low_fee",
            &[],
            "testnet",
            50_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "2222000022220000222200002222000022220000222200002222000022220000",
            "mid_fee",
            &[],
            "testnet",
            100_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "3333000033330000333300003333000033330000333300003333000033330000",
            "high_fee",
            &[],
            "testnet",
            200_000,
            30_000,
            "2026-01-03T00:00:00Z",
        );

        let filter = cache::CacheFilter {
            min_fee: Some(100_000),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.total_stroops >= 100_000));
    });
}

#[test]
fn test_query_cache_max_fee_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "low_fee",
            &[],
            "testnet",
            50_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "2222000022220000222200002222000022220000222200002222000022220000",
            "mid_fee",
            &[],
            "testnet",
            100_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "3333000033330000333300003333000033330000333300003333000033330000",
            "high_fee",
            &[],
            "testnet",
            200_000,
            30_000,
            "2026-01-03T00:00:00Z",
        );

        let filter = cache::CacheFilter {
            max_fee: Some(100_000),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.total_stroops <= 100_000));
    });
}

#[test]
fn test_query_cache_fee_range_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "low_fee",
            &[],
            "testnet",
            50_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "2222000022220000222200002222000022220000222200002222000022220000",
            "mid_fee",
            &[],
            "testnet",
            100_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "3333000033330000333300003333000033330000333300003333000033330000",
            "high_fee",
            &[],
            "testnet",
            200_000,
            30_000,
            "2026-01-03T00:00:00Z",
        );

        let filter = cache::CacheFilter {
            min_fee: Some(75_000),
            max_fee: Some(150_000),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].function, "mid_fee");
        assert_eq!(res[0].total_stroops, 100_000);
    });
}

#[test]
fn test_query_cache_since_filter() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "early",
            &[],
            "testnet",
            100_000,
            10_000,
            "2025-12-31T23:59:59Z",
        );
        seed_query_test_entry(
            tmp,
            "2222000022220000222200002222000022220000222200002222000022220000",
            "on_boundary",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "3333000033330000333300003333000033330000333300003333000033330000",
            "late",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-02T12:00:00Z",
        );

        let since_dt = cache::parse_since_timestamp("2026-01-01").expect("parse since");
        let filter = cache::CacheFilter {
            since: Some(since_dt),
            ..Default::default()
        };
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|e| e.function != "early"));
    });
}

#[test]
fn test_query_cache_multiple_filters_and_semantics() {
    with_temp_home(|tmp| {
        // Entry 1: matches ALL criteria
        seed_query_test_entry(
            tmp,
            "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000",
            "transfer",
            &["arg1"],
            "testnet",
            150_000,
            10_000,
            "2026-02-01T12:00:00Z",
        );
        // Entry 2: wrong function
        seed_query_test_entry(
            tmp,
            "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000",
            "mint",
            &["arg2"],
            "testnet",
            150_000,
            10_000,
            "2026-02-01T12:00:00Z",
        );
        // Entry 3: wrong network
        seed_query_test_entry(
            tmp,
            "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000",
            "transfer",
            &["arg3"],
            "mainnet",
            150_000,
            10_000,
            "2026-02-01T12:00:00Z",
        );
        // Entry 4: fee too low
        seed_query_test_entry(
            tmp,
            "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000",
            "transfer",
            &["arg4"],
            "testnet",
            50_000,
            10_000,
            "2026-02-01T12:00:00Z",
        );
        // Entry 5: timestamp too early
        seed_query_test_entry(
            tmp,
            "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000",
            "transfer",
            &["arg5"],
            "testnet",
            150_000,
            10_000,
            "2025-12-01T12:00:00Z",
        );

        let since_dt = cache::parse_since_timestamp("2026-01-01").expect("parse since");
        let filter = cache::CacheFilter {
            wasm_hash: Some("deadbeef".to_string()),
            function: Some("transfer".to_string()),
            network: Some("testnet".to_string()),
            min_fee: Some(100_000),
            max_fee: Some(200_000),
            since: Some(since_dt),
            to: None,
        };

        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(
            res.len(),
            1,
            "only entry satisfying all AND filters should match"
        );
        assert_eq!(res[0].function, "transfer");
        assert_eq!(res[0].network, "testnet");
        assert_eq!(res[0].total_stroops, 150_000);
    });
}

#[test]
fn test_query_cache_no_filters_returns_all() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "fn1",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );
        seed_query_test_entry(
            tmp,
            "2222000022220000222200002222000022220000222200002222000022220000",
            "fn2",
            &[],
            "mainnet",
            200_000,
            20_000,
            "2026-01-02T00:00:00Z",
        );

        let filter = cache::CacheFilter::default();
        let res = cache::query_cache(&filter).expect("query");
        assert_eq!(
            res.len(),
            2,
            "no filters should return all cached estimates across all networks"
        );
    });
}

#[test]
fn test_query_cache_invalid_inputs_return_app_error() {
    // Invalid network
    let filter_net = cache::CacheFilter {
        network: Some("nonexistent_network".to_string()),
        ..Default::default()
    };
    assert!(cache::query_cache(&filter_net).is_err());

    // Negative min fee
    let filter_fee_neg = cache::CacheFilter {
        min_fee: Some(-10),
        ..Default::default()
    };
    assert!(cache::query_cache(&filter_fee_neg).is_err());

    // Invalid fee range (min > max)
    let filter_fee_range = cache::CacheFilter {
        min_fee: Some(200_000),
        max_fee: Some(100_000),
        ..Default::default()
    };
    assert!(cache::query_cache(&filter_fee_range).is_err());

    // Invalid wasm hash (non-hex)
    let filter_hash = cache::CacheFilter {
        wasm_hash: Some("not_hexadecimal!".to_string()),
        ..Default::default()
    };
    assert!(cache::query_cache(&filter_hash).is_err());

    // Empty function name
    let filter_fn = cache::CacheFilter {
        function: Some(String::new()),
        ..Default::default()
    };
    assert!(cache::query_cache(&filter_fn).is_err());

    // Invalid date parsing
    assert!(cache::parse_since_timestamp("invalid-date-string").is_err());
    assert!(cache::parse_to_timestamp("2026-99-99").is_err());
}

#[test]
fn test_query_cache_read_only() {
    with_temp_home(|tmp| {
        seed_query_test_entry(
            tmp,
            "1111000011110000111100001111000011110000111100001111000011110000",
            "fn1",
            &[],
            "testnet",
            100_000,
            10_000,
            "2026-01-01T00:00:00Z",
        );

        let filter = cache::CacheFilter::default();
        let _ = cache::query_cache(&filter).expect("first query");
        let _ = cache::query_cache(&filter).expect("second query");

        // Verify count and entry remains unaltered
        let stats = cache::cache_stats().expect("cache stats");
        assert_eq!(stats.total_entries, 1);
        let entry = cache::load_estimate(
            "1111000011110000111100001111000011110000111100001111000011110000",
            "fn1",
            &[],
        )
        .expect("load")
        .expect("exists");
        assert_eq!(entry.timestamp, "2026-01-01T00:00:00Z");
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Ledger I/O footprint (#278)
// ─────────────────────────────────────────────────────────────────────────

fn sample_io() -> cache::IoFootprint {
    cache::IoFootprint {
        read_entries: 3,
        write_entries: 2,
        read_bytes: 40,
        write_bytes: 136,
    }
}

#[test]
fn test_save_estimate_with_io_roundtrip() {
    with_temp_home(|_tmp| {
        let io = sample_io();
        cache::save_estimate_with_io("h", "f", &[], "testnet", 7, 500, 20, 10, io, Some(12), true)
            .expect("save with io");

        let loaded = cache::load_estimate("h", "f", &[])
            .expect("load")
            .expect("entry should exist");
        assert_eq!(loaded.io, Some(io));
        assert_eq!(loaded.execution_duration_ms, 12);
        assert_eq!(loaded.duration_ms(), Some(12));
    });
}

#[test]
fn test_save_estimate_without_io_leaves_footprint_absent() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h", "f", &[], "testnet", 1, 100, 10, 5, None, true).expect("save");

        let loaded = cache::load_estimate("h", "f", &[])
            .expect("load")
            .expect("entry should exist");
        assert!(
            loaded.io.is_none(),
            "save_estimate must not invent an I/O footprint"
        );
    });
}

/// Re-running the same key replaces the stored footprint along with the other
/// metrics.
#[test]
fn test_save_estimate_with_io_updates_existing_entry() {
    with_temp_home(|_tmp| {
        cache::save_estimate("h", "f", &[], "testnet", 1, 100, 10, 5, None, true).expect("first");

        let io = sample_io();
        cache::save_estimate_with_io("h", "f", &[], "testnet", 2, 200, 20, 10, io, None, true)
            .expect("second");

        let loaded = cache::load_estimate("h", "f", &[])
            .expect("load")
            .expect("entry should exist");
        assert_eq!(loaded.ledger, 2);
        assert_eq!(loaded.io, Some(io));
    });
}

/// `export_cached_estimates` reads every column the row mapping needs,
/// including the ones added after the initial schema.
#[test]
fn test_export_cached_estimates_includes_footprint() {
    with_temp_home(|_tmp| {
        let io = sample_io();
        cache::save_estimate_with_io("h", "f", &[], "testnet", 3, 42, 7, 9, io, Some(5), true)
            .expect("save");

        let exported = cache::export_cached_estimates().expect("export");
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].io, Some(io));
        assert_eq!(exported[0].execution_duration_ms, 5);
        assert_eq!(exported[0].duration_ms(), Some(5));
        assert!(exported[0].success);
    });
}

/// A database written by a build that predates the `io_json` column is
/// upgraded in place when the library opens it, and its rows still load.
#[test]
fn test_legacy_database_without_io_column_still_loads() {
    with_temp_home(|home| {
        let dir = home.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");

        // The schema exactly as it existed before `io_json` was added.
        {
            let conn = rusqlite::Connection::open(&db).expect("open cache db");
            conn.execute_batch(
                "CREATE TABLE estimates (
                    version          INTEGER NOT NULL,
                    wasm_hash        TEXT NOT NULL,
                    function         TEXT NOT NULL,
                    args_hash        TEXT NOT NULL,
                    network          TEXT NOT NULL,
                    ledger           INTEGER NOT NULL,
                    total_stroops    INTEGER NOT NULL,
                    cpu_instructions INTEGER NOT NULL,
                    memory_bytes     INTEGER NOT NULL,
                    timestamp        TEXT NOT NULL,
                    duration_ms      INTEGER,
                    success          INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (wasm_hash, function, args_hash)
                );",
            )
            .expect("create legacy schema");
            conn.execute(
                "INSERT INTO estimates \
                 (version, wasm_hash, function, args_hash, network, ledger, total_stroops, cpu_instructions, memory_bytes, timestamp, success) \
                 VALUES (?1, 'old', 'legacy_fn', ?2, 'testnet', 3, 100, 10, 5, '2026-01-01T00:00:00Z', 1)",
                rusqlite::params![
                    cache::CACHE_SCHEMA_VERSION as i64,
                    args_hash_of(&["a"]),
                ],
            )
            .expect("insert legacy row");
        }

        let loaded = cache::load_estimate("old", "legacy_fn", &["a".to_string()])
            .expect("load legacy entry")
            .expect("legacy entry should exist");
        assert_eq!(loaded.total_stroops, 100);
        assert!(
            loaded.io.is_none(),
            "a pre-io_json row has no footprint to report"
        );
    });
}

// ── Cache listing and filtered deletion (#516, #517) ───────────────────────

/// Seeds one cache entry for the listing and deletion tests.
fn seed_listing_entry(
    wasm_hash: &str,
    function: &str,
    network: &str,
    ledger: u32,
    total_stroops: i64,
) {
    cache::save_estimate(
        wasm_hash,
        function,
        &[],
        network,
        ledger,
        total_stroops,
        100,
        50,
        None,
        true,
    )
    .expect("seed estimate");
}

/// A filter that matches nothing but is still valid.
fn filter_for_network(network: &str) -> cache::CacheFilter {
    cache::CacheFilter {
        network: Some(network.to_string()),
        ..Default::default()
    }
}

#[test]
fn test_list_cached_estimates_paginated_filters_by_network() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("bb22", "approve", "mainnet", 2, 200);

        let listing = cache::list_cached_estimates_paginated(&filter_for_network("testnet"), 50, 0)
            .expect("list estimates");

        assert_eq!(listing.total, 1);
        assert_eq!(listing.entries.len(), 1);
        assert_eq!(listing.entries[0].function, "transfer");
        assert_eq!(listing.entries[0].network, "testnet");
    });
}

#[test]
fn test_list_cached_estimates_paginated_without_network_filter_covers_every_network() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("bb22", "approve", "mainnet", 2, 200);

        let listing = cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 50, 0)
            .expect("list estimates");

        assert_eq!(listing.total, 2);
        assert_eq!(listing.entries.len(), 2);
    });
}

#[test]
fn test_list_cached_estimates_paginated_matches_function_exactly() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("aa11", "approve", "testnet", 2, 200);

        let filter = cache::CacheFilter {
            function: Some("transfer".to_string()),
            ..Default::default()
        };
        let listing =
            cache::list_cached_estimates_paginated(&filter, 50, 0).expect("list estimates");

        assert_eq!(listing.total, 1);
        assert_eq!(listing.entries[0].function, "transfer");
    });
}

#[test]
fn test_list_cached_estimates_paginated_matches_wasm_hash_prefix() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aabbccdd", "transfer", "testnet", 1, 100);
        seed_listing_entry("ffeeddcc", "transfer", "testnet", 2, 200);

        let filter = cache::CacheFilter {
            wasm_hash: Some("aabb".to_string()),
            ..Default::default()
        };
        let listing =
            cache::list_cached_estimates_paginated(&filter, 50, 0).expect("list estimates");

        assert_eq!(listing.total, 1);
        assert_eq!(listing.entries[0].wasm_hash, "aabbccdd");
    });
}

#[test]
fn test_list_cached_estimates_paginated_empty_result() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let filter = cache::CacheFilter {
            function: Some("does-not-exist".to_string()),
            ..Default::default()
        };
        let listing =
            cache::list_cached_estimates_paginated(&filter, 50, 0).expect("list estimates");

        assert_eq!(listing.total, 0);
        assert!(listing.entries.is_empty());
    });
}

#[test]
fn test_list_cached_estimates_paginated_paginates() {
    with_temp_home(|_tmp| {
        for index in 0..5u32 {
            seed_listing_entry("aa11", &format!("fn{index}"), "testnet", index + 1, 100);
        }

        let first = cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 2, 0)
            .expect("list estimates");
        assert_eq!(first.total, 5);
        assert_eq!(first.entries.len(), 2);
        assert_eq!(first.limit, 2);
        assert_eq!(first.offset, 0);

        let second = cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 2, 2)
            .expect("list estimates");
        assert_eq!(second.total, 5);
        assert_eq!(second.entries.len(), 2);
        assert_eq!(second.offset, 2);

        // The pages must not overlap.
        let first_names: Vec<&str> = first.entries.iter().map(|e| e.function.as_str()).collect();
        let second_names: Vec<&str> = second.entries.iter().map(|e| e.function.as_str()).collect();
        for name in &second_names {
            assert!(!first_names.contains(name), "pages overlap on {name}");
        }
    });
}

#[test]
fn test_list_cached_estimates_paginated_caps_the_limit() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let listing =
            cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 100_000, 0)
                .expect("list estimates");
        assert_eq!(listing.limit, cache::MAX_LIST_LIMIT);
    });
}

#[test]
fn test_list_cached_estimates_paginated_rejects_an_invalid_wasm_hash() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let filter = cache::CacheFilter {
            wasm_hash: Some("not-hex".to_string()),
            ..Default::default()
        };
        let error =
            cache::list_cached_estimates_paginated(&filter, 50, 0).expect_err("invalid hash");
        assert!(
            error.to_string().contains("invalid wasm hash filter"),
            "error should explain the bad hash: {error}"
        );
    });
}

#[test]
fn test_list_cached_estimates_paginated_filters_by_time_range() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let now = chrono::Utc::now();
        let inside = cache::CacheFilter {
            since: Some(now - chrono::TimeDelta::hours(1)),
            to: Some(now + chrono::TimeDelta::hours(1)),
            ..Default::default()
        };
        assert_eq!(
            cache::list_cached_estimates_paginated(&inside, 50, 0)
                .expect("list estimates")
                .total,
            1
        );

        let outside = cache::CacheFilter {
            since: Some(now + chrono::TimeDelta::hours(1)),
            ..Default::default()
        };
        assert_eq!(
            cache::list_cached_estimates_paginated(&outside, 50, 0)
                .expect("list estimates")
                .total,
            0
        );
    });
}

#[test]
fn test_save_estimate_with_metadata_roundtrip() {
    with_temp_home(|_tmp| {
        cache::save_estimate_with_metadata(
            "hash_meta",
            "test_fn",
            &[],
            "testnet",
            100,
            250,
            5000,
            1024,
            Some(45),
            true,
            Some("Test SDF Network ; September 2015"),
            Some("v21.1.0"),
        )
        .expect("save with metadata");

        let loaded = cache::load_estimate("hash_meta", "test_fn", &[])
            .expect("load")
            .expect("entry should exist");
        assert_eq!(loaded.execution_duration_ms, 45);
        assert_eq!(loaded.duration_ms(), Some(45));
        assert_eq!(
            loaded.network_passphrase.as_deref(),
            Some("Test SDF Network ; September 2015")
        );
        assert_eq!(loaded.core_version.as_deref(), Some("v21.1.0"));
    });
}

#[test]
fn test_count_cached_estimates_matches_the_listing() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("bb22", "approve", "testnet", 2, 200);
        seed_listing_entry("cc33", "transfer", "mainnet", 3, 300);

        let filter = filter_for_network("testnet");
        assert_eq!(cache::count_cached_estimates(&filter).expect("count"), 2);
        assert_eq!(
            cache::list_cached_estimates_paginated(&filter, 50, 0)
                .expect("list estimates")
                .total,
            2
        );
    });
}

#[test]
fn test_delete_cached_estimates_removes_only_the_matching_entries() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("aa11", "approve", "testnet", 2, 200);
        seed_listing_entry("bb22", "transfer", "mainnet", 3, 300);

        let filter = cache::CacheFilter {
            network: Some("testnet".to_string()),
            function: Some("transfer".to_string()),
            ..Default::default()
        };
        let removed = cache::delete_cached_estimates(&filter).expect("delete estimates");
        assert_eq!(removed, 1);

        // The matching entry is gone, the others survive untouched.
        let remaining =
            cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 50, 0)
                .expect("list estimates");
        assert_eq!(remaining.total, 2);
        let functions: Vec<&str> = remaining
            .entries
            .iter()
            .map(|e| e.function.as_str())
            .collect();
        assert!(functions.contains(&"approve"));
        assert!(functions.contains(&"transfer"));
    });
}

#[test]
fn test_delete_cached_estimates_without_matches_removes_nothing() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let filter = cache::CacheFilter {
            function: Some("does-not-exist".to_string()),
            ..Default::default()
        };
        let removed = cache::delete_cached_estimates(&filter).expect("delete estimates");
        assert_eq!(removed, 0);
        assert_eq!(
            cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 50, 0)
                .expect("list estimates")
                .total,
            1
        );
    });
}

#[test]
fn test_delete_cached_estimates_never_touches_other_networks() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("bb22", "transfer", "mainnet", 2, 200);

        let removed =
            cache::delete_cached_estimates(&filter_for_network("testnet")).expect("delete");
        assert_eq!(removed, 1);

        let mainnet = filter_for_network("mainnet");
        assert_eq!(
            cache::list_cached_estimates_paginated(&mainnet, 50, 0)
                .expect("list estimates")
                .total,
            1
        );
    });
}

#[test]
fn test_delete_cached_estimates_uses_the_same_matching_as_queries() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);
        seed_listing_entry("bb22", "approve", "testnet", 2, 200);

        let filter = cache::CacheFilter {
            function: Some("transfer".to_string()),
            ..Default::default()
        };
        assert_eq!(
            cache::count_cached_estimates(&filter).expect("count"),
            cache::query_cache(&filter).expect("query").len()
        );
        assert_eq!(cache::delete_cached_estimates(&filter).expect("delete"), 1);
    });
}

#[test]
fn test_delete_cached_estimates_rejects_an_invalid_filter() {
    with_temp_home(|_tmp| {
        seed_listing_entry("aa11", "transfer", "testnet", 1, 100);

        let filter = cache::CacheFilter {
            wasm_hash: Some("zz".to_string()),
            ..Default::default()
        };
        let error = cache::delete_cached_estimates(&filter).expect_err("invalid hash");
        assert!(
            error.to_string().contains("invalid wasm hash filter"),
            "error should explain the bad hash: {error}"
        );
    });
}

#[test]
fn test_cached_estimate_backward_compatible_deserialization() {
    // 1. Legacy entry with duration_ms
    let legacy_json = r#"{
        "schema_version": 2,
        "wasm_hash": "h1",
        "function": "f",
        "args_hash": "a1",
        "network": "testnet",
        "ledger": 10,
        "total_stroops": 100,
        "cpu_instructions": 20,
        "memory_bytes": 10,
        "timestamp": "2026-01-01T00:00:00Z",
        "duration_ms": 35,
        "success": true
    }"#;
    let parsed: cache::CachedEstimate =
        serde_json::from_str(legacy_json).expect("deserialize legacy duration_ms");
    assert_eq!(parsed.execution_duration_ms, 35);
    assert_eq!(parsed.network_passphrase, None);
    assert_eq!(parsed.core_version, None);

    // 2. Legacy entry with null duration_ms
    let null_duration_json = r#"{
        "schema_version": 2,
        "wasm_hash": "h1",
        "function": "f",
        "args_hash": "a1",
        "network": "testnet",
        "ledger": 10,
        "total_stroops": 100,
        "cpu_instructions": 20,
        "memory_bytes": 10,
        "timestamp": "2026-01-01T00:00:00Z",
        "duration_ms": null,
        "success": true
    }"#;
    let parsed_null: cache::CachedEstimate =
        serde_json::from_str(null_duration_json).expect("deserialize null duration_ms");
    assert_eq!(parsed_null.execution_duration_ms, 0);

    // 3. New entry with execution_duration_ms, network_passphrase, and core_version
    let new_json = r#"{
        "schema_version": 3,
        "wasm_hash": "h2",
        "function": "f2",
        "args_hash": "a2",
        "network": "testnet",
        "ledger": 12,
        "total_stroops": 120,
        "cpu_instructions": 30,
        "memory_bytes": 15,
        "timestamp": "2026-01-01T00:00:00Z",
        "execution_duration_ms": 78,
        "network_passphrase": "Public Global Stellar Network ; September 2015",
        "core_version": "v21.2.0",
        "success": true
    }"#;
    let parsed_new: cache::CachedEstimate =
        serde_json::from_str(new_json).expect("deserialize new metadata fields");
    assert_eq!(parsed_new.execution_duration_ms, 78);
    assert_eq!(
        parsed_new.network_passphrase.as_deref(),
        Some("Public Global Stellar Network ; September 2015")
    );
    assert_eq!(parsed_new.core_version.as_deref(), Some("v21.2.0"));
}

// ─────────────────────────────────────────────────────────────────────────
// Schema migration: comprehensive tests for all supported transitions
// ─────────────────────────────────────────────────────────────────────────

/// Test helper to create a database at a specific schema version with
/// realistic data.
fn build_database_at_schema_version(tmp: &Path, version: u32, entry_count: usize) {
    let dir = tmp.join(".soroban-cost-estimator");
    std::fs::create_dir_all(&dir).expect("create data dir");
    let db = dir.join("cache.db");
    let conn = rusqlite::Connection::open(&db).expect("open cache db");

    // Ensure the v3 schema exists first (for testing migrations from older
    // versions, we need all columns present in the current schema).
    cache::ensure_cache_schema(&conn).expect("ensure schema");

    // Insert entries at the requested schema version
    for i in 0..entry_count {
        let wasm_hash = format!("hash-v{}-{}", version, i);
        let function = format!("func-v{}-{}", version, i);
        let args_hash = format!("args-{}", i);
        insert_raw_row(
            tmp,
            version,
            &wasm_hash,
            &function,
            &[],
            "testnet",
            (10 + i as u32) * 100,
            &format!("2026-01-{:02}T{:02}:00:00Z", (i % 28) + 1, (i % 24)),
        );
    }
}

/// Schema v1 (initial) entries should load and migrate to v3 without error.
/// v1 entries lack `duration_ms`, `success`, and `last_accessed`.
#[test]
fn test_migrate_v1_to_latest() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, 3);

        // Load entries written at v1; they should migrate to current version
        let all = cache::list_cached_estimates("testnet").expect("list v1 entries");
        assert_eq!(all.len(), 3, "all 3 v1 entries should load");

        for entry in &all {
            assert_eq!(
                entry.version,
                current_schema_version(),
                "v1 entry must migrate to current version"
            );
            // Default values applied during v1→v2 migration
            assert_eq!(entry.success, true, "v1 entries default to success=true");
            assert_eq!(
                entry.execution_duration_ms, 0,
                "v1 entries default to unknown duration"
            );
        }

        // Verify the cache with migrated entries
        let statuses = cache::verify_cache().expect("verify migrated v1");
        assert_eq!(statuses.len(), 3);
        assert!(
            statuses.iter().all(|s| s.valid),
            "all migrated v1 entries must verify as valid"
        );
    });
}

/// Schema v2 entries should load and migrate to v3.
/// v2 entries lack `last_accessed` but have `duration_ms` and `success`.
#[test]
fn test_migrate_v2_to_latest() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::DURATION_SCHEMA_VERSION, 3);

        let all = cache::list_cached_estimates("testnet").expect("list v2 entries");
        assert_eq!(all.len(), 3, "all 3 v2 entries should load");

        for entry in &all {
            assert_eq!(
                entry.version,
                current_schema_version(),
                "v2 entry must migrate to current version"
            );
            // v2 entries already have success and duration_ms (set by insert_raw_row defaults)
            assert_eq!(entry.success, true, "v2 entries include success field");
        }

        let statuses = cache::verify_cache().expect("verify migrated v2");
        assert_eq!(statuses.len(), 3);
        assert!(
            statuses.iter().all(|s| s.valid),
            "all migrated v2 entries must verify as valid"
        );
    });
}

/// Schema v3 (current) entries should load unchanged.
#[test]
fn test_migrate_v3_to_latest_is_noop() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::ACCESS_TRACKING_SCHEMA_VERSION, 3);

        let all = cache::list_cached_estimates("testnet").expect("list v3 entries");
        assert_eq!(all.len(), 3);

        for entry in &all {
            assert_eq!(entry.version, current_schema_version());
            assert_eq!(entry.success, true);
        }
    });
}

/// Direct call to `migrate_to_latest` with a v1 struct should upgrade fields
/// through the v1→v2 step, then v2→v3 step.
#[test]
fn test_migrate_to_latest_v1_struct_applies_all_steps() {
    let v1_entry = cache::CachedEstimate {
        version: cache::INITIAL_SCHEMA_VERSION,
        wasm_hash: "abc".to_string(),
        function: "f".to_string(),
        args_hash: "def".to_string(),
        network: "testnet".to_string(),
        ledger: 1,
        total_stroops: 100,
        cpu_instructions: 10,
        memory_bytes: 5,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        // v1 entries lack these; serde fills them with defaults
        execution_duration_ms: 0,
        success: true,
        network_passphrase: None,
        core_version: None,
        io: None,
    };

    let migrated = cache::migrate_to_latest(v1_entry).expect("migrate v1 struct");
    assert_eq!(migrated.version, current_schema_version());
    assert_eq!(
        migrated.execution_duration_ms, 0,
        "v1→v2 sets default duration"
    );
    assert_eq!(migrated.success, true, "v1→v2 sets default success");
}

/// Direct call to `migrate_to_latest` with a v2 struct should upgrade to v3.
#[test]
fn test_migrate_to_latest_v2_struct_applies_v2_to_v3_step() {
    let v2_entry = cache::CachedEstimate {
        version: cache::DURATION_SCHEMA_VERSION,
        wasm_hash: "abc".to_string(),
        function: "f".to_string(),
        args_hash: "def".to_string(),
        network: "testnet".to_string(),
        ledger: 2,
        total_stroops: 200,
        cpu_instructions: 20,
        memory_bytes: 10,
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        execution_duration_ms: 42,
        success: false,
        network_passphrase: None,
        core_version: None,
        io: None,
    };

    let migrated = cache::migrate_to_latest(v2_entry).expect("migrate v2 struct");
    assert_eq!(migrated.version, current_schema_version());
    // v2 fields are preserved
    assert_eq!(migrated.execution_duration_ms, 42);
    assert_eq!(migrated.success, false);
}

/// Mixed-version cache (v1, v2, v3 entries together) should load and migrate
/// all entries to the current version.
#[test]
fn test_mixed_schema_versions_load_and_migrate() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Insert one entry at each version
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "hash-v1",
            "func-v1",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::DURATION_SCHEMA_VERSION,
            "hash-v2",
            "func-v2",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::ACCESS_TRACKING_SCHEMA_VERSION,
            "hash-v3",
            "func-v3",
            &[],
            "testnet",
            300,
            "2026-01-03T00:00:00Z",
        );

        let all = cache::list_cached_estimates("testnet").expect("list mixed versions");
        assert_eq!(all.len(), 3, "all 3 mixed-version entries should load");

        // All should be at the current version after migration
        assert!(
            all.iter().all(|e| e.version == current_schema_version()),
            "all migrated entries should be at current version"
        );

        let statuses = cache::verify_cache().expect("verify mixed");
        assert_eq!(statuses.len(), 3);
        assert!(
            statuses.iter().all(|s| s.valid),
            "all mixed-version entries must verify as valid: {statuses:?}"
        );
    });
}

/// After migrating a database, indexes must still be functional for queries
/// (no index corruption).
#[test]
fn test_migrated_cache_indexes_are_functional() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, 5);

        // Load by specific key
        let specific = cache::load_estimate("hash-v1-0", "func-v1-0", &[])
            .expect("load migrated entry")
            .expect("entry should exist");
        assert_eq!(specific.wasm_hash, "hash-v1-0");

        // List all (uses network index)
        let all = cache::list_cached_estimates("testnet").expect("list via network index");
        assert_eq!(all.len(), 5);

        // Query should work on migrated data
        let filter = cache::CacheFilter {
            function: Some("func-v1-0".to_string()),
            ..Default::default()
        };
        let queried = cache::query_cache(&filter).expect("query migrated cache");
        assert_eq!(queried.len(), 1);
        assert_eq!(queried[0].function, "func-v1-0");
    });
}

/// A transaction boundary must be respected during migration: if a partial
/// write occurs, the original schema/data must remain in the database
/// (atomicity). This is tested by simulating a partial write at the SQLite
/// level and confirming the database is recoverable.
///
/// Note: We cannot easily simulate a mid-transaction crash in the test, but
/// we can verify that all migration operations are transactional by checking
/// that either all rows are migrated or none are.
#[test]
fn test_migration_is_transactional() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, 3);

        // Before loading (which triggers migration), the raw database rows
        // should be at v1
        let dir = tmp.join(".soroban-cost-estimator");
        let db = dir.join("cache.db");
        let conn_check = rusqlite::Connection::open(&db).expect("check db before migration");
        let mut stmt = conn_check
            .prepare("SELECT DISTINCT version FROM estimates")
            .expect("query versions");
        let versions: Vec<u32> = stmt
            .query_map([], |row| row.get::<_, u32>(0))
            .expect("map versions")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect versions");

        assert_eq!(
            versions.len(),
            1,
            "all rows should be at one version initially"
        );
        assert_eq!(versions[0], cache::INITIAL_SCHEMA_VERSION);

        // Now trigger migration by loading
        let _all = cache::list_cached_estimates("testnet").expect("load and migrate");

        // After migration, all rows should be at the current version (atomic
        // batch update, not row-by-row with potential partial failures)
        let conn_after = rusqlite::Connection::open(&db).expect("check db after migration");
        let mut stmt_after = conn_after
            .prepare("SELECT DISTINCT version FROM estimates")
            .expect("query versions after");
        let versions_after: Vec<u32> = stmt_after
            .query_map([], |row| row.get::<_, u32>(0))
            .expect("map versions after")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect versions after");

        assert_eq!(
            versions_after.len(),
            1,
            "all rows should end up at one version (atomic)"
        );
        assert_eq!(versions_after[0], current_schema_version());
    });
}

/// When loading a v1 entry, verify that the indexes used by LRU eviction
/// (last_accessed) are properly populated during the migration.
#[test]
fn test_v1_entries_get_last_accessed_indexed() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, 2);

        let all = cache::list_cached_estimates("testnet").expect("load v1");
        assert_eq!(all.len(), 2);

        // Try eviction, which relies on last_accessed being present
        // (and not all entries being empty strings). This test documents
        // that eviction works on migrated v1 entries.
        let limits = cache::CacheLimits {
            max_entries: 1,
            max_bytes: 1_000_000,
        };
        cache::set_cache_limits(limits).expect("set limits");

        // Save a new entry to trigger eviction
        cache::save_estimate(
            "new",
            "new_func",
            &[],
            "testnet",
            1000,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("save new entry");

        let after = cache::list_cached_estimates("testnet").expect("list after eviction");
        assert_eq!(after.len(), 1, "eviction should have removed an old entry");
        assert_eq!(after[0].wasm_hash, "new", "new entry should remain");
    });
}

/// Verify that v2→v3 migration preserves the `success` and `execution_duration_ms`
/// fields from v2 entries.
#[test]
fn test_v2_to_v3_preserves_duration_and_success_fields() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Manually insert a v2 entry with specific duration_ms and success
        conn.execute(
            "INSERT INTO estimates (version, wasm_hash, function, args_hash, network, ledger, \
             total_stroops, cpu_instructions, memory_bytes, timestamp, execution_duration_ms, success) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                cache::DURATION_SCHEMA_VERSION as i64,
                "abc",
                "test_func",
                "args_hash",
                "testnet",
                123i64,
                50_000i64,
                100_000i64,
                10_000i64,
                "2026-01-01T12:34:56Z",
                567i64, // specific duration_ms
                0i64,   // success = false
            ],
        )
        .expect("insert v2 entry");

        let loaded = cache::load_estimate("abc", "test_func", &[])
            .expect("load v2 entry")
            .expect("entry should exist");

        assert_eq!(loaded.version, current_schema_version());
        assert_eq!(
            loaded.execution_duration_ms, 567,
            "v2 duration_ms preserved"
        );
        assert_eq!(loaded.success, false, "v2 success field preserved");
    });
}

/// Loading multiple entries from different schema versions in rapid
/// succession (rapid calls to `load_estimate`) should not corrupt the
/// migration state.
#[test]
fn test_rapid_loads_of_mixed_versions_does_not_corrupt() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Insert mixed-version entries
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "hash1",
            "f1",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::DURATION_SCHEMA_VERSION,
            "hash2",
            "f2",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::ACCESS_TRACKING_SCHEMA_VERSION,
            "hash3",
            "f3",
            &[],
            "testnet",
            300,
            "2026-01-03T00:00:00Z",
        );

        // Load rapidly in any order
        let e1 = cache::load_estimate("hash3", "f3", &[])
            .expect("load f3")
            .expect("f3 should exist");
        let e2 = cache::load_estimate("hash1", "f1", &[])
            .expect("load f1")
            .expect("f1 should exist");
        let e3 = cache::load_estimate("hash2", "f2", &[])
            .expect("load f2")
            .expect("f2 should exist");

        // All should have been migrated
        assert_eq!(e1.version, current_schema_version());
        assert_eq!(e2.version, current_schema_version());
        assert_eq!(e3.version, current_schema_version());

        // Verify none are corrupted
        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 3);
        assert!(statuses.iter().all(|s| s.valid));
    });
}

/// Rejection of future-version entries must occur during `load_estimate`,
/// preventing partial or corrupted reads.
#[test]
fn test_rejection_of_future_version_prevents_partial_reads() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "future",
            "new_func",
            &[],
            Some(current_schema_version() + 1),
            1,
        );

        // Attempting to load should fail cleanly, not crash or partially load
        let result = cache::load_estimate("future", "new_func", &[]);
        assert!(result.is_err(), "future-version should be rejected");

        // The error should be informative
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("newer") || err_msg.contains("upgrade"),
            "error should hint at upgrading: {err_msg}"
        );
    });
}

/// Verify that all indexes remain intact and functional after a migration.
/// This ensures queries don't silently miss entries or return duplicates.
#[test]
fn test_migrated_indexes_no_duplicates_or_misses() {
    with_temp_home(|tmp| {
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, 10);

        let all = cache::list_cached_estimates("testnet").expect("list after migration");
        assert_eq!(all.len(), 10, "no duplicates, no misses");

        // Count using a direct query should match
        let count = cache::count_cached_estimates(&cache::CacheFilter::default()).expect("count");
        assert_eq!(count, 10, "count must match list length");

        // Pagination should not lose or duplicate entries
        let page1 = cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 5, 0)
            .expect("page 1");
        let page2 = cache::list_cached_estimates_paginated(&cache::CacheFilter::default(), 5, 5)
            .expect("page 2");

        assert_eq!(page1.total, 10);
        assert_eq!(page2.total, 10);
        assert_eq!(page1.entries.len(), 5);
        assert_eq!(page2.entries.len(), 5);

        // No entry should appear in both pages
        let page1_hashes: Vec<&str> = page1.entries.iter().map(|e| e.wasm_hash.as_str()).collect();
        for page2_entry in &page2.entries {
            assert!(
                !page1_hashes.contains(&page2_entry.wasm_hash.as_str()),
                "entry {} appears in both pages",
                page2_entry.wasm_hash
            );
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Migration rollback and partial failure recovery
// ─────────────────────────────────────────────────────────────────────────

/// Simulate a migration failure scenario: if a database read fails midway
/// through loading a migrated entry, the original data should remain
/// unmodified on disk (no side effects).
///
/// This test verifies that `load_estimate` with an invalid or corrupted
/// entry fails cleanly and doesn't corrupt other entries.
#[test]
fn test_migration_failure_does_not_corrupt_unrelated_entries() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Insert one good v1 entry and one v2 entry with a very large CPU
        // instruction count (to differentiate them)
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "good",
            "good_func",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::DURATION_SCHEMA_VERSION,
            "also_good",
            "also_good_func",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );

        // Load the first entry (should migrate successfully)
        let loaded1 = cache::load_estimate("good", "good_func", &[])
            .expect("load first entry")
            .expect("first entry should exist");
        assert_eq!(loaded1.wasm_hash, "good");
        assert_eq!(loaded1.version, current_schema_version());

        // Load the second entry (should also migrate successfully)
        let loaded2 = cache::load_estimate("also_good", "also_good_func", &[])
            .expect("load second entry")
            .expect("second entry should exist");
        assert_eq!(loaded2.wasm_hash, "also_good");
        assert_eq!(loaded2.version, current_schema_version());

        // Both entries should still be queryable
        let all = cache::list_cached_estimates("testnet").expect("list after loads");
        assert_eq!(all.len(), 2, "both entries should still be present");

        // Verify both are valid
        let statuses = cache::verify_cache().expect("verify after loads");
        assert_eq!(statuses.len(), 2, "both entries should verify");
        assert!(
            statuses.iter().all(|s| s.valid),
            "both should be valid: {statuses:?}"
        );
    });
}

/// When a migration rollback is needed (i.e., an entry cannot be migrated),
/// the `load_estimate` call fails and other entries remain unaffected.
/// This tests the "reject and leave everything else alone" path.
#[test]
fn test_unmigrable_entry_does_not_affect_other_entries() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Insert a future-version entry that cannot be migrated
        insert_raw_row(
            tmp,
            current_schema_version() + 5,
            "unmigrable",
            "future_func",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );

        // Insert a valid v1 entry
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "good",
            "good_func",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );

        // Loading the unmigrable entry must fail
        let result = cache::load_estimate("unmigrable", "future_func", &[]);
        assert!(result.is_err(), "unmigrable entry should fail to load");

        // But the good entry should still be loadable
        let loaded = cache::load_estimate("good", "good_func", &[])
            .expect("load good entry after unmigrable failure")
            .expect("good entry should exist");
        assert_eq!(loaded.wasm_hash, "good");
        assert_eq!(loaded.version, current_schema_version());
    });
}

/// A database that has a mix of migratable and non-migratable entries must
/// report the non-migratable ones as invalid via `verify_cache`, while
/// keeping the migratable ones valid.
#[test]
fn test_verify_cache_with_unmigrable_entries() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // One good v1 entry
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "good",
            "good_func",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );

        // One future-version entry that cannot be migrated
        insert_raw_row(
            tmp,
            current_schema_version() + 10,
            "future",
            "future_func",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );

        let statuses = cache::verify_cache().expect("verify mixed valid/invalid");
        assert_eq!(statuses.len(), 2);

        let good_status = statuses
            .iter()
            .find(|s| s.filename.contains("good"))
            .expect("good entry should be reported");
        assert!(good_status.valid, "v1 entry should be valid");

        let future_status = statuses
            .iter()
            .find(|s| s.filename.contains("future"))
            .expect("future entry should be reported");
        assert!(
            !future_status.valid,
            "future-version entry should be invalid"
        );
    });
}

/// If a database has partially migrated data (some rows at v1, some at v3),
/// they should all migrate correctly on next load.
#[test]
fn test_partial_migration_state_resolves_on_next_load() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Simulate a partially-migrated state: some rows at v1, some at v3
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "hash-v1-1",
            "func-v1-1",
            &[],
            "testnet",
            100,
            "2026-01-01T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::ACCESS_TRACKING_SCHEMA_VERSION,
            "hash-v3-1",
            "func-v3-1",
            &[],
            "testnet",
            200,
            "2026-01-02T00:00:00Z",
        );
        insert_raw_row(
            tmp,
            cache::INITIAL_SCHEMA_VERSION,
            "hash-v1-2",
            "func-v1-2",
            &[],
            "testnet",
            300,
            "2026-01-03T00:00:00Z",
        );

        // Load all entries; the v1 ones should migrate, v3 should stay as-is
        let all = cache::list_cached_estimates("testnet").expect("list partial state");
        assert_eq!(all.len(), 3, "all entries should be present");

        for entry in &all {
            assert_eq!(
                entry.version,
                current_schema_version(),
                "all entries should be at current version"
            );
        }

        // Verify cache should report all as valid
        let statuses = cache::verify_cache().expect("verify partial");
        assert_eq!(statuses.len(), 3);
        assert!(
            statuses.iter().all(|s| s.valid),
            "all entries in partial state should verify as valid"
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Future schema version rejection
// ─────────────────────────────────────────────────────────────────────────

/// An entry written by a tool newer than the current version must be
/// rejected during `load_estimate` with an informative error message.
#[test]
fn test_future_version_entry_rejected_with_upgrade_message() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "future",
            "new_func",
            &[],
            Some(current_schema_version() + 1),
            1,
        );

        let result = cache::load_estimate("future", "new_func", &[]);
        assert!(result.is_err(), "future-version entry must be rejected");

        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("newer")
                || msg.contains("upgrade")
                || msg.contains("soroban-cost-estimator"),
            "error message should guide user to upgrade: {msg}"
        );
    });
}

/// Multiple future-version entries must all be flagged by `verify_cache`.
#[test]
fn test_verify_cache_flags_all_future_version_entries() {
    with_temp_home(|tmp| {
        // Insert multiple future-version entries
        for i in 0..3 {
            write_raw_entry(
                tmp,
                &format!("future-{}", i),
                &format!("future_func-{}", i),
                &[],
                Some(current_schema_version() + 1),
                (100 + i as u32) * 10,
            );
        }

        let statuses = cache::verify_cache().expect("verify");
        assert_eq!(statuses.len(), 3, "all 3 future entries should be reported");

        for status in &statuses {
            assert!(
                !status.valid,
                "all future entries must be invalid: {status:?}"
            );
        }
    });
}

/// An entry with a much-older schema version (e.g., v0 or negative) must be
/// migrated without panicking.
#[test]
fn test_very_old_schema_versions_migrate_gracefully() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        // Insert an entry with version 0 (pre-versioning)
        conn.execute(
            "INSERT INTO estimates (version, wasm_hash, function, args_hash, network, ledger, \
             total_stroops, cpu_instructions, memory_bytes, timestamp) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                0i64, // very old version
                "very_old",
                "old_func",
                "args_hash",
                "testnet",
                1i64,
                100i64,
                10i64,
                5i64,
                "2026-01-01T00:00:00Z",
            ],
        )
        .expect("insert version 0");

        // Load should not panic and should treat it as v1
        let loaded = cache::load_estimate("very_old", "old_func", &[])
            .expect("load very old version")
            .expect("entry should exist");
        assert_eq!(loaded.version, current_schema_version());
    });
}

/// Concurrent loads of a future-version entry from multiple threads must
/// each fail without corrupting the cache.
#[test]
fn test_concurrent_loads_of_future_version_entry_all_fail() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "future",
            "new_func",
            &[],
            Some(current_schema_version() + 1),
            1,
        );

        let handles: Vec<_> = (0..CONCURRENT_THREADS)
            .map(|_| {
                std::thread::spawn(|| {
                    let result = cache::load_estimate("future", "new_func", &[]);
                    assert!(
                        result.is_err(),
                        "concurrent load of future entry must fail: {result:?}"
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("concurrent load thread panicked");
        }

        // Cache should still verify cleanly (no corruption)
        let statuses = cache::verify_cache().expect("verify after concurrent rejects");
        assert_eq!(statuses.len(), 1);
        assert!(!statuses[0].valid);
    });
}

/// An entry exactly one version ahead should be rejected (we only accept
/// versions <= current).
#[test]
fn test_entry_one_version_ahead_is_rejected() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "v_plus_1",
            "func",
            &[],
            Some(current_schema_version() + 1),
            1,
        );

        let result = cache::load_estimate("v_plus_1", "func", &[]);
        assert!(result.is_err());
    });
}

/// An entry many versions ahead should be rejected with a clear error.
#[test]
fn test_entry_many_versions_ahead_is_rejected() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "v_plus_100",
            "func",
            &[],
            Some(current_schema_version() + 100),
            1,
        );

        let result = cache::load_estimate("v_plus_100", "func", &[]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("100") || err.contains("newer") || err.contains("upgrade"));
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Additional edge cases and robustness
// ─────────────────────────────────────────────────────────────────────────

/// After a successful migration, a second load should be a no-op (entry is
/// already at current version).
#[test]
fn test_repeated_loads_of_migrated_entry_are_idempotent() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "hash",
            "func",
            &[],
            Some(cache::INITIAL_SCHEMA_VERSION),
            1,
        );

        let first_load = cache::load_estimate("hash", "func", &[])
            .expect("first load")
            .expect("entry should exist");
        assert_eq!(first_load.version, current_schema_version());

        // Load again
        let second_load = cache::load_estimate("hash", "func", &[])
            .expect("second load")
            .expect("entry should exist");
        assert_eq!(second_load.version, current_schema_version());

        // Both should be identical
        assert_eq!(first_load.ledger, second_load.ledger);
        assert_eq!(first_load.total_stroops, second_load.total_stroops);
        assert_eq!(first_load.wasm_hash, second_load.wasm_hash);
    });
}

/// Migration through all intermediate versions (v1 → v2 → v3) should preserve
/// all numeric fields exactly.
#[test]
fn test_migration_preserves_numeric_fields_exactly() {
    with_temp_home(|tmp| {
        let dir = tmp.join(".soroban-cost-estimator");
        std::fs::create_dir_all(&dir).expect("create data dir");
        let db = dir.join("cache.db");
        let conn = rusqlite::Connection::open(&db).expect("open cache db");
        cache::ensure_cache_schema(&conn).expect("ensure schema");

        let wasm_hash = "test_hash";
        let function = "test_func";
        let args_hash = "test_args";
        let network = "testnet";
        let ledger = 12345u32;
        let total_stroops = 9_876_543i64;
        let cpu_instructions = 1_234_567u64;
        let memory_bytes = 987_654u64;

        // Insert a v1 entry with specific numeric values
        conn.execute(
            "INSERT INTO estimates (version, wasm_hash, function, args_hash, network, ledger, \
             total_stroops, cpu_instructions, memory_bytes, timestamp) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                cache::INITIAL_SCHEMA_VERSION as i64,
                wasm_hash,
                function,
                args_hash,
                network,
                ledger as i64,
                total_stroops,
                cpu_instructions as i64,
                memory_bytes as i64,
                "2026-01-01T00:00:00Z",
            ],
        )
        .expect("insert v1 entry");

        // Load and verify all numeric fields are preserved exactly
        let loaded = cache::load_estimate(wasm_hash, function, &[])
            .expect("load")
            .expect("entry should exist");

        assert_eq!(loaded.ledger, ledger, "ledger must be preserved exactly");
        assert_eq!(
            loaded.total_stroops, total_stroops,
            "total_stroops must be preserved exactly"
        );
        assert_eq!(
            loaded.cpu_instructions, cpu_instructions,
            "cpu_instructions must be preserved exactly"
        );
        assert_eq!(
            loaded.memory_bytes, memory_bytes,
            "memory_bytes must be preserved exactly"
        );
    });
}

/// When a large cache with many entries from different schema versions is
/// migrated, all entries should remain intact (no truncation or data loss).
#[test]
fn test_large_mixed_version_cache_full_integrity() {
    with_temp_home(|tmp| {
        let entry_count = 100;
        build_database_at_schema_version(tmp, cache::INITIAL_SCHEMA_VERSION, entry_count);

        let all = cache::list_cached_estimates("testnet").expect("list large cache");
        assert_eq!(
            all.len(),
            entry_count,
            "all {} entries must be present after migration",
            entry_count
        );

        for entry in &all {
            assert_eq!(entry.version, current_schema_version());
            assert!(entry.wasm_hash.starts_with("hash-v"), "hash must be intact");
            assert!(
                entry.function.starts_with("func-v"),
                "function must be intact"
            );
        }

        let statuses = cache::verify_cache().expect("verify large cache");
        assert_eq!(statuses.len(), entry_count);
        assert!(
            statuses.iter().all(|s| s.valid),
            "all entries in large cache must verify as valid"
        );
    });
}

/// A schema v1 entry that is saved again (via `save_estimate`) should become
/// v3 and be identical to a fresh v3 entry (except for timestamp/ledger).
#[test]
fn test_re_saved_migrated_entry_becomes_current_version() {
    with_temp_home(|tmp| {
        write_raw_entry(
            tmp,
            "hash",
            "func",
            &["arg"],
            Some(cache::INITIAL_SCHEMA_VERSION),
            100,
        );

        let loaded = cache::load_estimate("hash", "func", &["arg".to_string()])
            .expect("load migrated")
            .expect("entry should exist");
        assert_eq!(loaded.version, cache::INITIAL_SCHEMA_VERSION);

        // Now save over it with the modern save_estimate
        cache::save_estimate(
            "hash",
            "func",
            &["arg".to_string()],
            "testnet",
            200,
            100,
            10,
            5,
            None,
            true,
        )
        .expect("re-save");

        let re_saved = cache::load_estimate("hash", "func", &["arg".to_string()])
            .expect("load re-saved")
            .expect("entry should exist");
        assert_eq!(re_saved.version, current_schema_version());
    });
}
