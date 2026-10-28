//! Parallel execution on a dedicated thread pool.

use std::fmt;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{Error, Result};

/// Runs work on a dedicated thread pool.
///
/// Everything parallel inside baryon (tile decoding, resizing, crops, rotations, batches) runs on
/// the pool it was started from, so an executor bounds the CPU used by all the work it runs.
/// Work started outside any executor runs on rayon's global pool.
#[derive(Clone)]
pub struct Executor {
    pool: Arc<rayon::ThreadPool>,
}

impl Executor {
    /// Starts a pool with `num_threads` workers; `0` means one per logical CPU (or the value of
    /// `RAYON_NUM_THREADS`, if set).
    pub fn new(num_threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("baryon-{index}"))
            .build()
            .map_err(|err| Error::Internal(format!("failed to start thread pool: {err}")))?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// Number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Runs `op` on the pool, so that any parallel work it starts uses this executor's threads.
    /// Blocks the calling thread until `op` returns.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.pool.install(op)
    }

    /// Applies `op` to every item in parallel, returning results in input order.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], op: impl Fn(&T) -> R + Send + Sync) -> Vec<R> {
        self.install(|| items.par_iter().map(op).collect())
    }
}

impl fmt::Debug for Executor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executor")
            .field("num_threads", &self.num_threads())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn reports_its_thread_count() {
        assert_eq!(Executor::new(1).unwrap().num_threads(), 1);
        assert_eq!(Executor::new(3).unwrap().num_threads(), 3);
        assert!(Executor::new(0).unwrap().num_threads() >= 1);
    }

    #[test]
    fn map_preserves_order() {
        let items: Vec<u64> = (0..1000).collect();
        let squares = Executor::new(4).unwrap().map(&items, |x| x * x);
        assert_eq!(squares, items.iter().map(|x| x * x).collect::<Vec<_>>());
    }

    #[test]
    fn work_runs_on_the_executor_threads() {
        let executor = Executor::new(2).unwrap();
        let names = Mutex::new(HashSet::new());
        executor.map(&[(); 64], |()| {
            let name = std::thread::current().name().map(str::to_owned);
            names.lock().unwrap().insert(name);
        });
        let names = names.into_inner().unwrap();
        assert!(!names.is_empty());
        assert!(
            names
                .iter()
                .all(|n| n.as_deref().is_some_and(|n| n.starts_with("baryon-")))
        );
    }
}
