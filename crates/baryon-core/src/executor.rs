//! Parallel execution on a dedicated thread pool.

use std::cmp::Reverse;
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

    /// Like [`Executor::map`], but starts the most expensive items first.
    ///
    /// Items are handed to idle threads one at a time in descending `cost`, so a large item
    /// never starts last while the other threads wait for it (longest-processing-time-first
    /// scheduling). Use this when item costs vary widely and are known up front, such as
    /// decoding images of mixed sizes. Results are in input order.
    pub fn map_by_cost<T: Sync, R: Send>(
        &self,
        items: &[T],
        cost: impl Fn(&T) -> u64,
        op: impl Fn(&T) -> R + Send + Sync,
    ) -> Vec<R> {
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&index| Reverse(cost(&items[index])));
        // `par_bridge` lets each idle thread pull the next index from the shared iterator, which
        // keeps the order; `par_iter` would split the list into ranges up front instead.
        let mut done: Vec<(usize, R)> = self.install(|| {
            order
                .into_iter()
                .par_bridge()
                .map(|index| (index, op(&items[index])))
                .collect()
        });
        done.sort_unstable_by_key(|&(index, _)| index);
        done.into_iter().map(|(_, result)| result).collect()
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
    fn map_by_cost_preserves_order() {
        let items: Vec<u64> = (0..1000).collect();
        for threads in [1, 4] {
            let executor = Executor::new(threads).unwrap();
            let squares = executor.map_by_cost(&items, |&x| x % 7, |x| x * x);
            assert_eq!(squares, items.iter().map(|x| x * x).collect::<Vec<_>>());
        }
    }

    #[test]
    fn map_by_cost_starts_expensive_items_first() {
        let started = Mutex::new(Vec::new());
        Executor::new(1).unwrap().map_by_cost(
            &[3_u64, 9, 1, 7],
            |&x| x,
            |&x| started.lock().unwrap().push(x),
        );
        assert_eq!(started.into_inner().unwrap(), [9, 7, 3, 1]);
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
