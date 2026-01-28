use crate::arena::{Graph, Node, NodeId};
use crate::engine::{CachedRedex, RedexKey, RedexMemo};
use crate::learner::counterfactual::utils::resolve_safe;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

pub const EST_SIZE_CACHE_MAX: usize = 200_000;

static EST_SIZE_CACHE: OnceLock<Mutex<HashMap<EstSizeKey, usize>>> = OnceLock::new();

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct EstSizeKey {
    pub graph_id: u64,
    pub epoch: u64,
    pub node: NodeId,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct LossKey {
    pub program_hash: u64,
    pub input_hash: u64,
    pub expected_hash: u64,
    pub max_steps: usize,
    pub max_nodes: usize,
    pub max_ted_cells: usize,
}

#[derive(Clone, Copy)]
pub struct LossEntry {
    pub loss: f64,
    pub actual_hash: u64,
}

#[derive(Clone, Copy)]
pub struct LossSample {
    pub loss: f64,
    pub actual_hash: u64,
    pub expected_hash: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct CandidateKey {
    pub program_hash: u64,
    pub path: Vec<u8>,
    pub replacement_hash: u64,
    pub examples_hash: u64,
    pub max_steps: usize,
    pub max_nodes: usize,
    pub max_ted_cells: usize,
}

#[derive(Clone, Copy)]
pub struct CandidateStats {
    pub mean: f64,
    pub var: f64,
}

struct CacheEntry<V: Clone> {
    value: V,
    id: u64,
    size: usize,
}

pub struct LruCache<K: Eq + std::hash::Hash + Clone, V: Clone> {
    map: HashMap<K, CacheEntry<V>>,
    order: VecDeque<(K, u64)>,
    bytes: usize,
    budget: usize,
    counter: u64,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> LruCache<K, V> {
    pub fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget,
            counter: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.map.get_mut(key) {
            let id = self.counter;
            self.counter = self.counter.wrapping_add(1);
            entry.id = id;
            self.order.push_back((key.clone(), id));
            return Some(entry.value.clone());
        }
        None
    }

    pub fn insert(&mut self, key: K, value: V, size: usize) {
        if let Some(entry) = self.map.remove(&key) {
            self.bytes = self.bytes.saturating_sub(entry.size);
        }
        let id = self.counter;
        self.counter = self.counter.wrapping_add(1);
        self.map.insert(
            key.clone(),
            CacheEntry {
                value,
                id,
                size,
            },
        );
        self.order.push_back((key, id));
        self.bytes = self.bytes.saturating_add(size);
        self.evict();
    }

    fn evict(&mut self) {
        while self.bytes > self.budget {
            let Some((key, id)) = self.order.pop_front() else { break; };
            if let Some(entry) = self.map.get(&key) {
                if entry.id == id {
                    let entry = self.map.remove(&key).unwrap();
                    self.bytes = self.bytes.saturating_sub(entry.size);
                }
            }
        }
    }
}

pub struct MemoCaches {
    pub loss: Mutex<LruCache<LossKey, LossEntry>>,
    pub candidate: Mutex<LruCache<CandidateKey, CandidateStats>>,
    pub redex: Mutex<LruCache<RedexKey, Arc<CachedRedex>>>,
    pub redex_budget: usize,
    pub max_cached_nodes: usize,
}

impl MemoCaches {
    pub fn new(budget: usize, max_cached_nodes: usize) -> Self {
        let per = budget / 5;
        Self {
            loss: Mutex::new(LruCache::new(per)),
            candidate: Mutex::new(LruCache::new(per)),
            redex: Mutex::new(LruCache::new(per)),
            redex_budget: per,
            max_cached_nodes,
        }
    }

    pub fn get_loss(&self, key: &LossKey) -> Option<LossEntry> {
        let mut cache = self.loss.lock().unwrap();
        cache.get(key)
    }

    pub fn insert_loss(&self, key: LossKey, value: LossEntry, size: usize) {
        let mut cache = self.loss.lock().unwrap();
        cache.insert(key, value, size);
    }

    pub fn get_candidate(&self, key: &CandidateKey) -> Option<CandidateStats> {
        let mut cache = self.candidate.lock().unwrap();
        cache.get(key)
    }

    pub fn insert_candidate(&self, key: CandidateKey, value: CandidateStats, size: usize) {
        let mut cache = self.candidate.lock().unwrap();
        cache.insert(key, value, size);
    }
}

impl RedexMemo for MemoCaches {
    fn get_redex(&self, key: &RedexKey) -> Option<Arc<CachedRedex>> {
        let mut cache = self.redex.lock().unwrap();
        cache.get(key)
    }

    fn insert_redex(&self, key: RedexKey, value: Arc<CachedRedex>, size: usize, nodes: usize) {
        if nodes > self.max_cached_nodes {
            return;
        }
        if size > self.redex_budget {
            return;
        }
        let mut cache = self.redex.lock().unwrap();
        cache.insert(key, value, size);
    }

    fn max_redex_nodes(&self) -> usize {
        self.max_cached_nodes
    }
}

pub fn loss_entry_size(_key: &LossKey) -> usize {
    std::mem::size_of::<LossKey>() + std::mem::size_of::<LossEntry>()
}

pub fn candidate_entry_size(key: &CandidateKey) -> usize {
    std::mem::size_of::<CandidateKey>()
        + key.path.capacity()
        + std::mem::size_of::<CandidateStats>()
}

pub fn estimate_size(g: &Graph, root: NodeId) -> usize {
    let resolved_root = resolve_safe(g, root);
    if let Some(cache) = EST_SIZE_CACHE.get() {
        let key = EstSizeKey {
            graph_id: g.id,
            epoch: g.epoch,
            node: resolved_root,
        };
        if let Some(size) = cache.lock().unwrap().get(&key).copied() {
            return size;
        }
    }

    let mut memo: HashMap<NodeId, usize> = HashMap::new();
    let mut visiting: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    stack.push((resolved_root, false));
    while let Some((id, expanded)) = stack.pop() {
        let resolved = resolve_safe(g, id);
        if expanded {
            if memo.contains_key(&resolved) {
                visiting.remove(&resolved);
                continue;
            }
            let size = match g.get(resolved) {
                Node::Leaf | Node::Prim(_) | Node::Float(_) | Node::Handle(_) => 1,
                Node::Stem(inner) => 1 + memo.get(&resolve_safe(g, *inner)).copied().unwrap_or(1),
                Node::Fork(l, r) => {
                    1 + memo.get(&resolve_safe(g, *l)).copied().unwrap_or(1)
                        + memo.get(&resolve_safe(g, *r)).copied().unwrap_or(1)
                }
                Node::App { func, args } => {
                    let mut total = 1 + memo.get(&resolve_safe(g, *func)).copied().unwrap_or(1);
                    for arg in args {
                        total += memo.get(&resolve_safe(g, *arg)).copied().unwrap_or(1);
                    }
                    total
                }
                Node::Ind(inner) => memo.get(&resolve_safe(g, *inner)).copied().unwrap_or(1),
            };
            memo.insert(resolved, size);
            visiting.remove(&resolved);
        } else {
            if memo.contains_key(&resolved) {
                continue;
            }
            if !visiting.insert(resolved) {
                memo.insert(resolved, 1);
                continue;
            }
            stack.push((resolved, true));
            match g.get(resolved) {
                Node::Stem(inner) => stack.push((*inner, false)),
                Node::Fork(l, r) => {
                    stack.push((*l, false));
                    stack.push((*r, false));
                }
                Node::App { func, args } => {
                    stack.push((*func, false));
                    for arg in args {
                        stack.push((*arg, false));
                    }
                }
                Node::Ind(inner) => stack.push((*inner, false)),
                _ => {}
            }
        }
    }
    let size = memo.get(&resolved_root).copied().unwrap_or(1);
    let cache = EST_SIZE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if cache.len() >= EST_SIZE_CACHE_MAX {
        cache.clear();
    }
    for (node, node_size) in memo {
        if cache.len() >= EST_SIZE_CACHE_MAX {
            break;
        }
        cache.insert(
            EstSizeKey {
                graph_id: g.id,
                epoch: g.epoch,
                node,
            },
            node_size,
        );
    }
    size
}
