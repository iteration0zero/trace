//! Counterfactual learner: single-candidate, trace-driven edits.

use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::engine::{
    reduce, set_reduce_progress_ms_override, tree_hash, CachedRedex, EvalContext, RedexKey,
    RedexMemo,
};
use crate::learner::loss::tree_edit_distance_capped;
use crate::trace::{ExecutionTrace, RuleId};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const SLOW_EVAL_MS: u128 = 100;
const EVAL_STEP_CAP: usize = 200_000;
const STAGE_WARN_MS_DEFAULT: u128 = 3000;
const STAGE_WARN_REPEAT_MS_DEFAULT: u128 = 3000;
const UNPARSE_MAX_DEPTH: usize = 96;
const UNPARSE_MAX_NODES: usize = 4000;
const UNPARSE_MAX_ARGS: usize = 16;
static DEBUG_OOM: AtomicBool = AtomicBool::new(false);
static CAP_HIT: AtomicBool = AtomicBool::new(false);
static EVAL_BUDGET_BYTES: AtomicUsize = AtomicUsize::new(512 * 1024 * 1024);
static ITER_PATHS_SCANNED: AtomicUsize = AtomicUsize::new(0);
static ITER_CANDIDATE_EVALS: AtomicUsize = AtomicUsize::new(0);
static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
static TIMING_ACC: Mutex<Option<Arc<Mutex<TimingAcc>>>> = Mutex::new(None);
static STAGE_MAP: OnceLock<Mutex<HashMap<StageKey, StageInfo>>> = OnceLock::new();
static STAGE_WATCHDOG: OnceLock<()> = OnceLock::new();
const EST_SIZE_CACHE_MAX: usize = 200_000;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct EstSizeKey {
    graph_id: u64,
    epoch: u64,
    node: NodeId,
}

static EST_SIZE_CACHE: OnceLock<Mutex<HashMap<EstSizeKey, usize>>> = OnceLock::new();

fn debug_logs_enabled() -> bool {
    DEBUG_OOM.load(Ordering::Relaxed)
}

fn example_stage_enabled() -> bool {
    debug_logs_enabled()
}

fn stage_warn_ms() -> u128 {
    static WARN_MS: OnceLock<u128> = OnceLock::new();
    *WARN_MS.get_or_init(|| {
        std::env::var("TRACE_STAGE_WARN_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STAGE_WARN_MS_DEFAULT)
    })
}

fn stage_warn_repeat_ms() -> u128 {
    static REPEAT_MS: OnceLock<u128> = OnceLock::new();
    *REPEAT_MS.get_or_init(|| {
        std::env::var("TRACE_STAGE_WARN_REPEAT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STAGE_WARN_REPEAT_MS_DEFAULT)
    })
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct StageKey {
    phase: &'static str,
    idx: usize,
    stage: &'static str,
    program: u32,
    path: Option<Vec<u8>>,
    depth: Option<usize>,
}

#[derive(Clone)]
struct StageInfo {
    key: StageKey,
    start: Instant,
    last_log: Option<Instant>,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
}

fn stage_map() -> &'static Mutex<HashMap<StageKey, StageInfo>> {
    STAGE_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn start_stage_watchdog() {
    if !example_stage_enabled() {
        return;
    }
    if stage_warn_ms() == 0 {
        return;
    }
    STAGE_WATCHDOG.get_or_init(|| {
        std::thread::spawn(|| loop {
            std::thread::sleep(Duration::from_millis(10));
            let warn_ms = stage_warn_ms();
            if warn_ms == 0 {
                continue;
            }
            let mut to_log: Vec<(StageInfo, Duration)> = Vec::new();
            {
                let mut map = stage_map().lock().unwrap();
                for info in map.values_mut() {
                    if !example_stage_enabled() {
                        continue;
                    }
                    let elapsed = info.start.elapsed();
                    if elapsed.as_millis() < warn_ms {
                        continue;
                    }
                    let repeat_ms = stage_warn_repeat_ms();
                    let should_log = match info.last_log {
                        None => true,
                        Some(last) => repeat_ms > 0 && last.elapsed().as_millis() >= repeat_ms,
                    };
                    if should_log {
                        info.last_log = Some(Instant::now());
                        to_log.push((info.clone(), elapsed));
                    }
                }
            }
            for (info, elapsed) in to_log {
                if !example_stage_enabled() {
                    continue;
                }
                let path_str = info
                    .key
                    .path
                    .as_ref()
                    .map(|p| path_to_string(p))
                    .unwrap_or_else(|| "<none>".to_string());
                let depth_str = info
                    .key
                    .depth
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "<none>".to_string());
                println!(
                    "STAGE_SLOW phase={} idx={} stage={} elapsed={:?} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
                    info.key.phase,
                    info.key.idx,
                    info.key.stage,
                    elapsed,
                    info.prog_size,
                    info.input_size,
                    info.expected_size,
                    info.key.program,
                    path_str,
                    depth_str
                );
            }
        });
    });
}

#[derive(Default)]
struct TimingAcc {
    total: Duration,
    reduce_actual: Duration,
    reduce_expected: Duration,
    ted: Duration,
    evals: usize,
    reduce_actual_hits: usize,
    reduce_expected_hits: usize,
}

#[derive(Clone, Default)]
struct DebugContext {
    program: Option<NodeId>,
    path: Option<Vec<u8>>,
    depth: Option<usize>,
}

#[derive(Clone)]
struct PeakInfo {
    nodes: usize,
    phase: &'static str,
    steps: usize,
    step_limit: usize,
    applied_size: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
    program: Option<NodeId>,
    path: Option<Vec<u8>>,
    depth: Option<usize>,
}

static DEBUG_CONTEXT: Mutex<DebugContext> = Mutex::new(DebugContext {
    program: None,
    path: None,
    depth: None,
});
static PEAK_INFO: Mutex<Option<PeakInfo>> = Mutex::new(None);

fn set_debug_oom(enabled: bool) {
    DEBUG_OOM.store(enabled, Ordering::Relaxed);
}

fn set_timing_acc(acc: Option<Arc<Mutex<TimingAcc>>>) {
    if acc.is_some() {
        TIMING_ENABLED.store(true, Ordering::Relaxed);
    } else {
        TIMING_ENABLED.store(false, Ordering::Relaxed);
    }
    let mut slot = TIMING_ACC.lock().unwrap();
    *slot = acc;
}

fn with_timing_acc<F: FnOnce(&Arc<Mutex<TimingAcc>>)>(f: F) {
    if !TIMING_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let acc = {
        let slot = TIMING_ACC.lock().unwrap();
        slot.clone()
    };
    if let Some(acc) = acc {
        f(&acc);
    }
}

fn log_slow_eval(
    phase: &'static str,
    elapsed: Duration,
    nodes: usize,
    steps: usize,
    step_limit: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
) {
    if !debug_logs_enabled() {
        return;
    }
    if elapsed.as_millis() < SLOW_EVAL_MS {
        return;
    }
    let ctx = get_debug_context();
    println!(
        "EVAL_SLOW phase={} elapsed={:?} nodes={} steps={} step_limit={} prog_size={} input_size={} expected_size={} program={:?} path={} depth={}",
        phase,
        elapsed,
        nodes,
        steps,
        step_limit,
        prog_size,
        input_size,
        expected_size,
        ctx.program.map(|p| p.0),
        ctx.path.as_ref().map(|p| path_to_string(p)).unwrap_or_else(|| "<none>".to_string()),
        ctx.depth.map(|d| d.to_string()).unwrap_or_else(|| "<none>".to_string()),
    );
}

fn log_slow_example(
    phase: &'static str,
    idx: usize,
    elapsed: Duration,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
) {
    if !debug_logs_enabled() {
        return;
    }
    if elapsed.as_millis() < SLOW_EVAL_MS {
        return;
    }
    let ctx = get_debug_context();
    println!(
        "EXAMPLE_SLOW phase={} idx={} elapsed={:?} prog_size={} input_size={} expected_size={} program={:?} path={} depth={}",
        phase,
        idx,
        elapsed,
        prog_size,
        input_size,
        expected_size,
        ctx.program.map(|p| p.0),
        ctx.path.as_ref().map(|p| path_to_string(p)).unwrap_or_else(|| "<none>".to_string()),
        ctx.depth.map(|d| d.to_string()).unwrap_or_else(|| "<none>".to_string()),
    );
}

fn set_eval_budget_bytes(bytes: usize) {
    EVAL_BUDGET_BYTES.store(bytes.max(1), Ordering::Relaxed);
}

fn reset_cap_hit() {
    CAP_HIT.store(false, Ordering::Relaxed);
}

fn mark_cap_hit() {
    CAP_HIT.store(true, Ordering::Relaxed);
}

fn cap_hit() -> bool {
    CAP_HIT.load(Ordering::Relaxed)
}

fn clear_cap_hit_if_allowed() -> bool {
    if !cap_hit() {
        return false;
    }
    if should_abort_on_penalty() {
        return true;
    }
    reset_cap_hit();
    false
}

fn reset_iter_counters() {
    ITER_PATHS_SCANNED.store(0, Ordering::Relaxed);
    ITER_CANDIDATE_EVALS.store(0, Ordering::Relaxed);
}

fn bump_iter_paths_scanned(n: usize) {
    if n > 0 {
        ITER_PATHS_SCANNED.fetch_add(n, Ordering::Relaxed);
    }
}

fn bump_iter_candidate_evals(n: usize) {
    if n > 0 {
        ITER_CANDIDATE_EVALS.fetch_add(n, Ordering::Relaxed);
    }
}

fn snapshot_iter_counters() -> (usize, usize) {
    (
        ITER_PATHS_SCANNED.load(Ordering::Relaxed),
        ITER_CANDIDATE_EVALS.load(Ordering::Relaxed),
    )
}

fn set_debug_context(program: Option<NodeId>, path: Option<&[u8]>, depth: Option<usize>) {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let mut ctx = DEBUG_CONTEXT.lock().unwrap();
    ctx.program = program;
    ctx.path = path.map(|p| p.to_vec());
    ctx.depth = depth;
}

fn get_debug_context() -> DebugContext {
    DEBUG_CONTEXT.lock().unwrap().clone()
}

fn reset_peak() {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let mut peak = PEAK_INFO.lock().unwrap();
    *peak = None;
}

fn update_peak(
    phase: &'static str,
    nodes: usize,
    steps: usize,
    step_limit: usize,
    applied_size: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
    ctx: DebugContext,
) {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let mut peak = PEAK_INFO.lock().unwrap();
    let replace = match &*peak {
        Some(info) => nodes > info.nodes,
        None => true,
    };
    if replace {
        *peak = Some(PeakInfo {
            nodes,
            phase,
            steps,
            step_limit,
            applied_size,
            prog_size,
            input_size,
            expected_size,
            program: ctx.program,
            path: ctx.path,
            depth: ctx.depth,
        });
    }
}

fn report_peak(iter: usize) {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let peak = PEAK_INFO.lock().unwrap().clone();
    if let Some(info) = peak {
        let path_str = info
            .path
            .as_ref()
            .map(|p| path_to_string(p))
            .unwrap_or_else(|| "<none>".to_string());
        let program_str = info
            .program
            .map(|p| p.0.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        let depth_str = info
            .depth
            .map(|d| d.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        eprintln!(
            "ITER_PEAK iter={} nodes={} phase={} steps={} step_limit={} applied_size={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
            iter,
            info.nodes,
            info.phase,
            info.steps,
            info.step_limit,
            info.applied_size,
            info.prog_size,
            info.input_size,
            info.expected_size,
            program_str,
            path_str,
            depth_str
        );
    }
}

fn should_abort_on_penalty() -> bool {
    DEBUG_OOM.load(Ordering::Relaxed)
}

fn log_node_limit_hit(
    phase: &'static str,
    nodes: usize,
    max_nodes: usize,
    steps: usize,
    step_limit: usize,
    applied_size: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "NODE_LIMIT_HIT phase={} nodes={} max_nodes={} steps={} step_limit={} applied_size={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
        phase,
        nodes,
        max_nodes,
        steps,
        step_limit,
        applied_size,
        prog_size,
        input_size,
        expected_size,
        program_str,
        path_str,
        depth_str
    );
    update_peak(
        phase,
        nodes,
        steps,
        step_limit,
        applied_size,
        prog_size,
        input_size,
        expected_size,
        ctx,
    );
}

fn log_step_limit_hit(
    phase: &'static str,
    nodes: usize,
    max_nodes: usize,
    steps: usize,
    step_limit: usize,
    applied_size: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "STEP_LIMIT_HIT phase={} nodes={} max_nodes={} steps={} step_limit={} applied_size={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
        phase,
        nodes,
        max_nodes,
        steps,
        step_limit,
        applied_size,
        prog_size,
        input_size,
        expected_size,
        program_str,
        path_str,
        depth_str
    );
    update_peak(
        phase,
        nodes,
        steps,
        step_limit,
        applied_size,
        prog_size,
        input_size,
        expected_size,
        ctx,
    );
}

fn scaled_eval_budget(max_steps: usize, size: usize) -> usize {
    let mut budget = max_steps.saturating_mul(size.max(1));
    if budget > EVAL_STEP_CAP {
        budget = EVAL_STEP_CAP;
    }
    budget
}

fn log_ted_cap_hit(actual_size: usize, expected_size: usize, cap: usize) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let cells = (actual_size as u128) * (expected_size as u128);
    eprintln!(
        "TED_CAP_HIT actual_size={} expected_size={} cells={} cap={} program={} path={} depth={}",
        actual_size,
        expected_size,
        cells,
        cap,
        program_str,
        path_str,
        depth_str
    );
}

fn log_preflight_limit_hit(
    phase: &'static str,
    total: usize,
    max_nodes: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "PREFLIGHT_LIMIT_HIT phase={} total={} max_nodes={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
        phase,
        total,
        max_nodes,
        prog_size,
        input_size,
        expected_size,
        program_str,
        path_str,
        depth_str
    );
}

fn log_example_stage(
    phase: &'static str,
    idx: usize,
    stage: &'static str,
    status: &'static str,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
    program: NodeId,
    path: Option<&[u8]>,
    depth: Option<usize>,
    elapsed: Option<Duration>,
) {
    if !example_stage_enabled() {
        return;
    }
    let path_str = path
        .map(path_to_string)
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let key = StageKey {
        phase,
        idx,
        stage,
        program: program.0,
        path: path.map(|p| p.to_vec()),
        depth,
    };
    if status == "begin" {
        if stage_warn_ms() > 0 {
            start_stage_watchdog();
            let info = StageInfo {
                key: key.clone(),
                start: Instant::now(),
                last_log: None,
                prog_size,
                input_size,
                expected_size,
            };
            stage_map().lock().unwrap().insert(key.clone(), info);
        }
    } else if status == "end" {
        if stage_warn_ms() > 0 {
            stage_map().lock().unwrap().remove(&key);
        }
    }
    if let Some(elapsed) = elapsed {
        println!(
            "EXAMPLE_STAGE phase={} idx={} stage={} status={} elapsed={:?} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
            phase,
            idx,
            stage,
            status,
            elapsed,
            prog_size,
            input_size,
            expected_size,
            program.0,
            path_str,
            depth_str
        );
    } else {
        println!(
            "EXAMPLE_STAGE phase={} idx={} stage={} status={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
            phase,
            idx,
            stage,
            status,
            prog_size,
            input_size,
            expected_size,
            program.0,
            path_str,
            depth_str
        );
    }
}

fn log_stage_slow(
    phase: &'static str,
    idx: usize,
    stage: &'static str,
    elapsed: Duration,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
    program: NodeId,
    path: Option<&[u8]>,
    depth: Option<usize>,
) {
    if !example_stage_enabled() {
        return;
    }
    let warn_ms = stage_warn_ms();
    if warn_ms == 0 || elapsed.as_millis() < warn_ms {
        return;
    }
    let path_str = path
        .map(path_to_string)
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    println!(
        "STAGE_SLOW phase={} idx={} stage={} elapsed={:?} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
        phase,
        idx,
        stage,
        elapsed,
        prog_size,
        input_size,
        expected_size,
        program.0,
        path_str,
        depth_str
    );
}

fn log_repr_cap_hit(phase: &'static str, size: usize, max_nodes: usize) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "REPR_CAP_HIT phase={} size={} max_nodes={} program={} path={} depth={}",
        phase, size, max_nodes, program_str, path_str, depth_str
    );
}

fn log_path_cap_hit(phase: &'static str, count: usize, max_paths: usize) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "PATH_CAP_HIT phase={} count={} max_paths={} program={} path={} depth={}",
        phase, count, max_paths, program_str, path_str, depth_str
    );
}

fn log_eval_budget_hit(
    phase: &'static str,
    bytes: usize,
    budget: usize,
    prog_size: usize,
    input_size: usize,
    expected_size: usize,
    max_ted_cells: usize,
) {
    mark_cap_hit();
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let ctx = get_debug_context();
    let path_str = ctx
        .path
        .as_ref()
        .map(|p| path_to_string(p))
        .unwrap_or_else(|| "<none>".to_string());
    let program_str = ctx
        .program
        .map(|p| p.0.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let depth_str = ctx
        .depth
        .map(|d| d.to_string())
        .unwrap_or_else(|| "<none>".to_string());
    eprintln!(
        "EVAL_BUDGET_HIT phase={} bytes={} budget={} max_ted_cells={} prog_size={} input_size={} expected_size={} program={} path={} depth={}",
        phase,
        bytes,
        budget,
        max_ted_cells,
        prog_size,
        input_size,
        expected_size,
        program_str,
        path_str,
        depth_str
    );
}

fn preflight_eval_limit(
    g: &Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_nodes: usize,
    max_ted_cells: usize,
    phase: &'static str,
) -> Option<(usize, usize, usize)> {
    let _ = (max_nodes, max_ted_cells, phase);
    let prog_size = estimate_size(g, program);
    let input_size = estimate_size(g, input);
    let expected_size = estimate_size(g, expected);
    Some((prog_size, input_size, expected_size))
}

fn preflight_eval_limit_with_replacement(
    g: &Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    input: NodeId,
    expected: NodeId,
    max_nodes: usize,
    max_ted_cells: usize,
    phase: &'static str,
) -> bool {
    let _ = (g, program, path, replacement, input, expected, max_nodes, max_ted_cells, phase);
    true
}

fn clone_subtree(
    g_src: &Graph,
    g_dst: &mut Graph,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
    eval_to_orig: Option<&mut HashMap<NodeId, NodeId>>,
) -> NodeId {
    let resolved = g_src.resolve(id);
    if let Some(&cached) = memo.get(&resolved) {
        return cached;
    }

    let mut eval_to_orig = eval_to_orig;
    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    stack.push((resolved, false));

    while let Some((node, expanded)) = stack.pop() {
        let node = g_src.resolve(node);
        if memo.contains_key(&node) {
            continue;
        }
        if expanded {
            let new_id = match g_src.get(node) {
                Node::Leaf => g_dst.add(Node::Leaf),
                Node::Prim(p) => g_dst.add(Node::Prim(*p)),
                Node::Float(f) => g_dst.add(Node::Float(*f)),
                Node::Handle(h) => g_dst.add(Node::Handle(*h)),
                Node::Ind(inner) => {
                    let inner = g_src.resolve(*inner);
                    memo.get(&inner).copied().unwrap_or_else(|| g_dst.add(Node::Leaf))
                }
                Node::Stem(inner) => {
                    let inner = g_src.resolve(*inner);
                    let c = memo.get(&inner).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    g_dst.add(Node::Stem(c))
                }
                Node::Fork(l, r) => {
                    let l = g_src.resolve(*l);
                    let r = g_src.resolve(*r);
                    let nl = memo.get(&l).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    let nr = memo.get(&r).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    g_dst.add(Node::Fork(nl, nr))
                }
                Node::App { func, args } => {
                    let func = g_src.resolve(*func);
                    let nf = memo.get(&func).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
                    for arg in args {
                        let arg = g_src.resolve(*arg);
                        let na = memo.get(&arg).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                        new_args.push(na);
                    }
                    g_dst.add(Node::App { func: nf, args: new_args })
                }
            };
            memo.insert(node, new_id);
            if let Some(map) = eval_to_orig.as_deref_mut() {
                map.insert(new_id, node);
            }
        } else {
            stack.push((node, true));
            match g_src.get(node) {
                Node::Stem(inner) => stack.push((g_src.resolve(*inner), false)),
                Node::Fork(l, r) => {
                    stack.push((g_src.resolve(*r), false));
                    stack.push((g_src.resolve(*l), false));
                }
                Node::App { func, args } => {
                    for arg in args.iter().rev() {
                        stack.push((g_src.resolve(*arg), false));
                    }
                    stack.push((g_src.resolve(*func), false));
                }
                Node::Ind(inner) => stack.push((g_src.resolve(*inner), false)),
                _ => {}
            }
        }
    }

    memo.get(&resolved).copied().unwrap_or_else(|| g_dst.add(Node::Leaf))
}

fn check_global_limit(g: &Graph, config: &CounterfactualConfig, phase: &'static str) -> bool {
    if config.max_global_nodes == 0 {
        return false;
    }
    let nodes = g.nodes.len();
    if nodes <= config.max_global_nodes {
        return false;
    }
    mark_cap_hit();
    if config.debug {
        let ctx = get_debug_context();
        let path_str = ctx
            .path
            .as_ref()
            .map(|p| path_to_string(p))
            .unwrap_or_else(|| "<none>".to_string());
        let program_str = ctx
            .program
            .map(|p| p.0.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        let depth_str = ctx
            .depth
            .map(|d| d.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        eprintln!(
            "GLOBAL_NODE_LIMIT_HIT phase={} nodes={} max_nodes={} program={} path={} depth={}",
            phase, nodes, config.max_global_nodes, program_str, path_str, depth_str
        );
    }
    true
}

#[derive(Debug, Clone)]
pub struct CounterfactualConfig {
    pub max_iterations: usize,
    // Hard cap on reduction steps per evaluation (no scaling).
    pub max_eval_steps: usize,
    pub max_eval_nodes: usize,
    pub max_global_nodes: usize,
    pub max_ted_cells: usize,
    pub edit_threshold_base: f64,
    pub variance_weight: f64,
    pub output_collision_penalty: f64,
    pub max_edits_per_iter: usize,
    pub max_edits_per_path: usize,
    pub max_replacement_nodes: usize,
    pub max_edit_depth: usize,
    pub max_candidate_evals: usize,
    pub max_candidates_per_iter: usize,
    pub memo_budget_bytes: usize,
    pub verbose: bool,
    pub debug: bool,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_eval_steps: 5000,
            max_eval_nodes: 100_000,
            max_global_nodes: 2_000_000,
            max_ted_cells: 50_000_000,
            edit_threshold_base: 0.5,
            variance_weight: 0.5,
            output_collision_penalty: 100.0,
            max_edits_per_iter: 10,
            max_edits_per_path: 1000,
            max_replacement_nodes: 1024,
            max_edit_depth: 5,
            max_candidate_evals: 100_000,
            max_candidates_per_iter: 5_000,
            memo_budget_bytes: 512 * 1024 * 1024,
            verbose: true,
            debug: false,
        }
    }
}

#[derive(Debug, Clone)]
struct EditPlan {
    path: Vec<u8>,
    replacement: NodeId,
    replacement_hash: u64,
    improvement: f64,
    depth: usize,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct EditKey {
    path: Vec<u8>,
    replacement_hash: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct LossKey {
    program_hash: u64,
    input_hash: u64,
    expected_hash: u64,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
}

#[derive(Clone, Copy)]
struct LossEntry {
    loss: f64,
    actual_hash: u64,
}

#[derive(Clone, Copy)]
struct LossSample {
    loss: f64,
    actual_hash: u64,
    expected_hash: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct CandidateKey {
    program_hash: u64,
    path: Vec<u8>,
    replacement_hash: u64,
    examples_hash: u64,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
}


#[derive(Clone, Copy)]
struct CandidateStats {
    mean: f64,
    var: f64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct ReduceKey {
    root_hash: u64,
    max_steps: usize,
    max_nodes: usize,
}

#[derive(Clone, Copy)]
struct Replacement {
    node: NodeId,
    hash: u64,
}

struct CacheEntry<V: Clone> {
    value: V,
    id: u64,
    size: usize,
}

struct LruCache<K: Eq + std::hash::Hash + Clone, V: Clone> {
    map: HashMap<K, CacheEntry<V>>,
    order: std::collections::VecDeque<(K, u64)>,
    bytes: usize,
    budget: usize,
    counter: u64,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> LruCache<K, V> {
    fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            bytes: 0,
            budget,
            counter: 0,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.map.get_mut(key) {
            let id = self.counter;
            self.counter = self.counter.wrapping_add(1);
            entry.id = id;
            self.order.push_back((key.clone(), id));
            return Some(entry.value.clone());
        }
        None
    }

    fn insert(&mut self, key: K, value: V, size: usize) {
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

struct MemoCaches {
    loss: std::sync::Mutex<LruCache<LossKey, LossEntry>>,
    candidate: std::sync::Mutex<LruCache<CandidateKey, CandidateStats>>,
    reduce: std::sync::Mutex<LruCache<ReduceKey, NodeId>>,
    redex: std::sync::Mutex<LruCache<RedexKey, std::sync::Arc<CachedRedex>>>,
    redex_budget: usize,
    max_cached_nodes: usize,
}

impl MemoCaches {
    fn new(budget: usize, max_cached_nodes: usize) -> Self {
        let per = budget / 5;
        Self {
            loss: std::sync::Mutex::new(LruCache::new(per)),
            candidate: std::sync::Mutex::new(LruCache::new(per)),
            reduce: std::sync::Mutex::new(LruCache::new(per)),
            redex: std::sync::Mutex::new(LruCache::new(per)),
            redex_budget: per,
            max_cached_nodes,
        }
    }

    fn get_loss(&self, key: &LossKey) -> Option<LossEntry> {
        let mut cache = self.loss.lock().unwrap();
        cache.get(key)
    }

    fn insert_loss(&self, key: LossKey, value: LossEntry, size: usize) {
        let mut cache = self.loss.lock().unwrap();
        cache.insert(key, value, size);
    }

    fn get_candidate(&self, key: &CandidateKey) -> Option<CandidateStats> {
        let mut cache = self.candidate.lock().unwrap();
        cache.get(key)
    }

    fn insert_candidate(&self, key: CandidateKey, value: CandidateStats, size: usize) {
        let mut cache = self.candidate.lock().unwrap();
        cache.insert(key, value, size);
    }

    fn get_reduce(&self, key: &ReduceKey) -> Option<NodeId> {
        let mut cache = self.reduce.lock().unwrap();
        cache.get(key)
    }

    fn insert_reduce(&self, key: ReduceKey, value: NodeId, size: usize) {
        let mut cache = self.reduce.lock().unwrap();
        cache.insert(key, value, size);
    }
}

impl RedexMemo for MemoCaches {
    fn get_redex(&self, key: &RedexKey) -> Option<std::sync::Arc<CachedRedex>> {
        let mut cache = self.redex.lock().unwrap();
        cache.get(key)
    }

    fn insert_redex(&self, key: RedexKey, value: std::sync::Arc<CachedRedex>, size: usize, nodes: usize) {
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

fn threshold_for_blame(base: f64, blame: f64, total_blame: f64) -> f64 {
    let denom = if total_blame > 0.0 { total_blame } else { 1.0 };
    let prop = (blame / denom).clamp(0.0, 1.0);
    let decay = 7.0;
    let t = base * (-(decay * prop)).exp();
    if t < 1e-6 { 0.0 } else { t }
}

fn hash_examples(g: &Graph, examples: &[(NodeId, NodeId)]) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    for (input, expected) in examples {
        let input_hash = tree_hash(g, *input);
        let expected_hash = tree_hash(g, *expected);
        h.write_u64(input_hash);
        h.write_u64(expected_hash);
    }
    h.finish()
}

fn loss_entry_size(_key: &LossKey) -> usize {
    48
}

fn apply_output_collision_penalty(losses: &mut [f64], samples: &[LossSample], penalty: f64) {
    if penalty <= 0.0 {
        return;
    }
    let mut expected_by_actual: HashMap<u64, HashSet<u64>> = HashMap::new();
    for sample in samples {
        expected_by_actual
            .entry(sample.actual_hash)
            .or_insert_with(HashSet::new)
            .insert(sample.expected_hash);
    }
    let mut penalty_by_actual: HashMap<u64, f64> = HashMap::new();
    for (actual_hash, expected_set) in expected_by_actual {
        if expected_set.len() > 1 {
            let bump = penalty * (expected_set.len() as f64 - 1.0);
            penalty_by_actual.insert(actual_hash, bump);
        }
    }
    for (idx, sample) in samples.iter().enumerate() {
        if let Some(bump) = penalty_by_actual.get(&sample.actual_hash) {
            losses[idx] += *bump;
        }
    }
}

fn candidate_entry_size(key: &CandidateKey) -> usize {
    64 + key.path.len()
}

fn reduce_entry_size(nodes: usize) -> usize {
    32 * nodes
}

fn make_replacement(g: &Graph, node: NodeId) -> Replacement {
    Replacement {
        node,
        hash: tree_hash(g, node),
    }
}

fn unparse_limited(g: &Graph, id: NodeId) -> String {
    let mut nodes_left = UNPARSE_MAX_NODES;
    let mut visiting: HashSet<NodeId> = HashSet::new();
    unparse_limited_rec(g, id, UNPARSE_MAX_DEPTH, &mut nodes_left, &mut visiting)
}

fn unparse_limited_rec(
    g: &Graph,
    id: NodeId,
    depth: usize,
    nodes_left: &mut usize,
    visiting: &mut HashSet<NodeId>,
) -> String {
    enum Frame<'a> {
        Enter(NodeId, usize),
        Exit(NodeId),
        Text(&'a str),
        Owned(String),
    }

    let mut out = String::new();
    let mut stack: Vec<Frame<'_>> = Vec::new();
    stack.push(Frame::Enter(id, depth));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Text(s) => out.push_str(s),
            Frame::Owned(s) => out.push_str(&s),
            Frame::Exit(id) => {
                visiting.remove(&id);
            }
            Frame::Enter(curr, curr_depth) => {
                if *nodes_left == 0 || curr_depth == 0 {
                    out.push_str("...");
                    continue;
                }
                let resolved = g.resolve(curr);
                if !visiting.insert(resolved) {
                    out.push_str("<cycle>");
                    continue;
                }
                *nodes_left = nodes_left.saturating_sub(1);
                stack.push(Frame::Exit(resolved));
                match g.get(resolved) {
                    Node::Leaf => out.push_str("n"),
                    Node::Stem(inner) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(*inner, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Text("n"));
                        stack.push(Frame::Text("("));
                    }
                    Node::Fork(l, r) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(*r, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Enter(*l, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Text("n"));
                        stack.push(Frame::Text("("));
                    }
                    Node::App { func, args } => {
                        let limit = UNPARSE_MAX_ARGS.min(args.len());
                        stack.push(Frame::Text(")"));
                        if args.len() > UNPARSE_MAX_ARGS {
                            stack.push(Frame::Text(" ..."));
                        }
                        for arg in args.iter().take(limit).rev() {
                            stack.push(Frame::Enter(*arg, curr_depth - 1));
                            stack.push(Frame::Text(" "));
                        }
                        stack.push(Frame::Enter(*func, curr_depth - 1));
                        stack.push(Frame::Text("("));
                    }
                    Node::Prim(p) => out.push_str(&format!("{:?}", p)),
                    Node::Float(f) => out.push_str(&format!("{}", f)),
                    Node::Handle(h) => out.push_str(&format!("<Handle {}>", h)),
                    Node::Ind(inner) => stack.push(Frame::Enter(*inner, curr_depth - 1)),
                }
            }
        }
    }

    out
}

pub fn synthesize(
    g: &mut Graph,
    examples: Vec<(NodeId, NodeId)>,
    config: CounterfactualConfig,
) -> Option<NodeId> {
    if examples.is_empty() {
        return None;
    }

    let examples_hash = hash_examples(g, &examples);
    let caches = MemoCaches::new(config.memo_budget_bytes, config.max_replacement_nodes);
    set_debug_oom(config.debug);
    if config.debug {
        set_reduce_progress_ms_override(None);
    } else {
        set_reduce_progress_ms_override(Some(0));
    }
    let eval_budget = 512 * 1024 * 1024;
    set_eval_budget_bytes(eval_budget);
    let seed = select_seed(g, &examples, examples_hash, &config, &caches);
    let seed_candidates = {
        let leaf = g.add(Node::Leaf);
        vec![leaf]
    };
    let mut seed_cursor = seed_candidates
        .iter()
        .position(|&s| s == seed)
        .unwrap_or(0);
    let mut current = seed;
    let mut best_program = seed;
    let mut best_score = f64::INFINITY;
    let mut best_mean = f64::INFINITY;
    let mut best_var = f64::INFINITY;
    let mut history: Vec<(NodeId, u64, EditKey)> = Vec::new();
    let mut banned_by_program: HashMap<u64, HashSet<EditKey>> = HashMap::new();
    let mut banned_paths_by_program: HashMap<u64, HashSet<Vec<u8>>> = HashMap::new();
    if config.verbose {
        println!("Counterfactual learner: starting seed {}", seed.0);
    }

    for iter in 0..config.max_iterations {
        reset_peak();
        reset_cap_hit();
        reset_iter_counters();
        set_debug_context(Some(current), None, None);
        let current_hash = tree_hash(g, current);
        let iter_start = Instant::now();
        let (losses, active_paths) = evaluate_with_traces(
            g,
            current,
            &examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            Some(&caches),
        );
        let eval_time = iter_start.elapsed();
        if clear_cap_hit_if_allowed() {
            if let Some((parent, parent_hash, last_edit)) = history.pop() {
                banned_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.clone());
                banned_paths_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.path.clone());
                banned_paths_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.path.clone());
                if config.verbose || config.debug {
                    println!(
                        "  Cap hit; backtracking and banning edit {} -> {}",
                        path_to_string(&last_edit.path),
                        last_edit.replacement_hash
                    );
                }
                current = parent;
                report_peak(iter);
                continue;
            } else {
                if config.verbose || config.debug {
                    println!("  Cap hit; stopping (no parent).");
                }
                report_peak(iter);
                break;
            }
        }
        let base_mean = mean(&losses);
        let base_var = variance(&losses);
        let base_score = base_mean + config.variance_weight * base_var;

        if base_score < best_score {
            best_score = base_score;
            best_mean = base_mean;
            best_var = base_var;
            best_program = current;
            if config.verbose {
                println!(
                    "  Best so far: mean_loss={:.4} var={:.4} score={:.4}",
                    best_mean, best_var, best_score
                );
            }
        }

        if config.verbose {
            println!(
                "Iter {}: mean_loss={:.4} var={:.4}",
                iter, base_mean, base_var
            );
        }
        if config.verbose {
            println!("  Program: {}", unparse_limited(g, current));
            let total_events: usize = active_paths.iter().map(|v| v.len()).sum();
            println!("  Trace events total: {}", total_events);
            for (idx, loss) in losses.iter().enumerate() {
                let events = active_paths.get(idx).map(|v| v.len()).unwrap_or(0);
                println!("  Example {}: loss={:.4} events={}", idx, loss, events);
            }
            println!("  Eval time: {:?}\n", eval_time);
        }

        if base_mean <= 1e-9 {
            return Some(current);
        }

        let banned = banned_by_program.get(&current_hash);
        let banned_paths = banned_paths_by_program.get(&current_hash);
        reset_cap_hit();
        let edit = candidate_edits(
            g,
            current,
            &examples,
            &losses,
            &active_paths,
            base_mean,
            base_var,
            &config,
            config.verbose,
            banned,
            banned_paths,
            examples_hash,
            &caches,
        );
        if config.debug {
            let (paths_scanned, candidate_evals) = snapshot_iter_counters();
            println!(
                "  Iteration stats: examples={} paths_scanned={} candidate_evals={}",
                examples.len(),
                paths_scanned,
                candidate_evals
            );
        }

        if clear_cap_hit_if_allowed() {
            if let Some((parent, parent_hash, last_edit)) = history.pop() {
                banned_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.clone());
                if config.verbose || config.debug {
                    println!(
                        "  Cap hit during edit search; backtracking and banning edit {} -> {}",
                        path_to_string(&last_edit.path),
                        last_edit.replacement_hash
                    );
                }
                current = parent;
                report_peak(iter);
                continue;
            } else {
                if config.verbose || config.debug {
                    println!("  Cap hit during edit search; stopping (no parent).");
                }
                report_peak(iter);
                break;
            }
        }

        let edit = match edit {
            Some(edit) => edit,
            None => {
                if let Some((parent, parent_hash, last_edit)) = history.pop() {
                    banned_by_program
                        .entry(parent_hash)
                        .or_insert_with(HashSet::new)
                        .insert(last_edit.clone());
                    banned_paths_by_program
                        .entry(parent_hash)
                        .or_insert_with(HashSet::new)
                        .insert(last_edit.path.clone());
                    if config.verbose || config.debug {
                        println!(
                            "  No edits (max depth reached); backtracking and banning edit {} -> {}",
                            path_to_string(&last_edit.path),
                            last_edit.replacement_hash
                        );
                    }
                    current = parent;
                    report_peak(iter);
                    continue;
                } else {
                    let mut restart = None;
                    for _ in 0..seed_candidates.len() {
                        seed_cursor = (seed_cursor + 1) % seed_candidates.len();
                        let cand = seed_candidates[seed_cursor];
                        if cand != current {
                            restart = Some(cand);
                            break;
                        }
                    }
                    if let Some(cand) = restart {
                        if config.verbose || config.debug {
                            println!("  No edits found; restarting from seed {}", cand.0);
                        }
                        current = cand;
                        report_peak(iter);
                        continue;
                    }
                    if config.verbose || config.debug {
                        println!("  No edits found; stopping.");
                    }
                    report_peak(iter);
                    break;
                }
            }
        };

        if config.verbose {
            println!(
                "  Chosen edit:\n    path {} depth={} improvement={:.6} replacement={}",
                path_to_string(&edit.path),
                edit.depth,
                edit.improvement,
                unparse_limited(g, edit.replacement)
            );
        }

        set_debug_context(Some(current), Some(&edit.path), Some(edit.depth));
        if check_global_limit(g, &config, "apply_replacement") {
            return Some(best_program);
        }
        let child = replace_at_path(g, current, &edit.path, edit.replacement);
        if check_global_limit(g, &config, "apply_edit") {
            return Some(best_program);
        }
        let child_stats = program_stats_cached(
            g,
            child,
            &examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            &caches,
        );
        if clear_cap_hit_if_allowed() {
            if config.verbose || config.debug {
                println!(
                    "  Cap hit during child eval; banning edit {} -> {}",
                    path_to_string(&edit.path),
                    edit.replacement_hash
                );
            }
            banned_by_program
                .entry(current_hash)
                .or_insert_with(HashSet::new)
                .insert(EditKey {
                    path: edit.path.clone(),
                    replacement_hash: edit.replacement_hash,
                });
            report_peak(iter);
            continue;
        }
        let child_score = child_stats.mean + config.variance_weight * child_stats.var;
        if child_stats.mean > base_mean && child_stats.var > base_var || child_score > base_score {
            if config.verbose || config.debug {
                println!(
                    "  Rejected edit (worse): score {:.4}->{:.4} mean {:.4}->{:.4} var {:.4}->{:.4}",
                    base_score,
                    child_score,
                    base_mean,
                    child_stats.mean,
                    base_var,
                    child_stats.var
                );
            }
            banned_by_program
                .entry(current_hash)
                .or_insert_with(HashSet::new)
                .insert(EditKey {
                    path: edit.path.clone(),
                    replacement_hash: edit.replacement_hash,
                });
            report_peak(iter);
            continue;
        }
        history.push((
            current,
            current_hash,
            EditKey {
                path: edit.path.clone(),
                replacement_hash: edit.replacement_hash,
            },
        ));
        current = child;
        report_peak(iter);
    }

    if config.debug {
        println!(
            "  Final best: mean_loss={:.4} var={:.4} score={:.4}",
            best_mean, best_var, best_score
        );
        println!("  Best program: {}", unparse_limited(g, best_program));
    }
    Some(best_program)
}

fn select_best_edit(edits: Vec<EditPlan>) -> Option<EditPlan> {
    edits.into_iter().max_by(|a, b| {
        a.improvement
            .partial_cmp(&b.improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn candidate_edits(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    losses: &[f64],
    active_paths: &[Vec<Vec<u8>>],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    log_search: bool,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> Option<EditPlan> {
    set_debug_context(Some(program), None, None);
    if clear_cap_hit_if_allowed() {
        return None;
    }
    let total_loss: f64 = losses.iter().sum();
    let path_map = collect_paths_map(g, program);
    if config.max_eval_nodes > 0 && path_map.len() > config.max_eval_nodes {
        log_path_cap_hit("collect_paths_map", path_map.len(), config.max_eval_nodes);
        return None;
    }
    let mut all_program_paths: HashSet<Vec<u8>> = HashSet::new();
    for paths in path_map.values() {
        for p in paths {
            all_program_paths.insert(p.clone());
        }
    }
    if config.max_eval_nodes > 0 && all_program_paths.len() > config.max_eval_nodes {
        log_path_cap_hit(
            "collect_program_paths",
            all_program_paths.len(),
            config.max_eval_nodes,
        );
        return None;
    }
    let mut path_blame: HashMap<Vec<u8>, f64> = HashMap::new();
    for (idx, paths) in active_paths.iter().enumerate() {
        let loss = losses[idx];
        if loss <= 0.0 {
            continue;
        }
        let events = paths.len();
        if events == 0 {
            continue;
        }
        let per_event = loss / (events as f64);
        for path in paths {
            *path_blame.entry(path.clone()).or_insert(0.0) += per_event;
        }
    }

    if path_blame.is_empty() {
        if total_loss > 0.0 {
            eprintln!(
                "ERROR: zero blame with nonzero loss; trace_events={}",
                active_paths.iter().map(|v| v.len()).sum::<usize>()
            );
            panic!("Zero blame with nonzero loss");
        }
        return None;
    }

    let mut all_paths: HashSet<Vec<u8>> = HashSet::new();
    for path in path_blame.keys() {
        let mut p = path.clone();
        loop {
            all_paths.insert(p.clone());
            if p.is_empty() {
                break;
            }
            p.pop();
        }
    }

    let mut ordered_paths: Vec<Vec<u8>> = all_paths.into_iter().collect();
    ordered_paths.sort_by(|a, b| b.len().cmp(&a.len()));

    if log_search {
        println!("  Candidate program: {}", unparse_limited(g, program));
        let total_events: usize = active_paths.iter().map(|v| v.len()).sum();
        println!("  Trace events total: {}", total_events);
        for (idx, loss) in losses.iter().enumerate() {
            let events = active_paths.get(idx).map(|v| v.len()).unwrap_or(0);
            println!("  Example {}: loss={:.4} events={}", idx, loss, events);
        }
        println!("  mean_loss={:.4} var={:.4}", base_mean, base_var);
        let mut blame_list: Vec<(Vec<u8>, f64)> = path_blame
            .iter()
            .map(|(p, b)| (p.clone(), *b))
            .collect();
        blame_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Blame map (path -> blame):");
        for (p, b) in blame_list {
            println!("    {} -> {:.6}", path_to_string(&p), b);
        }
    }

    let mut all_program_paths: Vec<Vec<u8>> = all_program_paths.into_iter().collect();
    all_program_paths.sort_by(|a, b| b.len().cmp(&a.len()));

    let base_path_blame = path_blame.clone();
    let total_blame: f64 = base_path_blame.values().sum();
    let max_depth = config.max_edit_depth.max(1);
    for depth in 1..=max_depth {
        if log_search {
            println!("  Candidate edit depth {}", depth);
        }

        let local = collect_local_edits_with_propagation(
            g,
            program,
            &ordered_paths,
            &base_path_blame,
            total_blame,
            examples,
            base_mean,
            base_var,
            config,
            depth,
            log_search,
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if clear_cap_hit_if_allowed() {
            return None;
        }
        if !local.is_empty() {
            return select_best_edit(local);
        }

        if log_search {
            println!(
                "  No passing local edits; skipping non-local scan at depth {}",
                depth
            );
        }

    }

    None
}

fn select_seed(
    g: &mut Graph,
    examples: &[(NodeId, NodeId)],
    _examples_hash: u64,
    config: &CounterfactualConfig,
    caches: &MemoCaches,
) -> NodeId {
    let leaf = g.add(Node::Leaf);
    // Always start from Leaf.
    let _ = (examples, config, caches);
    leaf
}

fn evaluate_with_traces(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: Option<&MemoCaches>,
) -> (Vec<f64>, Vec<Vec<Vec<u8>>>) {
    let len = examples.len();
    let mut samples = Vec::with_capacity(len);
    let mut active_paths = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let (sample, active) = eval_with_trace(
            g,
            program,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            caches,
            idx,
        );
        samples.push(sample);
        active_paths.push(active);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    (losses, active_paths)
}

fn eval_with_trace(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    caches: Option<&MemoCaches>,
    example_idx: usize,
) -> (LossSample, Vec<Vec<u8>>) {
    let program_resolved = g.resolve(program);
    let _ = preflight_eval_limit(
        g,
        program_resolved,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "trace_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let mut eval_to_orig: HashMap<NodeId, NodeId> = HashMap::new();
    let program_eval = clone_subtree(
        g,
        &mut eval_g,
        program_resolved,
        &mut prog_memo,
        Some(&mut eval_to_orig),
    );
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let base_nodes = eval_g.nodes.len();
    let applied = eval_g.add(Node::App {
        func: program_eval,
        args: smallvec::smallvec![input_eval],
    });

    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    let mut trace = ExecutionTrace::new();
    record_initial_canonicalization(&mut eval_g, &mut trace, program_eval, input_eval, applied);
    let mut actual = applied;
    let mut steps = 0usize;
    let mut step_limit = budget;
    let mut limit_hit = false;
    let mut node_limit_hit = false;
    let mut step_limit_hit = false;
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.exec_trace = Some(&mut trace);
    ctx.redex_cache = None;
    actual = reduce(&mut eval_g, applied, &mut ctx);
    steps = ctx.steps;
    step_limit = ctx.step_limit;
    node_limit_hit = ctx.node_limit_hit;
    step_limit_hit = ctx.step_limit_hit;
    limit_hit = node_limit_hit || step_limit_hit;
    if limit_hit {
        let prog_size = estimate_size(&eval_g, program_eval);
        let input_size = estimate_size(&eval_g, input_eval);
        let expected_size = estimate_size(&eval_g, expected_eval);
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        eprintln!(
            "CAP_HIT trace_reduce_actual example program={} nodes={} max_nodes={} steps={} step_limit={} size={} prog_size={} input_size={} expected_size={} node_limit_hit={} step_limit_hit={}",
            program_resolved.0,
            eval_nodes,
            max_nodes,
            steps,
            step_limit,
            size,
            prog_size,
            input_size,
            expected_size,
            node_limit_hit,
            step_limit_hit
        );
        if node_limit_hit {
            log_node_limit_hit(
                "trace_reduce_actual",
                eval_nodes,
                max_nodes,
                steps,
                step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if step_limit_hit {
            log_step_limit_hit(
                "trace_reduce_actual",
                eval_nodes,
                max_nodes,
                steps,
                step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
    }
    if trace.events.is_empty() {
        // No trace info (e.g., pre-eval cap); continue without blame.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let actual_size = estimate_size(&eval_g, actual);
    let expected_size = estimate_size(&eval_g, expected_nf);
    let loss = if max_ted_cells > 0
        && (actual_size as u128) * (expected_size as u128) > max_ted_cells as u128
    {
        eprintln!(
            "CAP_HIT ted_cap example program={} actual_size={} expected_size={} cells={} cap={}",
            program_resolved.0,
            actual_size,
            expected_size,
            (actual_size as u128) * (expected_size as u128),
            max_ted_cells
        );
        log_ted_cap_hit(actual_size, expected_size, max_ted_cells);
        (actual_size + expected_size) as f64
    } else {
        match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d as f64,
            Err((a, b)) => {
                eprintln!(
                    "CAP_HIT ted_cap example program={} actual_size={} expected_size={} cells={} cap={}",
                    program_resolved.0,
                    a,
                    b,
                    (a as u128) * (b as u128),
                    max_ted_cells
                );
                log_ted_cap_hit(a, b, max_ted_cells);
                (a + b) as f64
            }
        }
    };
    if example_idx == 0 {
        eprintln!(
            "TED_TRACE idx={} actual_size={} expected_size={} loss={}",
            example_idx, actual_size, expected_size, loss
        );
    }

    let mut origin_path_map: HashMap<NodeId, Vec<u8>> = HashMap::new();
    for (node, paths) in collect_paths_map(g, program_resolved) {
        if let Some(best) = paths.into_iter().min_by_key(|p| p.len()) {
            origin_path_map.insert(node, best);
        }
    }
    let mut eval_to_path: HashMap<NodeId, Vec<u8>> = HashMap::new();
    for (eval_node, orig_node) in eval_to_orig {
        if let Some(path) = origin_path_map.get(&orig_node) {
            eval_to_path.insert(eval_node, path.clone());
        }
    }
    let mut active_paths: Vec<Vec<u8>> = Vec::new();
    for event in &trace.events {
        let active_eval = leftmost_head(&eval_g, event.redex);
        let path = map_active_to_path(&eval_g, active_eval, &origin_path_map, &eval_to_path);
        eval_to_path.entry(event.result).or_insert(path.clone());
        active_paths.push(path);
    }

    let prog_size = estimate_size(&eval_g, program_eval);
    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "trace_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );

    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    (
        LossSample {
            loss,
            actual_hash,
            expected_hash,
        },
        active_paths,
    )
}

fn leftmost_head(g: &Graph, root: NodeId) -> NodeId {
    let mut curr = g.resolve(root);
    loop {
        match g.get(curr) {
            Node::App { func, .. } => {
                curr = g.resolve(*func);
            }
            _ => return curr,
        }
    }
}

fn map_active_to_path(
    eval_g: &Graph,
    active_eval: NodeId,
    origin_map: &HashMap<NodeId, Vec<u8>>,
    eval_to_path: &HashMap<NodeId, Vec<u8>>,
) -> Vec<u8> {
    if let Some(path) = origin_map
        .get(&active_eval)
        .or_else(|| eval_to_path.get(&active_eval))
    {
        return path.clone();
    }

    // Search within the active subtree for any mapped descendant.
    let mut stack = Vec::new();
    let mut seen = HashSet::new();
    stack.push(active_eval);
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        let resolved = eval_g.resolve(node);
        if !seen.insert(resolved) {
            continue;
        }
        visited += 1;
        if visited > 512 {
            break;
        }
        if let Some(path) = origin_map
            .get(&resolved)
            .or_else(|| eval_to_path.get(&resolved))
        {
            return path.clone();
        }
        match eval_g.get(resolved) {
            Node::Stem(inner) => stack.push(*inner),
            Node::Fork(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Node::App { func, args } => {
                stack.push(*func);
                for arg in args {
                    stack.push(*arg);
                }
            }
            Node::Ind(inner) => stack.push(*inner),
            _ => {}
        }
    }

    Vec::new()
}

fn record_initial_canonicalization(
    g: &mut Graph,
    trace: &mut ExecutionTrace,
    func: NodeId,
    arg: NodeId,
    applied: NodeId,
) {
    if matches!(g.get(g.resolve(applied)), Node::App { .. }) {
        return;
    }
    let redex = g.resolve(func);
    trace.record(RuleId::App, redex, vec![arg], applied);
}

fn dump_trace_empty(
    g: &Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    eval_g: &Graph,
    applied: NodeId,
    steps: usize,
    step_limit: usize,
) {
    if !debug_logs_enabled() {
        return;
    }
    let (head, argc) = app_head_info(eval_g, applied);
    eprintln!("TRACE_EMPTY: steps={} step_limit={}", steps, step_limit);
    eprintln!(
        "TRACE_EMPTY: applied_kind={} head_kind={} argc={}",
        node_kind_name(eval_g, applied),
        node_kind_name(eval_g, head),
        argc
    );
    eprintln!(
        "TRACE_EMPTY: sizes program={} input={} expected={}",
        estimate_size(g, program),
        estimate_size(g, input),
        estimate_size(g, expected)
    );
    eprintln!(
        "TRACE_EMPTY: eval_sizes applied={} head={}",
        estimate_size(eval_g, applied),
        estimate_size(eval_g, head)
    );
}

fn app_head_info(g: &Graph, root: NodeId) -> (NodeId, usize) {
    let mut curr = g.resolve(root);
    let mut argc = 0usize;
    loop {
        match g.get(curr) {
            Node::App { func, args } => {
                argc += args.len();
                curr = g.resolve(*func);
            }
            _ => return (curr, argc),
        }
    }
}

fn node_kind_name(g: &Graph, id: NodeId) -> &'static str {
    match g.get(g.resolve(id)) {
        Node::Leaf => "Leaf",
        Node::Stem(_) => "Stem",
        Node::Fork(_, _) => "Fork",
        Node::Prim(_) => "Prim",
        Node::Float(_) => "Float",
        Node::Ind(_) => "Ind",
        Node::Handle(_) => "Handle",
        Node::App { .. } => "App",
    }
}

fn path_to_string(path: &[u8]) -> String {
    if path.is_empty() {
        return "[]".to_string();
    }
    let mut out = String::new();
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        out.push_str(&p.to_string());
    }
    out
}

fn expand_tree_with_leaves(g: &mut Graph, root: NodeId) -> NodeId {
    #[derive(Clone, Copy)]
    enum Frame {
        Enter(NodeId),
        Exit(NodeId),
    }

    let resolved = g.resolve(root);
    let mut memo: HashMap<NodeId, NodeId> = HashMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    stack.push(Frame::Enter(resolved));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(id) => {
                let id = g.resolve(id);
                if memo.contains_key(&id) {
                    continue;
                }
                match g.get(id) {
                    Node::Ind(inner) => stack.push(Frame::Enter(*inner)),
                    Node::Leaf
                    | Node::Prim(_)
                    | Node::Float(_)
                    | Node::Handle(_) => {
                        stack.push(Frame::Exit(id));
                    }
                    Node::Stem(inner) => {
                        stack.push(Frame::Exit(id));
                        stack.push(Frame::Enter(*inner));
                    }
                    Node::Fork(l, r) => {
                        stack.push(Frame::Exit(id));
                        stack.push(Frame::Enter(*r));
                        stack.push(Frame::Enter(*l));
                    }
                    Node::App { func, args } => {
                        stack.push(Frame::Exit(id));
                        for arg in args.iter().rev() {
                            stack.push(Frame::Enter(*arg));
                        }
                        stack.push(Frame::Enter(*func));
                    }
                }
            }
            Frame::Exit(id) => {
                if memo.contains_key(&id) {
                    continue;
                }
                let new_id = match g.get(id).clone() {
                    Node::Leaf => {
                        let l = g.add(Node::Leaf);
                        g.add(Node::Stem(l))
                    }
                    Node::Stem(inner) => {
                        let c = *memo.get(&g.resolve(inner)).unwrap();
                        let l = g.add(Node::Leaf);
                        g.add(Node::Fork(c, l))
                    }
                    Node::Fork(l, r) => {
                        let nl = *memo.get(&g.resolve(l)).unwrap();
                        let nr = *memo.get(&g.resolve(r)).unwrap();
                        g.add(Node::Fork(nl, nr))
                    }
                    Node::Prim(p) => g.add(Node::Prim(p)),
                    Node::Float(f) => g.add(Node::Float(f)),
                    Node::Handle(h) => g.add(Node::Handle(h)),
                    Node::Ind(inner) => *memo.get(&g.resolve(inner)).unwrap(),
                    Node::App { func, args } => {
                        let nf = *memo.get(&g.resolve(func)).unwrap();
                        let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
                        for arg in args {
                            let na = *memo.get(&g.resolve(arg)).unwrap();
                            new_args.push(na);
                        }
                        g.add_raw(Node::App { func: nf, args: new_args })
                    }
                };
                memo.insert(id, new_id);
            }
        }
    }

    memo[&resolved]
}

fn evaluate_program_on_examples(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
) -> Vec<f64> {
    let len = examples.len();
    let mut samples = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let sample = eval_loss_uncached(
            g,
            program,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            None,
            "prog_example",
            idx,
        );
        samples.push(sample);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    losses
}

fn evaluate_program_on_examples_cached(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> Vec<f64> {
    let len = examples.len();
    let log_examples = false;
    let mut samples = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let ex_start = Instant::now();
        if log_examples {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            println!(
                "EXAMPLE_BEGIN phase=prog_example idx={} prog_size={} input_size={} expected_size={}",
                idx, prog_size, input_size, expected_size
            );
        }
        let loss = eval_loss_cached(
            g,
            program,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            caches,
            "prog_example",
            idx,
        );
        let ex_elapsed = ex_start.elapsed();
        if log_examples && ex_elapsed.as_millis() >= SLOW_EVAL_MS {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            log_slow_example(
                "prog_example",
                idx,
                ex_elapsed,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if log_examples {
            println!(
                "EXAMPLE_DONE phase=prog_example idx={} elapsed={:?}",
                idx, ex_elapsed
            );
        }
        samples.push(loss);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    losses
}

fn evaluate_edit_on_examples_cached(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> Vec<f64> {
    let len = examples.len();
    let log_examples = false;
    let mut samples = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let ex_start = Instant::now();
        if log_examples {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            println!(
                "EXAMPLE_BEGIN phase=edit_example idx={} prog_size={} input_size={} expected_size={}",
                idx, prog_size, input_size, expected_size
            );
        }
        let loss = eval_loss_edit_uncached(
            g,
            program,
            path,
            replacement,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            Some(caches),
            "edit_example",
            idx,
        );
        let ex_elapsed = ex_start.elapsed();
        if log_examples && ex_elapsed.as_millis() >= SLOW_EVAL_MS {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            log_slow_example(
                "edit_example",
                idx,
                ex_elapsed,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if log_examples {
            println!(
                "EXAMPLE_DONE phase=edit_example idx={} elapsed={:?}",
                idx, ex_elapsed
            );
        }
        samples.push(loss);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    losses
}

fn eval_loss(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
) -> f64 {
    eval_loss_uncached(
        g,
        program,
        input,
        expected,
        max_steps,
        max_nodes,
        max_ted_cells,
        None,
        "prog_example",
        0,
    )
    .loss
}

fn eval_loss_cached(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    caches: &MemoCaches,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let key = LossKey {
        program_hash: tree_hash(g, program),
        input_hash: tree_hash(g, input),
        expected_hash: tree_hash(g, expected),
        max_steps,
        max_nodes,
        max_ted_cells,
    };
    if let Some(v) = caches.get_loss(&key) {
        return LossSample {
            loss: v.loss,
            actual_hash: v.actual_hash,
            expected_hash: key.expected_hash,
        };
    }
    let sample = eval_loss_uncached(
        g,
        program,
        input,
        expected,
        max_steps,
        max_nodes,
        max_ted_cells,
        Some(caches),
        phase,
        example_idx,
    );
    caches.insert_loss(
        key.clone(),
        LossEntry {
            loss: sample.loss,
            actual_hash: sample.actual_hash,
        },
        loss_entry_size(&key),
    );
    sample
}

fn eval_loss_uncached(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    caches: Option<&MemoCaches>,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let _ = preflight_eval_limit(
        g,
        program,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "loss_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let prog_eval = clone_subtree(g, &mut eval_g, program, &mut prog_memo, None);
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let prog_size = estimate_size(&eval_g, prog_eval);
    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);

    let total_start = Instant::now();
    let mut reduce_actual_time = Duration::ZERO;
    let mut reduce_expected_time = Duration::ZERO;
    let mut ted_time = Duration::ZERO;

    let base_nodes = eval_g.nodes.len();
    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });
    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    if max_nodes > 0 && size > max_nodes {
        log_node_limit_hit(
            "loss_pre_eval",
            size,
            max_nodes,
            0,
            budget,
            size,
            prog_size,
            input_size,
            expected_size,
        );
    }

    let mut limit_hit = false;
    let mut actual_steps = 0usize;
    let mut actual_step_limit = budget;
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "begin",
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
        None,
    );
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.redex_cache = None;
    let reduce_start = Instant::now();
    let actual = reduce(&mut eval_g, applied, &mut ctx);
    let reduce_elapsed = reduce_start.elapsed();
    reduce_actual_time = reduce_actual_time.saturating_add(reduce_elapsed);
    actual_steps = ctx.steps;
    actual_step_limit = ctx.step_limit;
    if reduce_elapsed.as_millis() >= SLOW_EVAL_MS {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        log_slow_eval(
            "loss_reduce_actual",
            reduce_elapsed,
            eval_nodes,
            ctx.steps,
            ctx.step_limit,
            prog_size,
            input_size,
            expected_size,
        );
    }
    if ctx.node_limit_hit || ctx.step_limit_hit {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        if ctx.node_limit_hit {
            log_node_limit_hit(
                "loss_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if ctx.step_limit_hit {
            log_step_limit_hit(
                "loss_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        limit_hit = true;
    }
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "end",
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
        Some(reduce_actual_time),
    );
    log_stage_slow(
        phase,
        example_idx,
        "reduce_actual",
        reduce_actual_time,
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
    );
    if limit_hit {
        // Keep going with the partial reduction result.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let ted_actual = estimate_size(&eval_g, actual);
    let ted_expected = estimate_size(&eval_g, expected_nf);
    log_example_stage(
        phase,
        example_idx,
        "ted",
        "begin",
        prog_size,
        ted_actual,
        ted_expected,
        program,
        None,
        None,
        None,
    );
    let dist = if max_ted_cells > 0
        && (ted_actual as u128) * (ted_expected as u128) > max_ted_cells as u128
    {
        log_ted_cap_hit(ted_actual, ted_expected, max_ted_cells);
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
            Some(Duration::ZERO),
        );
        ted_actual.saturating_add(ted_expected)
    } else {
        let ted_start = Instant::now();
        let mut ted_capped = false;
        let d = match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d,
            Err((a, b)) => {
                ted_capped = true;
                log_ted_cap_hit(a, b, max_ted_cells);
                a.saturating_add(b)
            }
        };
        let ted_elapsed = ted_start.elapsed();
        ted_time = ted_time.saturating_add(ted_elapsed);
        if !ted_capped && ted_elapsed.as_millis() >= SLOW_EVAL_MS {
            let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
            let steps = actual_steps;
            let step_limit = actual_step_limit;
            log_slow_eval(
                "loss_ted",
                ted_elapsed,
                eval_nodes,
                steps,
                step_limit,
                prog_size,
                input_size,
                expected_size,
            );
        }
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
            Some(ted_elapsed),
        );
        log_stage_slow(
            phase,
            example_idx,
            "ted",
            ted_elapsed,
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
        );
        d
    };
    let total_elapsed = total_start.elapsed();
    with_timing_acc(|acc| {
        let mut acc = acc.lock().unwrap();
        acc.total = acc.total.saturating_add(total_elapsed);
        acc.reduce_actual = acc.reduce_actual.saturating_add(reduce_actual_time);
        acc.reduce_expected = acc.reduce_expected.saturating_add(reduce_expected_time);
        acc.ted = acc.ted.saturating_add(ted_time);
        acc.evals = acc.evals.saturating_add(1);
    });
    let steps = actual_steps;
    let step_limit = actual_step_limit;
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "loss_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );
    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    LossSample {
        loss: dist as f64,
        actual_hash,
        expected_hash,
    }
}

fn eval_loss_edit_uncached(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    caches: Option<&MemoCaches>,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let _ = preflight_eval_limit_with_replacement(
        g,
        program,
        path,
        replacement,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "edit_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let prog_eval_src = clone_subtree(g, &mut eval_g, program, &mut prog_memo, None);
    let mut repl_memo = HashMap::new();
    let repl_eval = clone_subtree(g, &mut eval_g, replacement, &mut repl_memo, None);
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);

    let total_start = Instant::now();
    let mut reduce_actual_time = Duration::ZERO;
    let mut reduce_expected_time = Duration::ZERO;
    let mut ted_time = Duration::ZERO;

    let base_nodes = eval_g.nodes.len();
    let prog_eval = replace_at_path(&mut eval_g, prog_eval_src, path, repl_eval);
    let prog_size = estimate_size(&eval_g, prog_eval);
    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });

    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    if max_nodes > 0 && size > max_nodes {
        log_node_limit_hit(
            "edit_pre_eval",
            size,
            max_nodes,
            0,
            budget,
            size,
            prog_size,
            input_size,
            expected_size,
        );
    }

    let mut limit_hit = false;
    let mut actual_steps = 0usize;
    let mut actual_step_limit = budget;
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "begin",
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
        None,
    );
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.redex_cache = None;
    let reduce_start = Instant::now();
    let actual = reduce(&mut eval_g, applied, &mut ctx);
    let reduce_elapsed = reduce_start.elapsed();
    reduce_actual_time = reduce_actual_time.saturating_add(reduce_elapsed);
    actual_steps = ctx.steps;
    actual_step_limit = ctx.step_limit;
    if reduce_elapsed.as_millis() >= SLOW_EVAL_MS {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        log_slow_eval(
            "edit_reduce_actual",
            reduce_elapsed,
            eval_nodes,
            ctx.steps,
            ctx.step_limit,
            prog_size,
            input_size,
            expected_size,
        );
    }
    if ctx.node_limit_hit || ctx.step_limit_hit {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        if ctx.node_limit_hit {
            log_node_limit_hit(
                "edit_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if ctx.step_limit_hit {
            log_step_limit_hit(
                "edit_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        limit_hit = true;
    }
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "end",
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
        Some(reduce_actual_time),
    );
    log_stage_slow(
        phase,
        example_idx,
        "reduce_actual",
        reduce_actual_time,
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
    );
    if limit_hit {
        // Keep going with the partial reduction result.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let ted_actual = estimate_size(&eval_g, actual);
    let ted_expected = estimate_size(&eval_g, expected_nf);
    log_example_stage(
        phase,
        example_idx,
        "ted",
        "begin",
        prog_size,
        ted_actual,
        ted_expected,
        program,
        Some(path),
        None,
        None,
    );
    let dist = if max_ted_cells > 0
        && (ted_actual as u128) * (ted_expected as u128) > max_ted_cells as u128
    {
        log_ted_cap_hit(ted_actual, ted_expected, max_ted_cells);
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
            Some(Duration::ZERO),
        );
        ted_actual.saturating_add(ted_expected)
    } else {
        let ted_start = Instant::now();
        let mut ted_capped = false;
        let d = match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d,
            Err((a, b)) => {
                ted_capped = true;
                log_ted_cap_hit(a, b, max_ted_cells);
                a.saturating_add(b)
            }
        };
        let ted_elapsed = ted_start.elapsed();
        ted_time = ted_time.saturating_add(ted_elapsed);
        if !ted_capped && ted_elapsed.as_millis() >= SLOW_EVAL_MS {
            let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
            let steps = actual_steps;
            let step_limit = actual_step_limit;
            log_slow_eval(
                "edit_ted",
                ted_elapsed,
                eval_nodes,
                steps,
                step_limit,
                prog_size,
                input_size,
                expected_size,
            );
        }
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
            Some(ted_elapsed),
        );
        log_stage_slow(
            phase,
            example_idx,
            "ted",
            ted_elapsed,
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
        );
        d
    };
    let total_elapsed = total_start.elapsed();
    with_timing_acc(|acc| {
        let mut acc = acc.lock().unwrap();
        acc.total = acc.total.saturating_add(total_elapsed);
        acc.reduce_actual = acc.reduce_actual.saturating_add(reduce_actual_time);
        acc.reduce_expected = acc.reduce_expected.saturating_add(reduce_expected_time);
        acc.ted = acc.ted.saturating_add(ted_time);
        acc.evals = acc.evals.saturating_add(1);
    });
    let steps = actual_steps;
    let step_limit = actual_step_limit;
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "edit_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );

    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    LossSample {
        loss: dist as f64,
        actual_hash,
        expected_hash,
    }
}


fn candidate_stats(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    examples_hash: u64,
    caches: &MemoCaches,
    _log_timing: bool,
) -> CandidateStats {
    let log_timing = false;
    let replacement_hash = tree_hash(g, replacement);
    let key = CandidateKey {
        program_hash: tree_hash(g, program),
        path: path.to_vec(),
        replacement_hash,
        examples_hash,
        max_steps,
        max_nodes,
        max_ted_cells,
    };
    if let Some(stats) = caches.get_candidate(&key) {
        if log_timing {
            println!(
                "    Candidate cached path={} repl_nodes={} repl_hash={}",
                path_to_string(path),
                estimate_size(g, replacement),
                replacement_hash
            );
        }
        return stats;
    }
    if log_timing {
        println!(
            "    Candidate start path={} repl_nodes={} repl_hash={} evals={}",
            path_to_string(path),
            estimate_size(g, replacement),
            replacement_hash,
            examples.len()
        );
    }
    let timing = if log_timing {
        Some(Arc::new(Mutex::new(TimingAcc::default())))
    } else {
        None
    };
    if let Some(acc) = timing.as_ref() {
        set_timing_acc(Some(acc.clone()));
    }
    let wall_start = Instant::now();
    let losses = evaluate_edit_on_examples_cached(
        g,
        program,
        path,
        replacement,
        examples,
        max_steps,
        max_nodes,
        max_ted_cells,
        output_collision_penalty,
        caches,
    );
    let wall_elapsed = wall_start.elapsed();
    if timing.is_some() {
        set_timing_acc(None);
        let acc = timing.unwrap();
        let acc = acc.lock().unwrap();
        println!(
            "    Candidate timing path={} repl_nodes={} repl_hash={} evals={} wall={:?} cpu_total={:?} reduce_actual={:?} reduce_expected={:?} ted={:?} cache_hits(actual={}, expected={})",
            path_to_string(path),
            estimate_size(g, replacement),
            replacement_hash,
            acc.evals,
            wall_elapsed,
            acc.total,
            acc.reduce_actual,
            acc.reduce_expected,
            acc.ted,
            acc.reduce_actual_hits,
            acc.reduce_expected_hits
        );
    }
    let mean_loss = mean(&losses);
    let var_loss = variance(&losses);
    let stats = CandidateStats {
        mean: mean_loss,
        var: var_loss,
    };
    caches.insert_candidate(key.clone(), stats, candidate_entry_size(&key));
    stats
}

fn program_stats_cached(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> CandidateStats {
    let losses = evaluate_program_on_examples_cached(
        g,
        program,
        examples,
        max_steps,
        max_nodes,
        max_ted_cells,
        output_collision_penalty,
        caches,
    );
    CandidateStats {
        mean: mean(&losses),
        var: variance(&losses),
    }
}

fn best_edit_for_path_with_depth(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    depth: usize,
    max_edits: usize,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> (Option<EditPlan>, usize) {
    if let Some(banned_paths) = banned_paths {
        if banned_paths.contains(path) {
            return (None, 0);
        }
    }
    let target = match node_at_path(g, program, path) {
        Some(t) => t,
        None => return (None, 0),
    };
    bump_iter_paths_scanned(1);
    set_debug_context(Some(program), Some(path), Some(depth));
    if config.max_eval_nodes > 0 && estimate_size(g, target) > config.max_eval_nodes {
        log_repr_cap_hit(
            "best_edit_target",
            estimate_size(g, target),
            config.max_eval_nodes,
        );
        return (None, 0);
    }
    let mut best: Option<EditPlan> = None;
    let mut evaluated = 0usize;
    let cap = config.max_candidate_evals.max(1);
    let max_edits = max_edits.max(1);
    let mut cap_seen = false;
    let replacements = generate_edits_with_depth(
        g,
        program,
        path,
        depth,
        max_edits,
        config.max_replacement_nodes,
    );
    let generated = replacements.len();
    for rep in replacements {
        if evaluated >= cap {
            break;
        }
        if let Some(banned) = banned {
            let key = EditKey {
                path: path.to_vec(),
                replacement_hash: rep.hash,
            };
            if banned.contains(&key) {
                continue;
            }
        }
        evaluated += 1;
        bump_iter_candidate_evals(1);
        reset_cap_hit();
        let stats = candidate_stats(
            g,
            program,
            path,
            rep.node,
            examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            examples_hash,
            caches,
            config.debug,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                cap_seen = true;
                break;
            }
            continue;
        }
        let improvement =
            (base_mean - stats.mean) + config.variance_weight * (base_var - stats.var);
        let plan = EditPlan {
            path: path.to_vec(),
            replacement: rep.node,
            replacement_hash: rep.hash,
            improvement,
            depth,
        };
        match &best {
            Some(existing) if existing.improvement >= plan.improvement => {}
            _ => best = Some(plan),
        }
    }
    if cap_seen {
        return (None, generated);
    }
    if evaluated >= cap && config.debug {
        println!(
            "  Candidate cap hit at path {} depth {}: {} -> {}",
            path_to_string(path),
            depth,
            evaluated,
            cap
        );
    }
    (best, generated)
}

#[derive(Clone, Copy)]
enum EditFilter {
    Threshold,
    Positive,
}

fn collect_local_edits_with_propagation(
    g: &mut Graph,
    program: NodeId,
    ordered_paths: &[Vec<u8>],
    base_path_blame: &HashMap<Vec<u8>, f64>,
    total_blame: f64,
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    depth: usize,
    log_search: bool,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> Vec<EditPlan> {
    let mut path_blame = base_path_blame.clone();
    let mut edits: Vec<EditPlan> = Vec::new();
    let mut best_positive: Option<EditPlan> = None;
    let mut edited_paths: HashSet<Vec<u8>> = HashSet::new();
    let mut remaining = config.max_candidate_evals.max(1);

    for path in ordered_paths {
        if remaining == 0 {
            if log_search {
                println!("  Local edit budget exhausted; stopping local scan.");
            }
            break;
        }
        if let Some(banned_paths) = banned_paths {
            if banned_paths.contains(path) {
                continue;
            }
        }
        let blame = path_blame.get(path).copied().unwrap_or(0.0);
        if blame <= 0.0 {
            continue;
        }
        if has_conflict(&edited_paths, path) {
            continue;
        }
        let threshold = threshold_for_blame(config.edit_threshold_base, blame, total_blame);
        let path_budget = remaining.min(config.max_edits_per_path.max(1));
        let (best, generated) = best_edit_for_path_with_depth(
            g,
            program,
            path,
            examples,
            base_mean,
            base_var,
            config,
            depth,
            path_budget,
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return Vec::new();
            }
            continue;
        }
        remaining = remaining.saturating_sub(generated);
        if log_search {
            let (best_improve, best_rep) = match &best {
                Some(plan) => (plan.improvement, unparse_limited(g, plan.replacement)),
                None => (0.0, "<none>".to_string()),
            };
            println!(
                "  Path {} blame={:.4} threshold={:.6} best_improve={:.6} best={}",
                path_to_string(path),
                blame,
                threshold,
                best_improve,
                best_rep
            );
        }
        if let Some(plan) = best.as_ref() {
            if plan.improvement > 0.0
                && best_positive
                    .as_ref()
                    .map(|p| plan.improvement > p.improvement)
                    .unwrap_or(true)
            {
                best_positive = Some(plan.clone());
            }
        }
        if let Some(plan) = best {
            if plan.improvement > threshold {
                edited_paths.insert(path.clone());
                edits.push(plan);
                continue;
            }
        }

        if path.is_empty() {
            if let Some(plan) = propagate_downward(
                g,
                program,
                path.clone(),
                blame,
                total_blame,
                examples,
                base_mean,
                base_var,
                config,
                &mut edited_paths,
                examples_hash,
                caches,
                banned,
                banned_paths,
            ) {
                edits.push(plan);
            }
            if cap_hit() {
                if clear_cap_hit_if_allowed() {
                    return Vec::new();
                }
                continue;
            }
        } else {
            let mut parent = path.clone();
            parent.pop();
            *path_blame.entry(parent).or_insert(0.0) += blame;
        }
    }

    if edits.is_empty() {
        if let Some(plan) = best_positive {
            edits.push(plan);
        }
    }
    edits
}

fn collect_edits_at_depth(
    g: &mut Graph,
    program: NodeId,
    paths: &[Vec<u8>],
    base_path_blame: &HashMap<Vec<u8>, f64>,
    total_blame: f64,
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    depth: usize,
    log_search: bool,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
    filter: EditFilter,
) -> Vec<EditPlan> {
    let cap = config.max_candidate_evals.max(1);
    let mut hit_cap = false;
    let mut cap_seen = false;
    let mut eval_count = 0usize;
    let mut passing: Vec<EditPlan> = Vec::new();
    'outer: for path in paths {
        set_debug_context(Some(program), Some(path), Some(depth));
        if let Some(banned_paths) = banned_paths {
            if banned_paths.contains(path) {
                continue;
            }
        }
        let target = match node_at_path(g, program, path) {
            Some(t) => t,
            None => continue,
        };
        if config.max_eval_nodes > 0 && estimate_size(g, target) > config.max_eval_nodes {
            log_repr_cap_hit(
                "exhaustive_target",
                estimate_size(g, target),
                config.max_eval_nodes,
            );
            cap_seen = true;
            break 'outer;
        }
        bump_iter_paths_scanned(1);
        let replacements = generate_edits_with_depth(
            g,
            program,
            path,
            depth,
            config.max_edits_per_path.max(1),
            config.max_replacement_nodes,
        );
        for rep in replacements {
            eval_count += 1;
            if eval_count > cap {
                hit_cap = true;
                break 'outer;
            }
            if let Some(banned) = banned {
                let key = EditKey {
                    path: path.clone(),
                    replacement_hash: rep.hash,
                };
                if banned.contains(&key) {
                    continue;
                }
            }
            reset_cap_hit();
            bump_iter_candidate_evals(1);
            let stats = candidate_stats(
                g,
                program,
                path,
                rep.node,
                examples,
                config.max_eval_steps,
                config.max_eval_nodes,
                config.max_ted_cells,
                config.output_collision_penalty,
                examples_hash,
                caches,
                config.debug,
            );
            if cap_hit() {
                if clear_cap_hit_if_allowed() {
                    cap_seen = true;
                    break 'outer;
                }
                continue;
            }
            let improvement =
                (base_mean - stats.mean) + config.variance_weight * (base_var - stats.var);
            let pass = match filter {
                EditFilter::Positive => improvement > 0.0,
                EditFilter::Threshold => {
                    let blame = base_path_blame.get(path).copied().unwrap_or(0.0);
                    let threshold = threshold_for_blame(
                        config.edit_threshold_base,
                        blame,
                        total_blame,
                    );
                    improvement > threshold
                }
            };
            if pass {
                passing.push(EditPlan {
                    path: path.clone(),
                    replacement: rep.node,
                    replacement_hash: rep.hash,
                    improvement,
                    depth,
                });
            }
        }
    }

    if cap_seen {
        return Vec::new();
    }

    if hit_cap && log_search {
        println!("  Max-depth candidate cap hit: {} -> {}", eval_count, cap);
    }

    passing
}

fn propagate_downward(
    g: &mut Graph,
    program: NodeId,
    path: Vec<u8>,
    blame: f64,
    total_blame: f64,
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    edited_paths: &mut HashSet<Vec<u8>>,
    examples_hash: u64,
    caches: &MemoCaches,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
) -> Option<EditPlan> {
    #[derive(Clone)]
    struct WorkItem {
        path: Vec<u8>,
        blame: f64,
    }

    let mut stack: Vec<WorkItem> = Vec::new();
    let mut seen: HashSet<(NodeId, Vec<u8>)> = HashSet::new();
    stack.push(WorkItem { path, blame });

    while let Some(item) = stack.pop() {
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return None;
            }
            continue;
        }
        if let Some(banned_paths) = banned_paths {
            if banned_paths.contains(&item.path) {
                continue;
            }
        }
        if item.blame <= 0.0 || has_conflict(edited_paths, &item.path) {
            continue;
        }
        let threshold = threshold_for_blame(config.edit_threshold_base, item.blame, total_blame);
        let (best, _) = best_edit_for_path_with_depth(
            g,
            program,
            &item.path,
            examples,
            base_mean,
            base_var,
            config,
            1,
            config.max_edits_per_path.max(1),
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return None;
            }
            continue;
        }
        if let Some(plan) = best {
            if plan.improvement > threshold {
                edited_paths.insert(item.path.clone());
                return Some(plan);
            }
        }

        let Some(node) = node_at_path(g, program, &item.path) else { continue; };
        if !seen.insert((node, item.path.clone())) {
            continue;
        }
        match g.get(node).clone() {
            Node::Stem(_) => {
                let mut child = item.path.clone();
                child.push(0);
                stack.push(WorkItem { path: child, blame: item.blame * 0.5 });
            }
            Node::Fork(_, _) => {
                let mut right = item.path.clone();
                right.push(1);
                let mut left = item.path.clone();
                left.push(0);
                stack.push(WorkItem { path: right, blame: item.blame * 0.5 });
                stack.push(WorkItem { path: left, blame: item.blame * 0.5 });
            }
            Node::App { func: _, args } => {
                let mut func_path = item.path.clone();
                func_path.push(0);
                for (idx, _arg) in args.iter().enumerate().take(4).rev() {
                    let mut arg_path = item.path.clone();
                    arg_path.push((idx as u8) + 1);
                    if node_at_path(g, program, &arg_path).is_some() {
                        stack.push(WorkItem { path: arg_path, blame: item.blame * 0.5 });
                    }
                }
                if node_at_path(g, program, &func_path).is_some() {
                    stack.push(WorkItem { path: func_path, blame: item.blame * 0.5 });
                }
            }
            Node::Ind(inner) => {
                if inner != node {
                    stack.push(WorkItem { path: item.path, blame: item.blame * 0.5 });
                }
            }
            _ => {}
        }
    }

    None
}

fn generate_edits(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    max_nodes: usize,
) -> Vec<Replacement> {
    let target = match node_at_path(g, program, path) {
        Some(t) => t,
        None => return Vec::new(),
    };
    generate_replacements_for_node(g, target, max_nodes)
}

fn generate_edits_with_depth(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    depth: usize,
    max_edits: usize,
    max_nodes: usize,
) -> Vec<Replacement> {
    let max_edits = max_edits.max(1);
    if depth <= 1 {
        let mut edits = generate_edits(g, program, path, max_nodes);
        if edits.len() > max_edits {
            edits.truncate(max_edits);
        }
        return edits;
    }
    let mut seen: HashSet<u64> = HashSet::new();
    let mut all: Vec<Replacement> = Vec::new();
    let mut frontier: Vec<Replacement> = Vec::new();
    let base = generate_edits(g, program, path, max_nodes);
    for rep in base {
        if seen.insert(rep.hash) {
            frontier.push(rep.clone());
            all.push(rep);
            if all.len() >= max_edits {
                return all;
            }
        }
    }

    for _ in 2..=depth {
        let mut next_frontier: Vec<Replacement> = Vec::new();
        'frontier: for node in &frontier {
            for rep in generate_replacements_for_node(g, node.node, max_nodes) {
                if seen.insert(rep.hash) {
                    next_frontier.push(rep.clone());
                    all.push(rep);
                    if all.len() >= max_edits {
                        return all;
                    }
                }
            }

            let mut child_paths: HashSet<Vec<u8>> = HashSet::new();
            let path_map = collect_paths_map(g, node.node);
            for paths in path_map.values() {
                for p in paths {
                    if p.is_empty() {
                        continue;
                    }
                    child_paths.insert(p.clone());
                }
            }

            for path in child_paths {
                let edits = generate_edits(g, node.node, &path, max_nodes);
                for rep in edits {
                    let new_root = replace_at_path(g, node.node, &path, rep.node);
                    if max_nodes > 0 && estimate_size(g, new_root) > max_nodes {
                        continue;
                    }
                    let new_rep = make_replacement(g, new_root);
                    if seen.insert(new_rep.hash) {
                        next_frontier.push(new_rep.clone());
                        all.push(new_rep);
                        if all.len() >= max_edits {
                            return all;
                        }
                    }
                }
                if all.len() >= max_edits {
                    break 'frontier;
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    all
}

fn generate_replacements_for_node(
    g: &mut Graph,
    target: NodeId,
    max_nodes: usize,
) -> Vec<Replacement> {
    let mut edits: Vec<Replacement> = Vec::new();
    let leaf = g.add(Node::Leaf);
    let fork_leaf_leaf = g.add(Node::Fork(leaf, leaf));

    fn push_if_small(
        g: &Graph,
        node: NodeId,
        max_nodes: usize,
        out: &mut Vec<Replacement>,
    ) {
        if max_nodes == 0 || estimate_size(g, node) <= max_nodes {
            out.push(make_replacement(g, node));
        }
    }

    let target = g.resolve(target);
    match g.get(target) {
        Node::Leaf => {
            let stem = g.add(Node::Stem(leaf));
            push_if_small(g, stem, max_nodes, &mut edits);
            push_if_small(g, fork_leaf_leaf, max_nodes, &mut edits);
        }
        Node::Stem(inner) => {
            let child = g.resolve(*inner);
            let fork_cl = g.add(Node::Fork(child, leaf));
            let fork_lc = g.add(Node::Fork(leaf, child));
            let fork_cc = g.add(Node::Fork(child, child));
            push_if_small(g, fork_leaf_leaf, max_nodes, &mut edits);
            push_if_small(g, fork_cl, max_nodes, &mut edits);
            push_if_small(g, fork_lc, max_nodes, &mut edits);
            push_if_small(g, fork_cc, max_nodes, &mut edits);
        }
        Node::Fork(left, right) => {
            let l = g.resolve(*left);
            let r = g.resolve(*right);
            let stem_l = g.add(Node::Stem(l));
            let stem_r = g.add(Node::Stem(r));
            let fork_rl = g.add(Node::Fork(r, l));
            let fork_ll = g.add(Node::Fork(l, l));
            let fork_rr = g.add(Node::Fork(r, r));
            push_if_small(g, leaf, max_nodes, &mut edits);
            push_if_small(g, stem_l, max_nodes, &mut edits);
            push_if_small(g, stem_r, max_nodes, &mut edits);
            push_if_small(g, fork_rl, max_nodes, &mut edits);
            push_if_small(g, fork_ll, max_nodes, &mut edits);
            push_if_small(g, fork_rr, max_nodes, &mut edits);
        }
        _ => {}
    }

    edits
}

fn has_conflict(existing: &HashSet<Vec<u8>>, candidate: &[u8]) -> bool {
    for path in existing {
        if is_prefix(path, candidate) || is_prefix(candidate, path) {
            return true;
        }
    }
    false
}

fn is_prefix(a: &[u8], b: &[u8]) -> bool {
    if a.len() > b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let mu = mean(xs);
    xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / xs.len() as f64
}

fn resolve_safe(g: &Graph, mut id: NodeId) -> NodeId {
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(id) {
            return id;
        }
        match g.get(id) {
            Node::Ind(inner) => id = *inner,
            _ => return id,
        }
    }
}

fn estimate_size(g: &Graph, root: NodeId) -> usize {
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

fn collect_paths_map(g: &Graph, root: NodeId) -> HashMap<NodeId, Vec<Vec<u8>>> {
    let mut map: HashMap<NodeId, Vec<Vec<u8>>> = HashMap::new();
    let mut stack: Vec<(NodeId, Vec<u8>)> = Vec::new();
    stack.push((root, Vec::new()));

    while let Some((id, path)) = stack.pop() {
        let resolved = resolve_safe(g, id);
        map.entry(resolved).or_default().push(path.clone());
        match g.get(resolved) {
            Node::Stem(inner) => {
                let mut p = path.clone();
                p.push(0);
                stack.push((*inner, p));
            }
            Node::Fork(l, r) => {
                let mut p0 = path.clone();
                p0.push(0);
                stack.push((*l, p0));
                let mut p1 = path.clone();
                p1.push(1);
                stack.push((*r, p1));
            }
            Node::App { func, args } => {
                let mut pf = path.clone();
                pf.push(0);
                stack.push((*func, pf));
                for (idx, arg) in args.iter().enumerate() {
                    if idx >= 250 {
                        break;
                    }
                    let mut pa = path.clone();
                    pa.push((idx as u8) + 1);
                    stack.push((*arg, pa));
                }
            }
            Node::Ind(inner) => {
                stack.push((*inner, path));
            }
            _ => {}
        }
    }

    map
}

fn node_at_path(g: &Graph, root: NodeId, path: &[u8]) -> Option<NodeId> {
    let mut curr = g.resolve(root);
    for &dir in path {
        match g.get(curr) {
            Node::Stem(inner) => {
                if dir != 0 {
                    return None;
                }
                curr = g.resolve(*inner);
            }
            Node::Fork(l, r) => {
                curr = if dir == 0 { g.resolve(*l) } else { g.resolve(*r) };
            }
            Node::App { func, args } => {
                if dir == 0 {
                    curr = g.resolve(*func);
                } else {
                    let idx = (dir - 1) as usize;
                    if idx >= args.len() {
                        return None;
                    }
                    curr = g.resolve(args[idx]);
                }
            }
            Node::Ind(inner) => {
                curr = g.resolve(*inner);
            }
            _ => return None,
        }
    }
    Some(curr)
}

fn replace_at_path(g: &mut Graph, root: NodeId, path: &[u8], replacement: NodeId) -> NodeId {
    if path.is_empty() {
        return replacement;
    }

    #[derive(Clone)]
    enum Trail {
        Stem,
        ForkLeft { right: NodeId },
        ForkRight { left: NodeId },
        AppFunc { args: SmallVec<[NodeId; 2]> },
        AppArg { func: NodeId, args: SmallVec<[NodeId; 2]>, idx: usize },
    }

    let mut curr = g.resolve(root);
    let mut idx = 0usize;
    let mut trail: Vec<Trail> = Vec::new();

    loop {
        if idx >= path.len() {
            break;
        }
        match g.get(curr).clone() {
            Node::Stem(inner) => {
                if path[idx] != 0 {
                    return curr;
                }
                trail.push(Trail::Stem);
                curr = g.resolve(inner);
                idx += 1;
            }
            Node::Fork(l, r) => {
                if path[idx] == 0 {
                    trail.push(Trail::ForkLeft { right: r });
                    curr = g.resolve(l);
                    idx += 1;
                } else {
                    trail.push(Trail::ForkRight { left: l });
                    curr = g.resolve(r);
                    idx += 1;
                }
            }
            Node::App { func, args } => {
                if path[idx] == 0 {
                    trail.push(Trail::AppFunc { args });
                    curr = g.resolve(func);
                    idx += 1;
                } else {
                    let arg_idx = (path[idx] - 1) as usize;
                    if arg_idx >= args.len() {
                        return curr;
                    }
                    let arg_node = args[arg_idx];
                    trail.push(Trail::AppArg { func, args, idx: arg_idx });
                    curr = g.resolve(arg_node);
                    idx += 1;
                }
            }
            Node::Ind(inner) => {
                curr = g.resolve(inner);
            }
            _ => return curr,
        }
    }

    let mut built = replacement;
    while let Some(frame) = trail.pop() {
        built = match frame {
            Trail::Stem => g.add(Node::Stem(built)),
            Trail::ForkLeft { right } => g.add(Node::Fork(built, right)),
            Trail::ForkRight { left } => g.add(Node::Fork(left, built)),
            Trail::AppFunc { args } => g.add(Node::App { func: built, args }),
            Trail::AppArg { func, mut args, idx } => {
                if idx < args.len() {
                    args[idx] = built;
                }
                g.add(Node::App { func, args })
            }
        };
    }

    built
}
