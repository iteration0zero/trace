use crate::arena::NodeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const EVAL_STEP_CAP: usize = 50_000_000;
pub const SLOW_EVAL_MS: u128 = 100;
pub const STAGE_WARN_MS_DEFAULT: u128 = 500;
pub const STAGE_WARN_REPEAT_MS_DEFAULT: u128 = 1000;

pub static DEBUG_OOM: AtomicBool = AtomicBool::new(false);
pub static TIMING_ACC: Mutex<Option<Arc<Mutex<TimingAcc>>>> = Mutex::new(None);
pub static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
pub static EVAL_BUDGET_BYTES: AtomicUsize = AtomicUsize::new(512 * 1024 * 1024);
pub static CAP_HIT: AtomicBool = AtomicBool::new(false);
pub static ITER_PATHS_SCANNED: AtomicUsize = AtomicUsize::new(0);
pub static ITER_CANDIDATE_EVALS: AtomicUsize = AtomicUsize::new(0);

static STAGE_MAP: OnceLock<Mutex<HashMap<StageKey, StageInfo>>> = OnceLock::new();
static STAGE_WATCHDOG: OnceLock<()> = OnceLock::new();

pub fn path_to_string(path: &[u8]) -> String {
    if path.is_empty() {
        return "root".to_string();
    }
    let mut s = String::new();
    for &b in path {
        if b == 0 {
            s.push('L');
        } else {
            s.push('R');
        }
    }
    s
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct StageKey {
    pub phase: &'static str,
    pub idx: usize,
    pub stage: &'static str,
    pub program: u32,
    pub path: Option<Vec<u8>>,
    pub depth: Option<usize>,
}

#[derive(Clone)]
pub struct StageInfo {
    pub key: StageKey,
    pub start: Instant,
    pub last_log: Option<Instant>,
    pub prog_size: usize,
    pub input_size: usize,
    pub expected_size: usize,
}

pub fn stage_map() -> &'static Mutex<HashMap<StageKey, StageInfo>> {
    STAGE_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn debug_logs_enabled() -> bool {
    DEBUG_OOM.load(Ordering::Relaxed)
}

pub fn example_stage_enabled() -> bool {
    debug_logs_enabled()
}

pub fn stage_warn_ms() -> u128 {
    static WARN_MS: OnceLock<u128> = OnceLock::new();
    *WARN_MS.get_or_init(|| {
        std::env::var("TRACE_STAGE_WARN_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STAGE_WARN_MS_DEFAULT)
    })
}

pub fn stage_warn_repeat_ms() -> u128 {
    static REPEAT_MS: OnceLock<u128> = OnceLock::new();
    *REPEAT_MS.get_or_init(|| {
        std::env::var("TRACE_STAGE_WARN_REPEAT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STAGE_WARN_REPEAT_MS_DEFAULT)
    })
}

pub fn start_stage_watchdog() {
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
pub struct TimingAcc {
    pub total: Duration,
    pub reduce_actual: Duration,
    pub reduce_expected: Duration,
    pub ted: Duration,
    pub evals: usize,
    pub reduce_actual_hits: usize,
    pub reduce_expected_hits: usize,
}

#[derive(Clone, Default)]
pub struct DebugContext {
    pub program: Option<NodeId>,
    pub path: Option<Vec<u8>>,
    pub depth: Option<usize>,
}

#[derive(Clone)]
pub struct PeakInfo {
    pub nodes: usize,
    pub phase: &'static str,
    pub steps: usize,
    pub step_limit: usize,
    pub applied_size: usize,
    pub prog_size: usize,
    pub input_size: usize,
    pub expected_size: usize,
    pub program: Option<NodeId>,
    pub path: Option<Vec<u8>>,
    pub depth: Option<usize>,
}

static DEBUG_CONTEXT: Mutex<DebugContext> = Mutex::new(DebugContext {
    program: None,
    path: None,
    depth: None,
});
static PEAK_INFO: Mutex<Option<PeakInfo>> = Mutex::new(None);

pub fn set_debug_oom(enabled: bool) {
    DEBUG_OOM.store(enabled, Ordering::Relaxed);
}

pub fn set_timing_acc(acc: Option<Arc<Mutex<TimingAcc>>>) {
    if acc.is_some() {
        TIMING_ENABLED.store(true, Ordering::Relaxed);
    } else {
        TIMING_ENABLED.store(false, Ordering::Relaxed);
    }
    let mut slot = TIMING_ACC.lock().unwrap();
    *slot = acc;
}

pub fn with_timing_acc<F: FnOnce(&Arc<Mutex<TimingAcc>>)>(f: F) {
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

pub fn set_eval_budget_bytes(bytes: usize) {
    EVAL_BUDGET_BYTES.store(bytes.max(1), Ordering::Relaxed);
}

pub fn reset_cap_hit() {
    CAP_HIT.store(false, Ordering::Relaxed);
}

pub fn mark_cap_hit() {
    CAP_HIT.store(true, Ordering::Relaxed);
}

pub fn cap_hit() -> bool {
    CAP_HIT.load(Ordering::Relaxed)
}

pub fn should_abort_on_penalty() -> bool {
    DEBUG_OOM.load(Ordering::Relaxed)
}

pub fn clear_cap_hit_if_allowed() -> bool {
    if !cap_hit() {
        return false;
    }
    if should_abort_on_penalty() {
        return true;
    }
    reset_cap_hit();
    false
}

pub fn reset_iter_counters() {
    ITER_PATHS_SCANNED.store(0, Ordering::Relaxed);
    ITER_CANDIDATE_EVALS.store(0, Ordering::Relaxed);
}

pub fn bump_iter_paths_scanned(n: usize) {
    if n > 0 {
        ITER_PATHS_SCANNED.fetch_add(n, Ordering::Relaxed);
    }
}

pub fn bump_iter_candidate_evals(n: usize) {
    if n > 0 {
        ITER_CANDIDATE_EVALS.fetch_add(n, Ordering::Relaxed);
    }
}

pub fn snapshot_iter_counters() -> (usize, usize) {
    (
        ITER_PATHS_SCANNED.load(Ordering::Relaxed),
        ITER_CANDIDATE_EVALS.load(Ordering::Relaxed),
    )
}

pub fn set_debug_context(program: Option<NodeId>, path: Option<&[u8]>, depth: Option<usize>) {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let mut ctx = DEBUG_CONTEXT.lock().unwrap();
    ctx.program = program;
    ctx.path = path.map(|p| p.to_vec());
    ctx.depth = depth;
}

pub fn get_debug_context() -> DebugContext {
    DEBUG_CONTEXT.lock().unwrap().clone()
}

pub fn reset_peak() {
    if !DEBUG_OOM.load(Ordering::Relaxed) {
        return;
    }
    let mut peak = PEAK_INFO.lock().unwrap();
    *peak = None;
}

pub fn update_peak(
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

pub fn report_peak(iter: usize) {
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

pub fn log_slow_eval(
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

pub fn log_slow_example(
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

pub fn log_node_limit_hit(
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

pub fn log_step_limit_hit(
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

pub fn log_ted_cap_hit(actual_size: usize, expected_size: usize, cap: usize) {
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

pub fn log_example_stage(
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

pub fn log_stage_slow(
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

pub fn log_repr_cap_hit(phase: &'static str, size: usize, max_nodes: usize) {
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

pub fn log_path_cap_hit(phase: &'static str, count: usize, max_paths: usize) {
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

pub fn scaled_eval_budget(max_steps: usize, size: usize) -> usize {
    let mut budget = max_steps.saturating_mul(size.max(1));
    if budget > EVAL_STEP_CAP {
        budget = EVAL_STEP_CAP;
    }
    budget
}
