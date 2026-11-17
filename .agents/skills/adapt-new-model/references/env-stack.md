# pypto Env Stack: Install, Refresh, Verify

The serving repo only pins kernel sources (`pypto-lib` submodule). The host
environment additionally provides a compiled runtime stack, and device
behavior depends on the two agreeing. Before reproducing anything on device —
especially before concluding a fault is caused by new model code — confirm the
stack is current and internally consistent.

## What the stack is

| Component | What it is | Placeholder |
| --- | --- | --- |
| pypto | Runtime/toolchain repo; its `runtime` submodule is simpler | `<pypto-checkout>` |
| simpler | Compiled runtime (non-editable install); pinned as pypto's `runtime` submodule | `<pypto-checkout>/runtime` |
| pto-isa | ISA sources; pinned by simpler via `pto_isa.pin` | `<pto-isa-checkout>` |
| ptoas | Assembler toolchain; version pinned in `pypto/toolchain/versions.env` | whl packages, installed into the env |
| conda env | Everything installs here | `<conda-env>` |

Substitute the user's actual checkout paths and env name for the placeholders
(ask when not given). `PYPTO_LIB_ROOT` overrides kernel-source discovery for
executors (default: the serving repo's `pypto-lib` submodule).

## Refresh procedure

Verify before refreshing: record the currently installed state (pypto commit,
simpler build commit, ptoas version) and compare it against the *target* the
user pinned for the task. Only when the target requires different commits —
or the user explicitly asks to move forward — run the refresh, and checkout
the target commits, not blindly the latest `origin/main`. Refreshing past the
target makes device results unverifiable against it.

1. In `<pypto-checkout>`: `git fetch origin`, checkout the target commit (the
   user's pin, or latest `origin/main` when the target is "current main"),
   and update the `runtime` submodule to the pinned commit.
2. Reinstall pypto: `rm -rf build && pip install -e ".[dev]"`.
3. Reinstall simpler: read `cat <pypto-checkout>/runtime/pto_isa.pin`,
   checkout `<pto-isa-checkout>` to that commit, then in
   `<pypto-checkout>/runtime`:
   `rm -rf build && PTO_ISA_ROOT=<pto-isa-checkout>/ pip install -v .`
4. Verify ptoas: the version pinned in `pypto/toolchain/versions.env` must
   match what the env has installed. If not, look for the matching whl in the
   local ptoas package directory, else fetch it from
   https://github.com/hw-native-sys/PTOAS/releases.
5. Verify the *installed* simpler matches the pin:
   `python -c "import _task_interface as t; print(t.__build_commit__)"`
   must equal `git -C <pypto-checkout>/runtime rev-parse HEAD`.
   simpler is a compiled, non-editable install — a correct submodule checkout
   says nothing about the installed `.so`. It has been silently two pins
   stale before. When the pin moves, always redo step 3 even if the checkout
   already looks current.
6. If the simpler build fails with ninja `multiple outputs aren't (yet?)
   supported by depslog`, the system ninja is too old (1.8.2); `pip install
   ninja` into the env (1.13+) puts a working ninja first on PATH.

Stack sanity gates after a refresh: the pypto repo's
`tests/test_qwen3_accuracy.py` and `tests/test_qwen3_jit_cache.py` (cold and
READY paths) should pass repeatedly on a single card (~80 s / ~123 s). If
these fail, fix the stack before debugging serving-side model code.

## Switching stack versions

"Switching" means re-pointing the env's installed packages at another
checkout, then redoing steps 2-4 for everything downstream of the change:
moving pypto requires rebuilding simpler (its pin may move with it), and a
moved `pto_isa.pin` requires rebuilding simpler even if pypto is untouched.

The default target is pypto `origin/main`, but a task may pin older commits —
verify against the task's target, not against whatever is newest. Notably,
the build-identity JIT
cache refinement (fast READY hits, `PYPTO_CACHE_IDENTITY=build`,
`PYPTO_CACHE_EPOCH` invalidation) is open PR hw-native-sys/pypto#2909 and is
**not** on main — do not expect those behaviors locally until it merges, and
re-verify serving-side compile-cache contracts when it does.

## JIT cache notes relevant to adaptation runs

- Persistence is fail-closed: a dirty simpler runtime git checkout or missing
  NumPy/CANN build evidence silently bypasses persistent caching. If every
  run recompiles, check for a dirty runtime checkout first.
- `PYPTO_PROG_BUILD_DIR` is treated as an explicit output/rebuild request.
  Setting it "helpfully" disables JIT cache reuse; only set it when a named
  build dir is wanted.
- First-run compile time is not a performance signal. Compare second runs
  when timing an adaptation.
