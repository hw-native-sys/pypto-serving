py_mod = import('python')
py     = py_mod.find_installation('python3', required: false)

if not py.found()
    message('python3 not found — skipping serving_platform bindings')
    subdir_done()
endif

pybind11_inc_dir = run_command(
    py, ['-c', 'import pybind11; print(pybind11.get_include())'],
    check: false,
)
if pybind11_inc_dir.returncode() != 0
    message('pybind11 not installed — skipping serving_platform bindings')
    subdir_done()
endif

pybind11_inc = include_directories(pybind11_inc_dir.stdout().strip())
examples_inc = include_directories('../examples/include')

py.extension_module(
    'serving_platform',
    'bridge.cpp',
    dependencies     : [servingBuildDep],
    include_directories: [pybind11_inc, examples_inc],
    cpp_args         : ['-fvisibility=hidden'],
    subdir           : 'platform',
    install          : false,
)
