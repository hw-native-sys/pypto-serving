import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def initialize(device, enable_sdma):
    from simpler.task_interface import ChipWorker
    from simpler_setup.runtime_builder import RuntimeBuilder

    bins = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer")
    warmup = bins.sdma_warmup_path
    if warmup is None or not Path(warmup).is_file():
        raise RuntimeError("SDMA warmup binary is missing; this probe would be inconclusive")
    print(json.dumps({"device": device, "enable_sdma": enable_sdma,
                      "warmup_path": str(warmup),
                      "warmup_sha256": hashlib.sha256(Path(warmup).read_bytes()).hexdigest()}), flush=True)
    worker = ChipWorker()
    try:
        print("INIT_BEGIN", flush=True)
        worker.init(device, bins, log_level=20, enable_sdma=enable_sdma)
        print("INIT_OK", flush=True)
    finally:
        print("FINALIZE_BEGIN", flush=True)
        worker.finalize()
        print("FINALIZE_OK", flush=True)


def probe_all(output):
    devices = [int(value) for value in os.environ["TASK_DEVICE"].replace(",", " ").split()]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise RuntimeError(f"Expected eight distinct assigned devices, got {devices}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # Finish controls before any SDMA failure can affect subsequent measurements.
    for enable_sdma in (False, True):
        for device in devices:
            label = f"device-{device}-sdma-{int(enable_sdma)}"
            log_dir = output / label
            log_dir.mkdir()
            cann = log_dir / "cann"
            cann.mkdir()
            env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(cann))
            started = time.monotonic()
            print(f"START {label}", flush=True)
            with (log_dir / "init.log").open("w") as log:
                command = [sys.executable, __file__, "--device", str(device)]
                if enable_sdma:
                    command.append("--enable-sdma")
                try:
                    rc = subprocess.run(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=240).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            result = {"device": device, "enable_sdma": enable_sdma,
                      "exit_code": rc, "seconds": round(time.monotonic() - started, 2)}
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            print((log_dir / "init.log").read_text(errors="replace"), flush=True)
    return int(any(result["exit_code"] != 0 for result in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    parser.add_argument("--enable-sdma", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device is not None:
        initialize(args.device, args.enable_sdma)
    elif args.output is not None:
        sys.exit(probe_all(args.output.resolve()))
    else:
        parser.error("--device or --output is required")
