# mini-swe-agent HTTP frontend

Run an interactive coding or chat session against PyPTO Serving's existing
OpenAI-compatible Chat API. The Agent can read workspace files, execute shell
commands after confirmation, edit code, and run tests. Model requests use HTTP;
the frontend does not start or own the Serving engine.

The example keeps entry points, implementation, and CPU checks in separate locations:

```text
mini_swe_agent/
  interactive.py       # Interactive terminal entry point
  smoke.py             # Bounded model/command round-trip probe
  client/
    model.py           # Complete text-action validation
    budget.py          # Token counting and history selection
    runtime.py         # Coding prompts and completion verification
  tests/
    test_client.py     # Action, context, and verification checks
    test_http.py       # Scripted HTTP round trip
  requirements.txt
```

## Install the optional client

From the repository root, create a separate client environment:

```bash
python3.10 -m venv .venvs/mini-swe-agent
.venvs/mini-swe-agent/bin/python -m pip install -r examples/mini_swe_agent/requirements.txt
```

The requirements pin the published `mini-swe-agent==2.4.6` release and its client
dependencies. They are optional example dependencies, not Serving runtime
dependencies. No NPU runtime is needed in the client environment.

## Start the server

Use the Serving environment and an available NPU. On a shared host, acquire the
device through the host's task queue before starting Serving; do not start a
second model on a device already in use.

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --served-model-name qwen3-agent \
  --backend npu --platform a2a3 --device 0 \
  --max-model-len 4096 --max-num-seqs 16 \
  --host 127.0.0.1 --port 8899
```

Replace the model directory and device with your installation and assigned
device. For additional server options, see the
[online serving guide](../../docs/user-guide/online-serving.md).

Wait for startup before running the client:

```bash
curl --noproxy '*' http://127.0.0.1:8899/health
```

## Start an interactive session

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
.venvs/mini-swe-agent/bin/python examples/mini_swe_agent/interactive.py \
  --api-base http://127.0.0.1:8899/v1 \
  --model qwen3-agent \
  --tokenizer-path /path/to/Qwen3-14B \
  --context-limit 4096 \
  --profile coding \
  --workspace /path/to/your/project \
  --task 'Read the project requirements, fix the bug, and run the tests.'
```

Use `--profile chat` for conversation and file summaries. The `coding` profile
uses upstream coding prompts adapted to plain `bash` text actions, so a
structured tool-call parser is not required on the server. `--thinking` enables
Qwen thinking within the same output budget; it is disabled by default.

At the action confirmation prompt, press Enter to approve the displayed command,
or type a comment to reject it and steer the Agent. At the completion prompt,
press Enter to leave or enter another message. Commands execute locally in
`--workspace` using the client's Python environment; the workspace setting is a
working directory, not an operating-system sandbox. Shell directory changes do
not persist between commands.

### Context and execution records

Set `--context-limit` to the server's actual `--max-model-len`, and use tokenizer
files matching the served model. The client reserves `--max-tokens` for output,
keeps the system prompt, original task, and user follow-ups, and drops old
action/observation groups when necessary. Large observations can be shortened
with an explicit marker. If protected instructions cannot fit, the request
fails instead of silently truncating the task. Full observations remain in the
trajectory; this is history selection, not model summarization.

Each session writes a separate directory under `outputs/mini_swe_agent_interactive/`
containing `trajectory.json` and any configured verification records. These
files contain model replies and tool output and should be reviewed before
sharing. Outputs are local artifacts and are not committed.

### Optional completion checks

To require evidence before accepting the Agent's completion:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
.venvs/mini-swe-agent/bin/python examples/mini_swe_agent/interactive.py \
  --api-base http://127.0.0.1:8899/v1 \
  --tokenizer-path /path/to/Qwen3-14B \
  --workspace /path/to/your/project \
  --verify-command 'python -m unittest -v' \
  --require-change module.py \
  --protect test_module.py
```

The verification command runs before submission is accepted. Listed source
files must differ from their initial contents; protected files must remain
unchanged. A failing check returns evidence to the Agent to continue working.
Verification itself must not modify tracked files. These are local acceptance
checks, not a security boundary or proof of general task correctness.

## Bounded integration smoke test

With the server running:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
.venvs/mini-swe-agent/bin/python examples/mini_swe_agent/smoke.py \
  --api-base http://127.0.0.1:8899/v1 --model qwen3-agent
```

The probe creates a random single-line file, asks the model to read it, and
checks that its final submission matches the observed content. It only allows
the read command and the bounded submission command. A passing run therefore
checks a real model-to-command-to-observation round trip, not just API reachability.
This is not a coding benchmark or a performance claim.

## CPU checks

No model weights or NPU are needed for these checks:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
.venvs/mini-swe-agent/bin/python -m unittest discover \
  -s examples/mini_swe_agent/tests -t examples/mini_swe_agent -v
```

Checks cover action parsing, incomplete generations, context budgeting,
completion acceptance, and an HTTP round trip with scripted model replies.
The HTTP fixture verifies protocol and execution behavior; it does not replace
the real NPU smoke test.
