# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public regression tests for the log report."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from report import summarize


def row(latency, status="ok", model="alpha"):
    return {"latency_ms": latency, "status": status, "model": model}


class ReportTests(unittest.TestCase):
    def test_failed_requests_do_not_contribute_latency(self):
        self.assertEqual(summarize([row(20), row(1000, "error")]), {
            "requests": 2, "successful": 1, "failed": 1, "mean_ms": 20, "p95_ms": 20,
        })

    def test_nearest_rank_p95(self):
        for size, expected in ((5, 5), (20, 19), (21, 20)):
            with self.subTest(size=size):
                self.assertEqual(summarize([row(n) for n in range(1, size + 1)])["p95_ms"], expected)

    def test_all_failed(self):
        self.assertEqual(summarize([row(500, "error")]), {
            "requests": 1, "successful": 0, "failed": 1, "mean_ms": 0, "p95_ms": 0,
        })

    def test_cli_model_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "access.jsonl"
            log.write_text("\n".join(json.dumps(r) for r in [row(10), row(90, model="beta"), row(999, "error")]))
            result = subprocess.run([sys.executable, "cli.py", str(log), "--model", "alpha"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), {
                "requests": 2, "successful": 1, "failed": 1, "mean_ms": 10, "p95_ms": 10,
            })


if __name__ == "__main__":
    unittest.main()
