# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyze one DPLB capture; optionally report a second capture alongside it."""

import argparse
import json
from pathlib import Path

from pypto_serving.observability.dplb_dfx_report import write_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    summaries = [write_report(args.capture)]
    if args.compare:
        summaries.append(write_report(args.compare))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
