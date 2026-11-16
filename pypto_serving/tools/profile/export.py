# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Streaming Chrome Trace export; memory scales with tracks and requests, not events."""

import json
from pathlib import Path


def merge(directory):
    directory = Path(directory)
    target = directory / "trace.json"
    pids, tids, requests = {}, {}, {}
    count = broken = 0
    links = {}
    with target.with_suffix('.tmp').open('w') as output:
        output.write('{"displayTimeUnit":"ms","traceEvents":[')
        for file in sorted(directory.rglob('*.jsonl')):
            with file.open() as source:
                for line in source:
                    try:
                        event = json.loads(line)
                        process = event['pid']
                        event['pid'] = pids.setdefault(process, len(pids)+1)
                        thread = process, str(event['tid'])
                        if thread not in tids:
                            tids[thread] = len(tids)+1
                            metadata = dict(ph="M",name="thread_name",pid=event['pid'],tid=tids[thread],
                                            args=dict(name=thread[1]))
                            output.write((',' if count else '')+json.dumps(metadata,separators=(',',':')))
                            count += 1
                        event['tid'] = tids[thread]
                        if event['ph'] != 'M':
                            for root in event.get('args', {}).get('roots', ()):
                                row = requests.setdefault(root, dict(first_us=event['ts'], last_us=event['ts'], stages={}, milestones={}))
                                row['first_us'] = min(row['first_us'], event['ts'])
                                row['last_us'] = max(row['last_us'], event['ts']+event.get('dur', 0))
                                if event['name'] in ('http.ingress', 'http.first_body', 'http.end',
                                                       'generation.finished', 'request.resources_retired'):
                                    row['milestones'].setdefault(event['name'], []).append(dict(
                                        ts_us=event['ts'], process=event['pid'], **event.get('args', {})))
                                if event.get('cat') not in ('python','native-call'):
                                    stage = row['stages'].setdefault(event['name'], dict(count=0, first_us=event['ts'], last_us=0, span_us=0))
                                    stage['first_us'] = min(stage['first_us'], event['ts'])
                                    stage['count'] += 1
                                    stage['last_us'] = max(stage['last_us'],event['ts']+event.get('dur',0))
                                    stage['span_us'] += event.get('dur',0)
                                if event['name'] in ('http.ingress','request.bind'):
                                    links.setdefault(root, []).append(dict(pid=event['pid'], tid=event['tid'],
                                        ts=event['ts'], name=event['name']))
                        output.write((',' if count else '') + json.dumps(event,separators=(',',':')))
                        count += 1
                    except (ValueError, KeyError, TypeError):
                        broken += 1
        for root, points in links.items():
            points = sorted((p for p in points if p['name'] != 'engine.tokens'), key=lambda p:p['ts'])
            for index,(left,right) in enumerate(zip(points,points[1:])):
                identity = root+":"+str(index)
                for ph, point in (("s",left),("f",right)):
                    flow = dict(name="request association", cat="association", ph=ph,
                                id=identity, ts=point['ts'], pid=point['pid'], tid=point['tid'],
                                args=dict(root=root, causal_clock_order_not_guaranteed=True))
                    if ph == "f":
                        flow['bp'] = 'e'
                    output.write((',' if count else '')+json.dumps(flow,separators=(',',':')))
                    count += 1
        output.write(']}\n')
    target.with_suffix('.tmp').replace(target)
    (directory/'requests.json').write_text(json.dumps(requests,indent=2))
    return dict(trace=str(target), requests=str(directory/'requests.json'), events=count,
                broken_records=broken, processes=len(pids), tracks=len(tids))
