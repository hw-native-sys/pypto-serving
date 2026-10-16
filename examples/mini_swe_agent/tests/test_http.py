# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the real text client and Agent loop against a scripted Chat API."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from minisweagent.agents.default import DefaultAgent

from client.model import CompleteTextModel
from smoke import FINISH_MARKER, SYSTEM_TEMPLATE, SmokeEnvironment


class HttpRoundTripTests(unittest.TestCase):
    def test_plain_chat_api_runs_commands_and_returns_observed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            expected = 'observed_' + secrets.token_hex(6)
            (workspace / 'input.txt').write_text(expected + '\n')
            requests = []

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    requests.append((self.path, payload))
                    command = ('cat input.txt' if len(requests) == 1 else
                               f"printf '%s\\n' '{FINISH_MARKER}' '{expected}'")
                    body = json.dumps({
                        'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0,
                        'model': 'qwen3-agent',
                        'choices': [{'index': 0, 'message': {'role': 'assistant',
                                      'content': f'```bash\n{command}\n```'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
                    }).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                model = CompleteTextModel(
                    model_name='openai/qwen3-agent', cost_tracking='ignore_errors',
                    action_regex=r'```bash\s*\n(.*?)\n```',
                    model_kwargs={
                        'api_base': f'http://127.0.0.1:{server.server_port}/v1',
                        'api_key': 'local-placeholder', 'temperature': 0.0,
                        'max_tokens': 256, 'stream': False, 'num_retries': 0, 'timeout': 10,
                        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
                    },
                )
                environment = SmokeEnvironment(cwd=directory, timeout=5)
                with self.assertRaises(ValueError):
                    environment.execute({'command': 'cat input.txt; touch unexpected'})
                self.assertFalse((workspace / 'unexpected').exists())
                self.assertEqual(environment.commands, [])
                agent = DefaultAgent(
                    model, environment, system_template=SYSTEM_TEMPLATE, instance_template='{{task}}',
                    step_limit=4, cost_limit=0, wall_time_limit_seconds=30,
                    output_path=workspace / 'trajectory.json',
                )
                with patch.dict('os.environ', {'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'}):
                    result = agent.run('Read input.txt and submit its content.')
                self.assertEqual(result['exit_status'], 'Submitted')
                self.assertEqual(result['submission'].strip(), expected)
                self.assertEqual(agent.n_calls, 2)
                self.assertEqual(len(requests), 2)
                self.assertIn(expected, requests[1][1]['messages'][-1]['content'])
                for path, payload in requests:
                    self.assertEqual(path, '/v1/chat/completions')
                    self.assertEqual(payload['model'], 'qwen3-agent')
                    self.assertEqual(payload['max_tokens'], 256)
                    self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
                    self.assertNotIn('agent_hint', payload)
                    self.assertNotIn('tools', payload)
                self.assertTrue((workspace / 'trajectory.json').is_file())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
