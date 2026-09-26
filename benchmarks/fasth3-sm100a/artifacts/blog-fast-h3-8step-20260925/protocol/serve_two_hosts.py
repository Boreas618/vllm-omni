"""Launch the declared UniServe head and its rank-1 host process.

Both hosts must have identical source, native builds, model assets and Python
environments at /workspace/UniServe. SSH rank-1 must work without interaction.
The host binary terminates its owned ranks when the head connection closes.
"""

import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/UniServe')
head = subprocess.Popen(sys.argv[1:], cwd=ROOT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
remote = None


def stop(signum, frame):
    head.terminate()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    for line in head.stdout:
        print(line, end='', flush=True)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        match = re.search(r'address="[^"]*:(\d+)"', clean)
        if remote is None and match and 'awaiting a launcher' in clean:
            port = int(match.group(1))
            # SSH_CLIENT names the head address reachable from the peer.
            code = (
                'import os; '
                f'os.chdir({str(ROOT)!r}); '
                f'os.environ["PYTHONPATH"]={str(ROOT)!r}; '
                f'binary={str(ROOT / "target/release/uniserve-host")!r}; '
                f'address=os.environ["SSH_CLIENT"].split()[0]+":{port}"; '
                'os.execv(binary,[binary,"--head",address,"--host-identity","rank-1"])'
            )
            remote = subprocess.Popen([
                'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                'rank-1', shlex.join(['/usr/bin/python3', '-c', code]),
            ])
    status = head.wait()
finally:
    if head.poll() is None:
        head.terminate()
        head.wait()
    if remote is not None:
        try:
            remote.wait(timeout=20)
        except subprocess.TimeoutExpired:
            remote.terminate()
            remote.wait()
sys.exit(status)
