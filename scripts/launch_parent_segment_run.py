"""Launch the approved runner with exclusive, preserved log/PID records."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_parent_segment_grade3 import protocol_and_directory,save

_,_,directory,_=protocol_and_directory()
directory.mkdir(parents=True,exist_ok=True)
# The runner itself acquires the exclusive full-run lock before any work.
stamp=str(time.time_ns())
log=directory/f'runner-{stamp}.log'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),
    'COHERE_BOTO3_PATH':'/private/tmp/pycon-cohere-boto3-20261002',
    'OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2',
    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'true'}
with log.open('xb') as output:
    process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_parent_segment_grade3.py')],
        cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,log=str(log.relative_to(ROOT)),directory=str(directory.relative_to(ROOT)),
    command='scripts/run_parent_segment_grade3.py',launched_time_ns=int(stamp))
save(directory/f'launch-{stamp}.json',record)
print(record)
