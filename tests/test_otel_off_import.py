"""Keep telemetry modules unloaded when export is off."""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_off_keeps_otel_out_of_cli_and_engine(tmp_path):
    engine_code = f"""import json,sys
class RejectOtel:
 def find_spec(self,name,*args):
  if name == 'stateframe.otel' or name.startswith('stateframe.otel.'):
   raise AssertionError('telemetry imported with export disabled')
sys.meta_path.insert(0,RejectOtel())
from stateframe.scripted_endpoint import Script
script=Script({{'steps':[{{'tool':'done','arguments':{{'summary':'finished'}}}}]}})
assert script.next_call(['done'])['tool']=='done'
script.assert_consumed()
sys.path.insert(0,{str(ROOT / 'src/stateframe/engine')!r})
import stateframe._child as engine
from contextlib import ExitStack
engine._main([],ExitStack(),teardown=None)
import kernel_stateframe_runtime
open('engine-run.json','w').write(json.dumps(sorted(n for n in sys.modules if n == 'stateframe.otel' or n.startswith('stateframe.otel.'))))
"""
    code = f"""import json,sys
class RejectOtel:
 def find_spec(self,name,*args):
  if name == 'stateframe.otel' or name.startswith('stateframe.otel.'):
   raise AssertionError('telemetry imported with export disabled')
sys.meta_path.insert(0,RejectOtel())
def modules(): return sorted(n for n in sys.modules if n == 'stateframe.otel' or n.startswith('stateframe.otel.'))
import stateframe.cli as cli
open('cli-import.json','w').write(json.dumps(modules()))
def scripted_run(options,printer):
 result=__import__('subprocess').run([sys.executable,'-B','-c',{engine_code!r}],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 return {{'exit_code':0}}
cli.run=scripted_run
assert cli.main(['run','--repo','.','--task','Complete the scripted task.','--endpoint','http://127.0.0.1:8080/v1','--model','scripted','--json'])==0
open('scripted-run.json','w').write(json.dumps(modules()))
"""
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("STATEFRAME_", "REACT_", "SF_LAB_"))}
    environment.update(PYTHONPATH=str(ROOT / "src"), OTEL_SDK_DISABLED="true")
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    output = (completed.stdout + completed.stderr).replace(str(tmp_path), "<tmp>").replace(str(ROOT), "<repo>")
    assert completed.returncode == 0, output
    for name in ("cli-import.json", "scripted-run.json", "engine-run.json"):
        assert json.loads((tmp_path / name).read_text(encoding="utf-8")) == []
