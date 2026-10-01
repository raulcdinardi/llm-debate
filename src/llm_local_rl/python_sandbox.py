"""Fresh restricted process per evaluation; time is measured by the host."""
from __future__ import annotations
import ctypes
import errno
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

WORKER=r'''
import ast,json,resource,sys,time
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if '_activate_sandbox' in globals():_activate_sandbox()
print('READY',flush=True)
x=json.load(sys.stdin);scope={}
exec(compile(x['code'],'<candidate>','exec'),scope)
setup=compile(x['task']['test_setup_code'],'<setup>','exec')
checks=[]
for text in x['task']['test_list']+x['task'].get('challenge_test_list',[]):
 tree=ast.parse(text);assert len(tree.body)==1 and isinstance(tree.body[0],ast.Assert)
 checks.append(compile(ast.Expression(tree.body[0].test),'<check>','eval'))
def suite():
 exec(setup,scope)
 return [bool(eval(test,scope)) for test in checks]
start=time.perf_counter_ns()
passed=suite()
initial_ns=time.perf_counter_ns()-start
if x['loops']==0:
 # A single suite can take seconds (#123). Never multiply it just to calibrate.
 if initial_ns < 1000000:
  start=time.perf_counter_ns()
  for _ in range(10):suite()
  initial_ns=(time.perf_counter_ns()-start)/10
 print(json.dumps({'correct':all(passed),'calibration_loops':max(1,min(30000,int(6000000/max(1,initial_ns))))}))
else:
 for _ in range(x['loops']-1):suite()
 print(json.dumps({'correct':all(passed),'checks_passed':sum(passed),'checks_total':len(passed)}))
'''

class PythonSandbox:
    def __init__(self,settings):
        self.settings=settings
        if settings.get('backend')=='chroot_seccomp':
            return
        self.filter=Path(settings['state_dir'])/'sandbox.bpf'
        self.filter.parent.mkdir(parents=True,exist_ok=True)
        lib=ctypes.CDLL('libseccomp.so.2')
        lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
        lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
        lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
        lib.seccomp_export_bpf.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.seccomp_release.argtypes=[ctypes.c_void_p]
        ctx=lib.seccomp_init(0x7fff0000)
        for name in 'clone clone3 fork vfork socket socketpair ptrace process_vm_readv process_vm_writev mount umount2 pivot_root unshare setns bpf perf_event_open keyctl add_key request_key userfaultfd io_uring_setup reboot kexec_load open_by_handle_at'.split():
            nr=lib.seccomp_syscall_resolve_name(name.encode())
            if nr>=0:assert lib.seccomp_rule_add(ctx,0x50000|errno.EPERM,nr,0)==0
        with self.filter.open('wb') as f:assert lib.seccomp_export_bpf(ctx,f.fileno())==0
        lib.seccomp_release(ctx)
    def run(self,task,*,code,loops):
        if self.settings.get('backend')=='chroot_seccomp':
            from llm_local_rl.python_jail import BOOTSTRAP
            cmd=[self.settings['python_host'],'-I','-S','-c',BOOTSTRAP+'\n'+WORKER,self.settings['jail_root']]
            return self._run_process(cmd,task,code,loops,())
        with self.filter.open('rb') as seccomp:
            cmd=[self.settings['bwrap'],'--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL']
            for host,guest in self.settings['readonly_mounts']:cmd+=['--ro-bind',host,guest]
            cmd+=['--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--clearenv','--seccomp',str(seccomp.fileno()),'--',self.settings['python_inside'],'-I','-S','-c',WORKER]
            return self._run_process(cmd,task,code,loops,(seccomp.fileno(),))

    def _run_process(self,cmd,task,code,loops,pass_fds):
            start=time.perf_counter_ns();deadline=time.monotonic()+self.settings['timeout_s']
            process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=pass_fds,start_new_session=True,env={})
            sel=selectors.DefaultSelector();sel.register(process.stdout,selectors.EVENT_READ,'out');sel.register(process.stderr,selectors.EVENT_READ,'err')
            chunks={'out':bytearray(),'err':bytearray()};timed_start=None;status='ok'
            def kill_group():
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            try:
                while sel.get_map():
                    remaining=deadline-time.monotonic()
                    if remaining<=0:status='timeout';break
                    for key,_ in sel.select(min(remaining,.1)):
                        data=os.read(key.fileobj.fileno(),8192)
                        if not data:sel.unregister(key.fileobj);continue
                        chunks[key.data].extend(data)
                        if sum(map(len,chunks.values()))>65536:status='output_limit';break
                        if timed_start is None and b'\n' in chunks['out']:
                            if chunks['out']!=b'READY\n':status='protocol_error';break
                            chunks['out'].clear();timed_start=time.perf_counter_ns()
                            process.stdin.write(json.dumps({'task':task,'code':code,'loops':loops}).encode());process.stdin.close()
                    if status!='ok':break
                if status!='ok':kill_group()
                process.wait(timeout=max(.1,deadline-time.monotonic()))
            except (BrokenPipeError,subprocess.TimeoutExpired):
                status='process_error';kill_group();process.wait()
            finally:sel.close()
            end=time.perf_counter_ns()
            if process.returncode!=0 and status=='ok':status='runtime_error'
            if status!='ok':return {'status':status,'wall_ns':end-start,'stderr':chunks['err'].decode(errors='replace')[-1000:]}
            try:
                result=json.loads(chunks['out'])
                assert set(result) in ({'correct','calibration_loops'},{'correct','checks_passed','checks_total'})
                assert type(result['correct']) is bool
            except (ValueError,AssertionError,TypeError):return {'status':'protocol_error','wall_ns':end-start}
            return dict(result,status='ok',elapsed_ns=end-timed_start,wall_ns=end-start)
