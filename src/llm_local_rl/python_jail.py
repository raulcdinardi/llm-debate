"""Trusted subprocess bootstrap for default Docker capabilities.

The jail has no /proc, devices, binaries, secrets or writable paths. All IDs and
capabilities are dropped before code arrives. A seccomp allowlist prevents
networking, process creation, signals, privilege changes and namespace changes.
"""

BOOTSTRAP = r'''
import ctypes,os,sys
lib=ctypes.CDLL('libseccomp.so.2')
lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_load.restype=ctypes.c_int
lib.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=lib.seccomp_init(0x50000|1)
assert ctx
for name in 'read write readv writev close close_range fstat newfstatat stat lstat statx lseek mmap mprotect munmap mremap madvise brk rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex getrandom openat access faccessat faccessat2 readlink readlinkat getcwd getdents64 fcntl ioctl clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep getpid getppid gettid getuid geteuid getgid getegid getgroups uname sysinfo sched_getaffinity getrusage getrlimit exit exit_group'.split():
 nr=lib.seccomp_syscall_resolve_name(name.encode())
 if nr>=0:assert lib.seccomp_rule_add(ctx,0x7fff0000,nr,0)==0
os.chroot(sys.argv[1]);os.chdir('/')
os.setgroups([]);os.setresgid(65534,65534,65534);os.setresuid(65534,65534,65534)
assert os.getresuid()==(65534,65534,65534)
os.environ.clear()
def _activate_sandbox():
 # libseccomp enables irreversible no_new_privs before loading the filter.
 assert lib.seccomp_load(ctx)==0
 lib.seccomp_release(ctx)
'''
