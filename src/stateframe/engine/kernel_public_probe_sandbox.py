"""Linux probe runner: finite resources and no descendants after trusted setup."""

RUNNER_SOURCE = '''
import ctypes
import errno
import resource
import sys

if sys.platform != 'linux':
    raise RuntimeError('public probe process containment currently requires Linux seccomp')
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
# Missing library, unsupported syscalls or filter installation errors fail closed.
seccomp = ctypes.CDLL('libseccomp.so.2', use_errno=True)
seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
seccomp.seccomp_init.restype = ctypes.c_void_p
seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
seccomp.seccomp_rule_add.restype = ctypes.c_int
seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
seccomp.seccomp_load.restype = ctypes.c_int
seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
context = seccomp.seccomp_init(0x7fff0000)
if not context:
    raise RuntimeError('seccomp initialization failed')
try:
    for name in (b'clone', b'clone3', b'fork', b'vfork', b'execve', b'execveat'):
        number = seccomp.seccomp_syscall_resolve_name(name)
        if number < 0 or seccomp.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
            raise RuntimeError('seccomp process rule unavailable')
    if seccomp.seccomp_load(context) != 0:
        raise RuntimeError('seccomp installation failed')
finally:
    seccomp.seccomp_release(context)
import pytest
sys.exit(pytest.main(sys.argv[1:]))
'''
