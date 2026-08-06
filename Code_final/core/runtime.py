"""Import first, before jax/numpy, in anything that runs in parallel worker
processes: keeps each worker single-threaded and memory-capped."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")


def rss_gb():
    """Resident memory of this process, in GB."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if r > 1e7 else r / 1e6  # macOS bytes vs Linux kB


def cap_memory(gb=1.2):
    """Hard address-space cap; a runaway process dies with MemoryError."""
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        want = int(gb * 1024**3)
        if hard == resource.RLIM_INFINITY or want < hard:
            resource.setrlimit(resource.RLIMIT_AS, (want, hard))
    except (ValueError, OSError):
        pass
