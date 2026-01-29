cuMemAlloc {
    if(oom_check()) {
        CUDA_ERROR("Out of memory");
    }
    return dlsym(RTLD_NEXT, "cuMemAlloc");
}