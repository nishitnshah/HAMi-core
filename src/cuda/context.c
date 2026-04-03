#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int ctx_activate[16];

/*
 * CUDA context scheduling fix for multi-pod GPU sharing.
 *
 * Problem:
 *   With CU_CTX_SCHED_AUTO (flags=0x00, the default), the CUDA driver
 *   automatically switches from spin-polling to OS-blocking (BLOCKING_SYNC)
 *   when it detects multiple processes sharing the same GPU. With 2 pods x 8
 *   ranks = 16 processes, AUTO degrades every cudaStreamSynchronize from ~4µs
 *   to ~257µs — even when the GPU kernel completed in 2µs. This causes a 3x
 *   training throughput regression when two pods share the same 8 H100s.
 *
 * Fix:
 *   Force CU_CTX_SCHED_SPIN on every context at creation time. HAMi intercepts
 *   all context creation calls (cuCtxCreate_v2, cuCtxCreate_v3,
 *   cuDevicePrimaryCtxSetFlags_v2), including those from NCCL and PyTorch
 *   internals, so this applies universally.
 *
 *   CU_CTX_SCHED_SPIN = 0x01: CPU spin-polls for GPU completion, returning
 *   immediately when done regardless of how many processes share the GPU.
 *   Tradeoff: higher CPU utilization during GPU waits, acceptable for
 *   GPU-bound training workloads.
 *
 * Opt-out:
 *   Set HAMI_CTX_SCHED_SPIN=0 in the environment to disable and restore the
 *   driver's default AUTO behavior (useful for CPU-bound inference workloads
 *   that prefer to yield the CPU while waiting for long GPU operations).
 */
#define CU_CTX_SCHED_MASK 0x07u
#define CU_CTX_SCHED_SPIN 0x01u

static int hami_force_spin = -1;  /* -1 = uninitialized */

static int should_force_spin() {
    if (hami_force_spin == -1) {
        const char *env = getenv("HAMI_CTX_SCHED_SPIN");
        /* Default ON: force spin unless explicitly disabled */
        hami_force_spin = (env == NULL || env[0] != '0') ? 1 : 0;
        if (hami_force_spin) {
            LOG_INFO("HAMi: forcing CU_CTX_SCHED_SPIN on all contexts (set HAMI_CTX_SCHED_SPIN=0 to disable)");
        } else {
            LOG_INFO("HAMi: CU_CTX_SCHED_SPIN disabled via HAMI_CTX_SCHED_SPIN=0");
        }
    }
    return hami_force_spin;
}

static unsigned int apply_spin_flag(unsigned int flags) {
    if (!should_force_spin())
        return flags;
    return (flags & ~CU_CTX_SCHED_MASK) | CU_CTX_SCHED_SPIN;
}


CUresult cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int* flags, int* active ){
    LOG_DEBUG("into cuDevicePrimaryCtxGetState dev=%d",dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxGetState,dev,flags,active);
    return res;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev){
    LOG_INFO("dev=%d context_size=%ld",dev,context_size);
    /* Force SPIN before activating the primary context.
     * cuDevicePrimaryCtxSetFlags must be called before the primary context
     * is active. Calling it here (before cuDevicePrimaryCtxRetain) ensures
     * SPIN is set even when callers (PyTorch, NCCL) retain the primary
     * context directly rather than going through cuCtxCreate. */
    if (should_force_spin()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxSetFlags_v2,dev,CU_CTX_SCHED_SPIN);
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRetain,pctx,dev);
    if (ctx_activate[dev] == 0) {
        add_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    if (context_size>0) {
        ctx_activate[dev] = 1;
    }
    return res;
}


CUresult cuDevicePrimaryCtxSetFlags_v2( CUdevice dev, unsigned int flags ){
    flags = apply_spin_flag(flags);
    LOG_DEBUG("into cuDevicePrimaryCtxSetFlags dev=%d flags=%d",dev,flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxSetFlags_v2,dev,flags);
}

CUresult cuDevicePrimaryCtxRelease_v2( CUdevice dev ){
    if (ctx_activate[dev] == 1) {
        rm_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    ctx_activate[dev] = 0;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    return res;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetDevice,device);
    return res;
}

CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int flags, CUdevice dev ){
    flags = apply_spin_flag(flags);
    LOG_DEBUG("into cuCtxCreate pctx=%p flags=%d dev=%d",pctx,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v2,pctx,flags,dev);
    return res;
}

CUresult cuCtxCreate_v3 ( CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev ){
    flags = apply_spin_flag(flags);
    LOG_DEBUG("into cuCtxCreate_v3 pctx=%p paramsArray=%p numParams=%d flags=%d dev=%d",pctx,paramsArray,numParams,flags,dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxCreate_v3,pctx,paramsArray,numParams,flags,dev);
    return res;
}

CUresult cuCtxDestroy_v2 ( CUcontext ctx ){
    LOG_DEBUG("into cuCtxDestroy_v2 ctx=%p",ctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxDestroy_v2,ctx);
}

CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ){
    LOG_INFO("into cuCtxGetApiVersion ctx=%p",ctx);
    CUresult res =  CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetApiVersion,ctx,version);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxGetApiVersion res=%d",res);
    }
    return res;
}

CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig ){
    LOG_DEBUG("into cuCtxGetCacheConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetCacheConfig,pconfig);
}

CUresult cuCtxGetCurrent ( CUcontext* pctx ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetCurrent,pctx);
    return res;
}

CUresult cuCtxGetFlags ( unsigned int* flags ){
    LOG_DEBUG("into cuCtxGetFlags flags=%p",flags);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetFlags,flags);
}

CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit ){
    LOG_DEBUG("into cuCtxGetLimit pvalue=%p",pvalue);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetLimit,pvalue,limit);
}

CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig ){
    LOG_DEBUG("cuCtxGetSharedMemConfig pConfig=%p",pConfig);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetSharedMemConfig,pConfig);
}

CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxGetStreamPriorityRange,leastPriority,greatestPriority);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxGetStreamPriorityRange err=%d",res);
    }
    return res;
}

CUresult cuCtxPopCurrent_v2 ( CUcontext* pctx ){
    LOG_INFO("cuCtxPopCurrent pctx=%p",pctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxPopCurrent_v2,pctx);
}

CUresult cuCtxPushCurrent_v2 ( CUcontext ctx ){
    LOG_INFO("cuCtxPushCurrent ctx=%p",ctx);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxPushCurrent_v2,ctx);
}

CUresult cuCtxSetCacheConfig ( CUfunc_cache config ){
    LOG_DEBUG("cuCtxSetCacheConfig config=%d",config);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetCacheConfig,config);
}

CUresult cuCtxSetCurrent ( CUcontext ctx ){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetCurrent,ctx);
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuCtxSetCurrent111 failed res=%d ctx=%p",res,ctx);
    }
    return res;
}

CUresult cuCtxSetLimit ( CUlimit limit, size_t value ){
    LOG_DEBUG("cuCtxSetLimit");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetLimit,limit,value);
}

CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config ){
    LOG_DEBUG("cuCtxSetSharedMemConfig");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSetSharedMemConfig,config);
}

CUresult cuCtxSynchronize ( void ){
    LOG_DEBUG("INTO CtxSync");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuCtxSynchronize);
    return res;
}
