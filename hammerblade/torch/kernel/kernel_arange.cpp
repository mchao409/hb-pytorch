#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_arange(
    hb_tensor_t* output, int* start_p, int* step_p) {
    HBTensor<long> y(output);
    int start = *start_p;
    int step  = *step_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_parallel_for(y.numel(), [&](size_t i) {
        y(i) = i * step + start;
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_arange, hb_tensor_t*, int*, int*);
}