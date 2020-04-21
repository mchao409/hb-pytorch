#include <kernel_common.hpp>


extern "C" {

  __attribute__ ((noinline)) int tensorlib_eye(
    hb_tensor_t* output,
    long* n, long* m) {
    auto y = HBTensor<float>(output);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  // HB_EMUL_REG_KERNEL(tensorlib_eye, hb_tensor_t*, int*, int*);

}