//====================================================================
// threshold kernel
// 03/19/2020 Lin Cheng (lc873@cornell.edu)
// 03/29/2020 Angela Zou (az292@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_threshold(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          bsg_tensor_t* t2_p,
          float* _threshold_scalar_p,
          float* _value_scalar_p) {

    float threshold = *_threshold_scalar_p;
    float value    = *_value_scalar_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    brg_tile_elementwise_for(t0_p, t1_p, t2_p,
      [&](float self, float other) {
        if (self <= threshold) {
          return value;
        } else {
          return other;
        }
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_threshold, bsg_tensor_t*, bsg_tensor_t*,
                     bsg_tensor_t*, float*, float*)

}