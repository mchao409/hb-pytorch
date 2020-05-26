#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <cmath>

namespace at { 
namespace native {

  Tensor& arange_out_hb(Tensor& result, Scalar start, Scalar end, Scalar step) {
    int n = std::ceil((end.to<float>() - start.to<float>()) / step.to<float>());
    result.resize_({n});
    hb_offload_kernel(result, start.to<int>(), step.to<int>(), "tensorlib_arange");
    return result;
  }

}}

