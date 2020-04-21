#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { 
namespace native {
  
  Tensor& eye_out_hb(Tensor& out, long n) {
    return eye_out_hb(out,n,n);
  //   std::vector<eva_t> device_args;
  //   std::vector<eva_t> device_ptrs;
  //   // std::vector<int> vect{ 10, 20, 30 };
  //   // auto output_t = at::empty(
  //   //                vect);
  // c10::hammerblade::offload_kernel("tensorlib_eye", device_args);
  // // cleanup_device(device_args, device_ptrs);
  // //  eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
  // return out;
  }

  Tensor& eye_out_hb(Tensor& out, long n, long m) {
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;
    auto output_t = at::empty(
                   {n,m}, out.options());
  c10::hammerblade::offload_kernel("tensorlib_eye", device_args);
  // cleanup_device(device_args, device_ptrs);
  //  eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
  return out;
  }

}}

