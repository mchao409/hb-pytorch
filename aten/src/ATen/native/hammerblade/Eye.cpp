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

  Tensor& eye_out_hb(Tensor& output, long n, long m) {
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;
    output.resize_({n,m});
    if(m == -1) {
      m = n;
    }
    // Tensor output = at::empty(
    //                {n,m}, input.options());
    device_args.push_back(create_device_tensor(output, device_ptrs));
    device_args.push_back(create_device_scalar(n));
    device_args.push_back(create_device_scalar(m));
    c10::hammerblade::offload_kernel("tensorlib_eye", device_args);
    cleanup_device(device_args, device_ptrs);
    //  eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
    return output;
  }

}}

