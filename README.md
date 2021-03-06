![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

# PyTorch HammerBlade Port <a href="https://travis-ci.com/github/cornell-brg/hb-pytorch" rel="Travis">![Travis status](https://travis-ci.com/cornell-brg/hb-pytorch.svg?branch=master)</a> ![Lint](https://github.com/cornell-brg/hb-pytorch/workflows/Lint/badge.svg)
This work aims to port PyTorch to HammerBlade.

### How to build PyTorch to use COSIM
  This assumes that you have a working COSIM installed. Then you can either put `hb-pytorch` under `bsg_bladerunner`, or set `BRG_BSG_BLADERUNNER_DIR` to your `bladerunner` path.
 - Clone hb-pytorch repo
    `git clone -b hb-device git@github.com:cornell-brg/hb-pytorch.git`
 - Create python virtual environment
    `python3.6 -m venv ./venv_pytorch`
 - Install dependencies
    `pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest ninja hypothesis`
 - Init pytorch third party dependencies
    `git submodule update --init --recursive`
 - Setup building environment variables.
    `cd hb-pytorch && source setup_cosim_build_env.sh`
 - Build pytorch. This step can take up to 15 minutes
    `python setup.py develop`
 - PyTorch can be used with cosim by running one of the following the executable in place of `python`:
    - `pycosim`: Runs python with cosim backend
    - `pycosim.trace`: Enables device instruction trace
    - `pycosim.wave`: Enbales device instruction trace AND waveform dumps

### How to build PyTorch with Emulation Layer

- Clone this repository:

      git clone git@github.com:cornell-brg/hb-pytorch.git

- Create a [Python virtual environment][venv]:

      python3 -m venv ./venv_pytorch
      source ./venv_pytorch/bin/activate

- Install some dependencies:

      pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest ninja hypothesis

- Init PyTorch third party dependencies:

      git submodule update --init --recursive

- Setup building environment variables:

      source setup_emul_build_env.sh

- Build PyTorch. This step can take up to 15 minutes:

      python setup.py develop

[venv]: https://docs.python.org/3/tutorial/venv.html

### Run Pytests
 - Goto hb-pytorch directory
    `cd hb-pytorch/hammerblade/torch`
 - Run pytest
    `python pytest_runner.py`


### Important files and directories related to HammerBlade
#### files used to run pytest (adapted from Baseline)
  - `hammerblade/fragments/`
  - `hammerblade/environment.mk`
  - `baseline-README.md`
  - `run-hb-pytest.sh` (`source` this one to run pytest!)
  - `hammerblade/torch/`
#### HammerBlade device code
  - `hammerblade/torch/kernel`
#### Pytest tests
  - `hammerblade/torch/tests/`
#### files that interacts with HammerBlade CUDALite runtime
  - `c10/hammerblade/`

### How to implement a new kernel
 1. Register the kernel for HammerBlade with PyTorch by editing `aten/src/ATen/native/native_functions.yaml`
 ```diff
func: sigmoid(Tensor self) -> Tensor
use_c10_dispatcher: full
supports_named_tensor: True
variants: function, method
dispatch:
  CPU: sigmoid
  CUDA: sigmoid
+ HammerBlade: sigmoid
  MkldnnCPU: mkldnn_sigmoid
 ```
 2. Add host code to `aten/src/ATen/native/hammerblade/Sigmoid.cpp` Add the dummiest host code possible, without calling the kernel.
 3. Add tests to `hammerblade/torch/tests/test_sigmoid.py`
 4. With Emulation Layer, make sure the code compiles and tests fail only because of incorrect results
 5. Add kernel code to `hammerblade/torch/kernel/kernel_sigmoid.cpp`, which is also the dummiest code.
 6. Change the host code to be more realistic: call the kernel and do nothing else.
 7. Implement both the host and kernel code for real, assuming 1x1 tile group.
 8. Make sure everything pass on Emulation layer, and write more tests. Then you are ready to create a PR!
 9. Make sure your code works on COSIM.
 10. Optimizations, like parallelization etc.

 ### Kernel Development Tips
 1. Maintaining two clones, one for emulation and one for cosim (eg., `hb-pytorch/` and `hb-pytorch-cosim/`), eases
 the burden of cosim evaluation. This requires two separate pytorch environments as well (eg., `venv_pytorch` and `venv_pytorch_cosim`).

 2. Ideally, you would only ever need to run once, to debug an issue. Use `gdb` extensively with emulation.
```
$ gdb python
(gdb) b tensorlib_sigmoid
(gdb) r -m pytest test_sigmoid.py
```
Linking would become a bottleneck when running in tight loop. As a result, `gdb` could save a lot of time compared to printf debugging.

 3. Sometimes new cpp files are not taken into account by cmake. Since kernel authors would only ever need to add new files
 either to `aten/src/Aten/native` or `hammerblade/torch/` running following command might solve the failure:
```
touch aten/src/ATen/CMakeLists.txt # New host code sources
touch c10/hammerblade/CMakeLists.txt # New device code sources
```

### Native Profiling Tools
To enable native execution time profiling, edit `CMakeList.txt` and set `PROFILE_ATEN` to `ON`.

To enable unimplemented HammerBlade kernel discovery, edit `CMakeList.txt` and set `PROFILE_UNIMPL` to `ON`. Warning: `PROFILE_UNIMPL` should be disabled to get more accurate execution time profiling.

Region of interest (ROI) should be marked with `torch.aten_profiler_start()` and `torch.aten_profiler_end()`. For example
```python
import torch

torch.aten_profiler_start()
x = torch.randn(10)
y = x + x
torch.aten_profiler_end()
```
