# MathCoreLibrary
A math computational interface of MathCoreLibrary based on Windows

##Pre-request
All the environment and some local dependencies relied of projects in this repository which is able to find in [here](https://github.com/CompileSense/caffe_windows_binary). **Please check before you start to config or use**.

## Windows Setup
**Requirements**: Visual Studio 2015 update 1, CUDA 8.0, third party libraries: https://pan.baidu.com/s/1jHJCsK2 , password: 5d1g. Please extract the archive into `./windows/thirdparty/`.

### Pre-Build Steps
Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`

By defaults Windows build requires `CUDA` and `cuDNN` libraries.
Both can be disabled by adjusting build variables in `.\windows\CommonSettings.props`.
Python support is disabled by default, but can be enabled via `.\windows\CommonSettings.props` as well.
3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 8.0` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).
If you don't have CUDA installed, you can experiment with CPU_ONLY build.
In `.\windows\CommonSettings.props` set `CpuOnlyBuild` to `true` and set `UseCuDNN` to `false`.

### cuDNN
Download `cuDNN v5.1` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to %CUDA_PATH% (environment variable set by CUDA installer).
Alternatively, you can unpack zip to any location and set `CuDnnPath` to point to this location in `.\windows\CommonSettings.props`.
`CuDnnPath` defined in `.\windows\CommonSettings.props`.
Also, you can disable cuDNN by setting `UseCuDNN` to `false` in the property file.


### Build
Now, you should be able to build `.\windows\Caffe.sln`

##Component
###MCL
This project is the .Net interface of MathCoreLibrary which is written by C++/CLI. In order to compile this project, you need set *Common language Runtime Support* in your Visual Studio.

For more details, please visit [here](https://github.com/CompileSense/Temporary_MathCoreLibrary/tree/master/windows/MCL).

###MCLC
This project is the native C++ interface of MathCoreLibrary which is written in C++11. In order to compile this project, you need C++11 support compiler. Noticed that the project is a extended of NameSpace *caffe*, *libcaffe.lib* and other caffe/boost header files are required.

For more details, please visit [here](https://github.com/CompileSense/Temporary_MathCoreLibrary/tree/master/windows/MCLC).


