#MCL(MathCoreLibrary)
This project is the .Net interface of MathCoreLibrary which is written by C++/CLI. In order to compile this project, you need set *Common language Runtime Support* in your Visual Studio.

**The current version is 0.8.9**
##Interfaces--CaffeModel

###Public Member Variables
| Type |   Name |   Description|
| :--------: | :--------:| :------: |
| static int |   DeviceCount |  The number of CUDA supported graphic cards in this computer |
| static string | Version |  Current version of MCL |
| static string | UpdateLog |  Historical update log |

###Public Member Functions
| Return Type|   Name | variable 1 | variable 2 | variable 3 |  Description|
| :--------: | :--------:| :------: |:------: |:------: |:------: |
| **Constructor** |CaffeModel|string *netFile*|string *modelFile*|--|*Constructor Function*, inits a CaffeModel from local \*.prototxt net defined file and \*.caffemodel weights file. |
|**Constructor**|CaffeModel|string *netFile*|string *modelFile*|int *deviceId*|*Constructor Function*, inits a CaffeModel from local \*.prototxt net defined file and \*.caffemodel weights file. Point out which device to load this model at the same time.|
|**Destructor** |Dispose|--|--|--|Destruct this model.|
|bool|Train (static)|string *Solverpath*|--|--|Train the net that the solver defined. Return true if the training is finished and succeed.|
|float|CosineDistanceProb (static)|float[] *feature1*| float[] *feature2*|--|Return the Cosine Distance between the vector *feature1* and *feature2*. |
|bool |Alignment (static)|Bitmap *imgData*| float[] *landmark*| Bitmap *dstImg*|Align the *imgData* to *dstImg* accroding the given *landmark*. Return true if successful.|
|bool[] |Alignment (static)|Bitmap[] *imgDatas*| float[] *landmarks*| Bitmap[] *dstImgs*|Align the *imgDatas* to *dstImgs* accroding the given *landmarks*. Return true if successful.|
|Bitmap[]| Align_Step1(static)| Bitmap[] *B* |Rectangle[] *MarginRect*|float[] *bbox, headpose*|The first step of Face Alignment.|
|Bitmap[]| Align_Step2(static)| Bitmap[] *B, C* , float[] *ipts5*|Rectangle[] *MarginRect*|int *Width, Height*|The second step of Face Alignment.|
|int | GetInputImageWidth /GetInputImageHeight /GetInputImageChannels /GetInputImageBatchsize| --|--|--|Get the Shape of the input MomoryData Layer.|
|float[] |ExtractBitmapOutputs |Bitmap[] *imgDatas*| string *layerName*| int *DeviceId*|Extract the features of *imgDatas* in *layerName* on device *DeviceId*|
|float[][] |ExtractBitmapOutputs |Bitmap[] *imgDatas*| string[] *layerNames*| int *DeviceId*|Extract the features of *imgDatas* in *layerNames* on device *DeviceId*|
|float[] |ExtractFileOutputs |Bitmap[] *imageFiles*| string *layerName*| int *DeviceId*|Extract the features of Images which located at *imageFiles* in *layerName* on device *DeviceId*|
|float[][] |ExtractFileOutputs |Bitmap[] *imageFiles*| string[] *layerNames*| int *DeviceId*|Extract the features of Images which located at *imageFiless* in *layerName* on device *DeviceId*|
|float[] |ExtractVectorOutputs |float[] *vectorData*| string *layerName*| int *DeviceId*|Extract the features of *vectorData* in *layerName* on device *DeviceId*|
|float[][] |ExtractVectorOutputs |float[]*vectorDatas*| string[] *layerNames*| int *DeviceId*|Extract the features of *vectorDatas* in *layerNames* on device *DeviceId*|

##Update Log

- Version0.5.1: 全面支持CUDA8.0和CUDNNv5；同时.NetFramework升级到4.5.2，VC++版本提升至v140；舍弃对OpenCV2.4的支持

- Version0.5.2: 实现弱鸡C++版本的MTCNN，但是在CUDA8.0上有未知原因bug

2016-12-1 09:52:23
- Version0.7.1: 添加训练模型的接口；实现更高自由度的API；支持大于1batchsize的并行

2016-12-28 11:05:24
- Version0.8.2: 全面使用MemoryDataLayer，支持全动态batchsize，预处理部分交还Caffe，使其更规范，重构工程，因而接口参数有删减；同时解决了从路径直接读取图像和从内存获取图像前向传播结果不同的问题的一部分，在使用CLI/.Net转码的时候和使用OpenCV 的情况下，在RGB24/RGBA32/Gray8之间转换的结果存在误差，然而对最后结果的影响并不是很大

2017-02-01 22:33:46
- Version0.8.5: 添加NVIDIA NCCL多GPU并行通讯支持，支持CUDNN5.1

2017-02-03 23:26:18
- Version0.8.6: 添加Python Layer支持

2017-02-12 
- Version0.8.7: 回滚NCCL问题，现阶段仍使用P2P Access

2017-02-17
- Version0.8.8: 支持简单加密prototxt，对称密钥加密**待替换**

2017-02-22
- Version0.8.9: 添加两步精细对齐方法，现阶段兼容原模型