#MCL(MathCoreLibrary)
This project is the .Net interface of MathCoreLibrary which is written by C++/CLI. In order to compile this project, you need set *Common language Runtime Support* in your Visual Studio.

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
|int | GetInputImageWidth /GetInputImageHeight /GetInputImageChannels /GetInputImageBatchsize| --|--|--|Get the Shape of the input MomoryData Layer.|
|float[] |ExtractBitmapOutputs |Bitmap[] *imgDatas*| string *layerName*| int *DeviceId*|Extract the features of *imgDatas* in *layerName* on device *DeviceId*|
|float[][] |ExtractBitmapOutputs |Bitmap[] *imgDatas*| string[] *layerNames*| int *DeviceId*|Extract the features of *imgDatas* in *layerNames* on device *DeviceId*|
|float[] |ExtractFileOutputs |Bitmap[] *imageFiles*| string *layerName*| int *DeviceId*|Extract the features of Images which located at *imageFiles* in *layerName* on device *DeviceId*|
|float[][] |ExtractFileOutputs |Bitmap[] *imageFiles*| string[] *layerNames*| int *DeviceId*|Extract the features of Images which located at *imageFiless* in *layerName* on device *DeviceId*|
|float[] |ExtractVectorOutputs |float[] *vectorData*| string *layerName*| int *DeviceId*|Extract the features of *vectorData* in *layerName* on device *DeviceId*|
|float[][] |ExtractVectorOutputs |float[]*vectorDatas*| string[] *layerNames*| int *DeviceId*|Extract the features of *vectorDatas* in *layerNames* on device *DeviceId*|
