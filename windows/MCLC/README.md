#MCLC(MathCoreLibraryCpp)
This project is the native C++ interface of MathCoreLibrary which is written in C++11. In order to compile this project, you need C++11 support compiler. Noticed that the project is a extended of NameSpace *caffe*, *libcaffe.lib* and other caffe/boost header files are required.

##Interfaces--MCLC
###Public Data Structure

    struct DataBlob {
    const float* data;
    std::vector<int> size;
    std::string name;};
  
###Public Member Functions
| Return Type|   Name | variable 1 | variable 2 | variable 3 |  Description|
| :--------: | :--------:| :------: |:------: |:------: |:------: |
| **Constructor** |MCLC|--|--|--|*Constructor Function*, just config google logging. |
|int |AddNet|std::string *prototxt_path*| std::string *weights_path*| int *gpu_id = 0*|Inits a MCLC from local *prototxt_path* net defined file and *weights_path* weights file. Point out which device(*gpu_id*) to load this model at the same time. Return a automatic generated NetID.|
|std::unordered_map <std::string, DataBlob> |Forward|std::vector<cv::Mat>& *input_image*| int *net_id*|--|Evaluate the features of *input_image* in *net_id*, and return all you need.|
