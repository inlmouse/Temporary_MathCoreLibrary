// Due to a bug caused by C++/CLI and boost (used indirectly via caffe headers, not this one), 
// we have to seperate code related to boost from CLI compiling environment.
// This wrapper class serves for this purpose.
// See: http://stackoverflow.com/questions/8144630/mixed-mode-c-cli-dll-throws-exception-on-exit
//	and http://article.gmane.org/gmane.comp.lib.boost.user/44515/match=string+binding+invalid+mixed

#pragma once

#include <vector>
#pragma warning(push, 0)
#include <opencv2/imgproc/imgproc.hpp>
#include <boost\smart_ptr\shared_ptr.hpp>
#pragma warning(push, 0) 

//Declare an abstract Net class instead of including caffe headers, which include boost headers.
//The definition of Net is defined in cpp code, which does include caffe header files.
namespace caffe
{
	template <class DType>
	class Net;
	class Datum;
	template <class DType>
	class MemoryDataLayer;
}

struct FloatArray
{
	const float* Data;
	int Size;
	std::vector<int> Shape;
	FloatArray(const float* data, int size);
	FloatArray(const float* data, int size, std::vector<int> shape);
	//FloatArray(const float* data);
};

typedef std::vector<float> FloatVec;

class _CaffeModel
{
public:
	caffe::Net<float>* _net;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;

	void SetMean(const std::string& mean_file);
	boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_data_layer;

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);


	static void SetDevice(int device_id); //Use a negative number for CPU only
										  //caffe::Blob<float>* inputlayer = _net->input_blobs[0];
	_CaffeModel(const std::string &netFile, const std::string &modelFile, bool isencoded = false);
	_CaffeModel(const std::string &netFile, const std::string &modelFile, int device = 0, bool isencoded = false);
	~_CaffeModel();
	int GetInputImageWidth();
	int GetInputImageHeight();
	int GetInputImageChannels();
	int GetInputImageBatchsize();

	//REVIEW ktran: these APIs only make sense for images
	FloatArray ExtractBitmapOutputs(std::vector<std::string> &imageData, const std::string &layerName, int DeviceId);
	std::vector<FloatArray> ExtractBitmapOutputs(std::vector<std::string> &imageData, const std::vector<std::string> &layerNames, int DeviceId);
	FloatArray ExtractFileOutputs(std::vector<std::string> &imageFile, const std::string &layerName, int DeviceId);
	std::vector<FloatArray> ExtractFileOutputs(std::vector<std::string> &imageFile, const std::vector<std::string> &layerNames, int DeviceId);
	FloatArray ExtractVectorOutputs(std::vector<float> vectorData, const std::string &layerName, int DeviceId);
	std::vector<FloatArray> ExtractVectorOutputs(std::vector<float> vectorData, const std::vector<std::string> &layerNames, int DeviceId);
	//
	FloatArray ExtractMatOutputs(std::vector<cv::Mat> &image, const std::string &layerName, int DeviceId);
	std::vector<FloatArray> ExtractMatOutputs(std::vector<cv::Mat> &image, const std::vector<std::string> &layerNames, int DeviceId);
	// imageData needs to be of size channel*height*width as required by the "data" blob. 
	// The C++/CLI caller can use GetInputImageWidth()/Height/Channels to get the desired dimension.
	static bool Alignment(cv::Mat &Ori, std::vector<float>landmerks, cv::Mat &dstimg);
	static void Alignment(cv::Mat &Ori, std::vector<cv::Rect> rect_A, _CaffeModel IPBbox, _CaffeModel IPTs5, std::vector<cv::Mat> &F, FloatArray &headpose);
	static bool Train(std::string solverpath);

private:
	void EvaluateFile(caffe::Net<float>* net, std::vector<std::string> imageFile, int DeviceId);
	void EvaluateBitmap(caffe::Net<float>* net, std::vector<std::string> imageData, int DeviceId);
	void EvaluateVector(caffe::Net<float>* net, std::vector<float> vectorData, int DeviceId);
	void EvaluateMat(caffe::Net<float>* net, std::vector<cv::Mat> image, int DeviceId);
};