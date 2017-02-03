#include "_CaffeModel.h"

#pragma warning(push, 0) // disable warnings from the following external headers
#include <vector>
#include <string>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/util/signal_handler.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#pragma warning(push, 0) 

using namespace boost;
using namespace caffe;

const cv::Scalar_<float> BlackPixel(0, 0, 0);

FloatArray::FloatArray(const float* data, int size) : Data(data), Size(size) {}
FloatArray::FloatArray(const float* data, int size, std::vector<int> shape) : Data(data), Size(size), Shape(shape) {}

_CaffeModel::_CaffeModel(const string &netFile, const string &modelFile)
{
	SetDevice(0);
	_net = new Net<float>(netFile, Phase::TEST);
	_net->CopyTrainedLayersFrom(modelFile);
	memory_data_layer = static_pointer_cast<MemoryDataLayer<float>>(_net->layer_by_name("data"));
}

_CaffeModel::_CaffeModel(const std::string &netFile, const std::string &modelFile, int device = 0)
{
	SetDevice(device);
	_net = new Net<float>(netFile, Phase::TEST);
	_net->CopyTrainedLayersFrom(modelFile);
	memory_data_layer = static_pointer_cast<MemoryDataLayer<float>>(_net->layer_by_name("data"));
}


_CaffeModel::~_CaffeModel()
{
	if (_net)
	{
		delete _net;
		_net = nullptr;
	}
}


void _CaffeModel::SetMean(const std::string& mean_file)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void _CaffeModel::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = _net->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void _CaffeModel::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== _net->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

void _CaffeModel::SetDevice(int deviceId)
{
	// Set GPU
	if (deviceId >= 0)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(deviceId);
	}
	else
		Caffe::set_mode(Caffe::CPU);
}

int _CaffeModel::GetInputImageWidth()
{
	return memory_data_layer->width();
}

int _CaffeModel::GetInputImageHeight()
{
	return memory_data_layer->height();
}

int _CaffeModel::GetInputImageChannels()
{
	return memory_data_layer->channels();
}

int _CaffeModel::GetInputImageBatchsize()
{
	return memory_data_layer->batch_size();
}

//Priviate Functions:
void _CaffeModel::EvaluateFile(caffe::Net<float>* net, std::vector<std::string> imageFile, int DeviceId)
{
	_CaffeModel::SetDevice(DeviceId);
	int height = memory_data_layer->height();
	int width = memory_data_layer->width();
	std::vector<cv::Mat> input_mats;
	std::vector<int> labels;
	for (int i = 0; i < imageFile.size(); i++)
	{
		cv::Mat img = cv::imread(imageFile[i]);
		resize(img, img, cv::Size(), double(width) / img.cols, double(height) / img.rows, CV_INTER_CUBIC);
		input_mats.push_back(img);
		labels.push_back(0);
	}
	memory_data_layer->set_batch_size(input_mats.size());
	memory_data_layer->AddMatVector(input_mats, labels);
	float loss = 0.0;
	net->Forward(&loss);
}

void _CaffeModel::EvaluateBitmap(caffe::Net<float>* net, std::vector<std::string> imageData, int DeviceId)
{
	_CaffeModel::SetDevice(DeviceId);
	std::vector<Datum> datums;
	float loss = 0.0;
	for (int i = 0; i < imageData.size(); i++)
	{
		Datum datum;
		datum.set_channels(3);
		datum.set_height(memory_data_layer->height());
		datum.set_width(memory_data_layer->width());
		datum.set_label(0);
		datum.clear_data();
		datum.clear_float_data();
		datum.set_data(imageData[i]);
		datums.push_back(datum);
	}
	memory_data_layer->set_batch_size(datums.size());
	memory_data_layer->AddDatumVector(datums);
	net->Forward(&loss);
}

void _CaffeModel::EvaluateVector(caffe::Net<float>* net, std::vector<float> vectorData, int DeviceId)
{
	_CaffeModel::SetDevice(DeviceId);
	std::vector<Datum> datums;
	float loss = 0.0;
	int channels = memory_data_layer->channels();
	for (int i = 0; i < vectorData.size() / channels; i++)
	{
		Datum datum;
		datum.set_channels(channels);
		datum.set_height(memory_data_layer->height());
		datum.set_width(memory_data_layer->width());
		datum.set_label(0);
		datum.clear_data();
		datum.clear_float_data();
		for (int j = 0; j < channels; j++)
		{
			datum.add_float_data(vectorData[i*channels + j]);
		}
		datums.push_back(datum);
	}
	memory_data_layer->set_batch_size(datums.size());
	memory_data_layer->AddDatumVector(datums);
	net->Forward(&loss);
}

void _CaffeModel::EvaluateMat(caffe::Net<float>* net, std::vector<cv::Mat> image, int DeviceId)
{
	_CaffeModel::SetDevice(DeviceId);
	//memory_data_layer->Reset(NULL, NULL, 0);
	int height = memory_data_layer->height();
	int width = memory_data_layer->width();
	std::vector<int> labels;
	for (int i = 0; i < image.size(); i++)
	{
		labels.push_back(0);
	}
	memory_data_layer->set_batch_size(image.size());
	memory_data_layer->AddMatVector(image, labels);
	float loss = 0.0;

	net->Forward(&loss);
}

//APIs:
FloatArray _CaffeModel::ExtractBitmapOutputs(std::vector<std::string> &imageData, const string &layerName, int DeviceId)
{
	EvaluateBitmap(_net, imageData, DeviceId);
	auto blob = _net->blob_by_name(layerName);
	return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractBitmapOutputs(std::vector<std::string> &imageData, const vector<string> &layerNames, int DeviceId)
{
	EvaluateBitmap(_net, imageData, DeviceId);
	vector<FloatArray> results;
	for (auto& name : layerNames)
	{
		auto blob = _net->blob_by_name(name);
		results.push_back(FloatArray(blob->cpu_data(), blob->count()));
	}
	return results;
}

FloatArray _CaffeModel::ExtractFileOutputs(std::vector<std::string> &imageFile, const std::string &layerName, int DeviceId)
{
	EvaluateFile(_net, imageFile, DeviceId);
	auto blob = _net->blob_by_name(layerName);
	return FloatArray(blob->cpu_data(), blob->count());
}

std::vector<FloatArray> _CaffeModel::ExtractFileOutputs(std::vector<std::string> &imageFile, const std::vector<std::string> &layerNames, int DeviceId)
{
	EvaluateFile(_net, imageFile, DeviceId);
	vector<FloatArray> results;
	for (auto& name : layerNames)
	{
		auto blob = _net->blob_by_name(name);
		results.push_back(FloatArray(blob->cpu_data(), blob->count()));
	}
	return results;
}

FloatArray _CaffeModel::ExtractVectorOutputs(std::vector<float> vectorData, const std::string &layerName, int DeviceId)
{
	EvaluateVector(_net, vectorData, DeviceId);
	auto blob = _net->blob_by_name(layerName);
	return FloatArray(blob->cpu_data(), blob->count());
}

std::vector<FloatArray> _CaffeModel::ExtractVectorOutputs(std::vector<float> vectorData, const std::vector<std::string> &layerNames, int DeviceId)
{
	EvaluateVector(_net, vectorData, DeviceId);
	vector<FloatArray> results;
	for (auto& name : layerNames)
	{
		auto blob = _net->blob_by_name(name);
		results.push_back(FloatArray(blob->cpu_data(), blob->count()));
	}
	return results;
}

FloatArray _CaffeModel::ExtractMatOutputs(std::vector<cv::Mat> &image, const std::string &layerName, int DeviceId)
{
	EvaluateMat(_net, image, DeviceId);
	auto blob = _net->blob_by_name(layerName);
	return FloatArray(blob->cpu_data(), blob->count(), blob->shape());
}

std::vector<FloatArray> _CaffeModel::ExtractMatOutputs(std::vector<cv::Mat> &image, const std::vector<std::string> &layerNames, int DeviceId)
{
	EvaluateMat(_net, image, DeviceId);
	vector<FloatArray> results;
	for (auto& name : layerNames)
	{
		auto blob = _net->blob_by_name(name);
		results.push_back(FloatArray(blob->cpu_data(), blob->count(), blob->shape()));
	}
	return results;
}

//Helper functions:
bool _CaffeModel::Alignment(cv::Mat &Ori, std::vector<float>landmerks, cv::Mat &dstimg)
{
	cv::Mat gray;
	if (Ori.channels() == 3)
	{
		cvtColor(Ori, gray, CV_BGR2GRAY);
	}
	else if (Ori.channels() == 4)
	{
		cvtColor(Ori, gray, CV_BGRA2GRAY);
	}
	else if (Ori.channels() == 1)
	{
		gray = Ori;
	}
	else
	{
		return false;
		//TODO: waitting for check!
	}
	cv::equalizeHist(gray, gray);
	//**********
	int broder = Ori.cols;
	cv::Point2f eye_left, eye_right;
	eye_left.x = landmerks[0] * broder;
	eye_left.y = landmerks[1] * broder;
	eye_right.x = landmerks[2] * broder;
	eye_right.y = landmerks[3] * broder;

	cv::Point2f eyesCenter = cv::Point2f((eye_left.x + eye_right.x) * 0.5f, (eye_left.y + eye_right.y) * 0.5f);

	double dy = (eye_right.y - eye_left.y);
	double dx = (eye_right.x - eye_left.x);
	double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

	int iWidth = gray.cols;
	int iHeight = gray.rows;

	cv::Mat rot_mat = cv::getRotationMatrix2D(eyesCenter, angle, 1.0);
	cv::Mat rot(cv::Size(iWidth, iHeight), gray.type(), cv::Scalar::all(0));

	cv::warpAffine(gray, rot, rot_mat, gray.size());

	gray.release();
	//gray.~Mat();
	rot_mat.release();
	//rot_mat.~Mat();
	dstimg = rot.clone();
	rot.release();
	//rot.~Mat();
	return true;
}

//Train Funtion:
DEFINE_string(solver, "",
	"The solver definition protocol buffer text file.");
DEFINE_string(gpu, "",
	"Optional; run in GPU mode on given device IDs separated by ','."
	"Use '-gpu all' to run on all available GPUs. The effective training "
	"batch size is multiplied by the number of devices.");
DEFINE_string(snapshot, "",
	"Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
	"Optional; the pretrained weights to initialize finetuning, "
	"separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(sigint_effect, "stop",
	"Optional; action to take when a SIGINT signal is received: "
	"snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
	"Optional; action to take when a SIGHUP signal is received: "
	"snapshot, stop or none.");

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
	if (FLAGS_gpu == "all") {
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size()) {
		vector<string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) {
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
		}
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
	const std::string& flag_value) {
	if (flag_value == "stop") {
		return caffe::SolverAction::STOP;
	}
	if (flag_value == "snapshot") {
		return caffe::SolverAction::SNAPSHOT;
	}
	if (flag_value == "none") {
		return caffe::SolverAction::NONE;
	}
	LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
	return caffe::SolverAction::NONE;
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
	std::vector<std::string> model_names;
	boost::split(model_names, model_list, boost::is_any_of(","));
	for (int i = 0; i < model_names.size(); ++i) {
		LOG(INFO) << "Finetuning from " << model_names[i];
		solver->net()->CopyTrainedLayersFrom(model_names[i]);
		for (int j = 0; j < solver->test_nets().size(); ++j) {
			solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
		}
	}
}

bool _CaffeModel::Train(std::string solverpath)
{
	try
	{
		FLAGS_solver = const_cast<char*>(solverpath.c_str());
		CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
		CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
			<< "Give a snapshot to resume training or weights to finetune "
			"but not both.";
		caffe::SolverParameter solver_param;
		caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
		// If the gpus flag is not provided, allow the mode and device to be set
		// in the solver prototxt.
		if (FLAGS_gpu.size() == 0
			&& solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
			if (solver_param.has_device_id()) {
				FLAGS_gpu = "" +
					boost::lexical_cast<string>(solver_param.device_id());
			}
			else {
				FLAGS_gpu = "" + boost::lexical_cast<string>(0);
			}
		}
		vector<int> gpus;
		get_gpus(&gpus);
		if (gpus.size() == 0) {
			LOG(INFO) << "Use CPU.";
			Caffe::set_mode(Caffe::CPU);
		}
		else {
			ostringstream s;
			for (int i = 0; i < gpus.size(); ++i) {
				s << (i ? ", " : "") << gpus[i];
			}
			LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
			cudaDeviceProp device_prop;
			for (int i = 0; i < gpus.size(); ++i) {
				cudaGetDeviceProperties(&device_prop, gpus[i]);
				LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
			}
#endif
			solver_param.set_device_id(gpus[0]);
			Caffe::SetDevice(gpus[0]);
			Caffe::set_mode(Caffe::GPU);
			Caffe::set_solver_count(gpus.size());
		}
		caffe::SignalHandler signal_handler(
			GetRequestedAction(FLAGS_sigint_effect),
			GetRequestedAction(FLAGS_sighup_effect));
		shared_ptr<caffe::Solver<float> >
			solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
		solver->SetActionFunction(signal_handler.GetActionFunction());

		if (FLAGS_snapshot.size()) {
			LOG(INFO) << "Resuming from " << FLAGS_snapshot;
			solver->Restore(FLAGS_snapshot.c_str());
		}
		else if (FLAGS_weights.size()) {
			CopyLayers(solver.get(), FLAGS_weights);
		}

		if (gpus.size() > 1) {
			/*caffe::P2PSync<float> sync(solver, NULL, solver->param());
			sync.Run(gpus);*/
#ifdef USE_NCCL
			caffe::NCCL<float> nccl(solver);
			nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
			LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
		}
		else {
			LOG(INFO) << "Starting Optimization";
			solver->Solve();
		}
		LOG(INFO) << "Optimization Done.";
		return true;
	}
	catch (std::exception err)
	{
		std::cout << "Train Fail!" << std::endl;
		return false;
	}
}