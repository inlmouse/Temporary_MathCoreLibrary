#pragma once
#ifndef PVA_FACE_DETECTOR_H_
#define PVA_FACE_DETECTOR_H_

#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

class PVAFaceDetector {
public:
	PVAFaceDetector(float conf_thresh = 0.8, float nms_thresh = 0.3, size_t scale = 640, size_t scale_multiple_of_ = 32, size_t max_size = 2000);
	virtual ~PVAFaceDetector() = default;

	void SetGpu(int gpu_id);
	vector<vector<float>> Run(const cv::Mat &img);

private:
	void LoadModel();
	vector<float> BBoxTransformInv(const vector<float> &bbox, const vector<float> &delta, int total_width, int total_height);

	//unique_ptr<Net<float>> net_;
	Net<float>* net_;

	float conf_thresh_, nms_thresh_;
	size_t scale_, scale_multiple_of_, max_size_;
	int gpu_id_ = 0;
	vector<cv::Scalar> pixel_means_{ 102.9801, 115.9465, 122.7717 };

};


#endif


