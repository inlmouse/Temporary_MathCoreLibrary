#include "pva_face_detector.h"
#include <cmath>
#include <ctime>
#include <caffe/util/nms.hpp>

PVAFaceDetector::PVAFaceDetector(float conf_thresh, float nms_thresh, size_t scale, size_t scale_multiple_of, size_t max_size)
	: conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), scale_(scale), scale_multiple_of_(scale_multiple_of), max_size_(max_size) {
	LoadModel();
}

void PVAFaceDetector::LoadModel() {
	const string prototxtPath = "C:\\DLFramework\\Temporary_MathCoreLibrary\\windows\\MTCNN\\model\\det1-memory.prototxt";
	const string modelPath = "C:\\DLFramework\\Temporary_MathCoreLibrary\\windows\\MTCNN\\model\\det1.caffemodel";
	Caffe::set_root_solver(true);
	//net_ = unique_ptr<Net<float>>(new Net<float>(prototxtPath, caffe::TEST));
	net_ = new Net<float>(prototxtPath, Phase::TEST, false);
	net_->CopyTrainedLayersFrom(modelPath);
}

void PVAFaceDetector::SetGpu(int gpu_id) {
	gpu_id_ = gpu_id;
	if (gpu_id >= 0) {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_id);
		//cv::gpu::setDevice(gpu_id);
	}
	else {
		Caffe::set_mode(Caffe::CPU);
	}
}

vector<vector<float>> PVAFaceDetector::Run(const cv::Mat &img) {
	// determine scale
	auto time1 = clock();
	int total_width = img.cols;
	int total_height = img.rows;
	int shorter_side = min(total_width, total_height);
	int longer_side = max(total_width, total_height);
	float scale = static_cast<float>(scale_) / shorter_side;
	if (longer_side * scale > max_size_) scale = (float)max_size_ / longer_side;
	// make input_width, input_height multiples of "scale_multiple_of_"
	int input_width = floor(img.cols * scale / scale_multiple_of_) * scale_multiple_of_;
	int input_height = floor(img.rows * scale / scale_multiple_of_) * scale_multiple_of_;
	float scale_x = static_cast<float>(input_width) / img.cols;
	float scale_y = static_cast<float>(input_height) / img.rows;

	// image pre-process
	auto data_blob = net_->blob_by_name("data");
	data_blob->Reshape(1, 3, input_height, input_width);
	float *data_buf = data_blob->mutable_cpu_data();
	cv::Mat tmp_img;
	img.convertTo(tmp_img, CV_32FC3);
	cv::Mat splitImg[3];
	/*if (gpu_id_ >= 0) {
	cv::gpu::GpuMat gpuImg(tmp_img);
	// split channels
	cv::gpu::GpuMat *splitGpuImg = new cv::gpu::GpuMat[3];
	cv::gpu::split(gpuImg, splitGpuImg);
	for (int i = 0; i < 3; i++) {
	// subtract pixel means
	cv::gpu::subtract(splitGpuImg[i], pixel_means_[i], splitGpuImg[i]);
	// resize
	cv::gpu::GpuMat gpuTmp;
	cv::gpu::resize(splitGpuImg[i], gpuTmp, cv::Size(input_width, input_height));
	// download image
	gpuTmp.download(splitImg[i]);
	}
	} else {*/
	cv::split(tmp_img, splitImg);
	for (int i = 0; i < 3; i++) {
		// subtract pixel means
		cv::subtract(splitImg[i], pixel_means_[i], splitImg[i]);
		// resize
		cv::resize(splitImg[i], splitImg[i], cv::Size(input_width, input_height));
	}
	//}

	// feed image data
	for (int i = 0; i < 3; i++) {
		if (splitImg[i].isContinuous()) {
			memcpy(data_buf, splitImg[i].ptr<float>(0), input_width * input_height * sizeof(float));
		}
		else {
			for (int j = 0; j < input_height; j++)
				memcpy(data_buf + j * input_height, splitImg[i].ptr<float>(j), input_width * sizeof(float));
		}
		data_buf += input_width * input_height;
	}
	// feed im_info
	float *im_info = net_->blob_by_name("im_info")->mutable_cpu_data();
	float p_im_info[6]{ static_cast<float>(input_height), static_cast<float>(input_width), scale_x, scale_y, scale_x, scale_y };
	memcpy(im_info, p_im_info, 6 * sizeof(float));

	auto time2 = clock();
	// net forward
	auto out_blobs = net_->Forward();
	auto time3 = clock();

	// harvest result
	auto prob_blob = out_blobs[1];
	int roi_count = prob_blob->shape(0);
	const float *prob = prob_blob->cpu_data();
	const float *bbox_pred = out_blobs[0]->cpu_data();
	const float *rois = net_->blob_by_name("rois")->cpu_data();

	// post-process
	vector<vector<float>> dets;
	for (int i = 0; i < roi_count; i++) {
		// unscale bbox back to raw image space
		vector<float> bbox{ rois[5 * i + 1] / scale_x, rois[5 * i + 2] / scale_y, rois[5 * i + 3] / scale_x, rois[5 * i + 4] / scale_y };
		// apply bbox regression + clip
		bbox = BBoxTransformInv(bbox, vector<float>{bbox_pred + 8 * i + 4, bbox_pred + 8 * i + 8}, total_width, total_height);

		// build dets
		dets.push_back(vector<float>{bbox[0], bbox[1], bbox[2], bbox[3], prob[2 * i + 1]});
		//dets[5*i] = bbox[0]; dets[5*i+1] = bbox[1]; dets[5*i+2] = bbox[2]; dets[5*i+3] = bbox[3]; dets[5*i+4] = prob[2*i+1];
	}
	// sort dets, nms
	sort(dets.begin(), dets.end(), [](vector<float> x, vector<float> y) { return x[4] > y[4]; });
	unique_ptr<float[]> sorted_dets(new float[5 * dets.size()]);
	for (size_t i = 0; i < dets.size(); i++)
		for (size_t j = 0; j < 5; j++) sorted_dets[5 * i + j] = dets[i][j];
	unique_ptr<int[]> keep_inds(new int[roi_count]);
	int num_nms_out;
	nms_cpu(roi_count, sorted_dets.get(), keep_inds.get(), &num_nms_out, 0, nms_thresh_, roi_count);
	// apply conf_thresh, store keep_inds to result
	vector<vector<float>> results;
	for (int i = 0; i < num_nms_out; i++) {
		int idx = keep_inds[i];
		if (dets[idx][4] > conf_thresh_) results.push_back(dets[idx]);
	}
	auto time4 = clock();

	cout << "pre-process  " << static_cast<float>(time2 - time1) / CLOCKS_PER_SEC << "s" << endl;
	cout << "net forward  " << static_cast<float>(time3 - time2) / CLOCKS_PER_SEC << "s" << endl;
	cout << "post-process " << static_cast<float>(time4 - time3) / CLOCKS_PER_SEC << "s" << endl;

	return results;
}

vector<float> PVAFaceDetector::BBoxTransformInv(const vector<float> &bbox, const vector<float> &delta, int total_width, int total_height) {
	float width = bbox[2] - bbox[0] + 1;
	float height = bbox[3] - bbox[1] + 1;
	float center_x = bbox[0] + width / 2;
	float center_y = bbox[1] + height / 2;

	float new_width = width * exp(delta[2]);
	float new_height = height * exp(delta[3]);
	float new_center_x = center_x + width * delta[0];
	float new_center_y = center_y + height * delta[1];

	float x1 = max(new_center_x - new_width / 2, 0.f);
	float x2 = min(new_center_x + new_width / 2, static_cast<float>(total_width - 1));
	float y1 = max(new_center_y - new_height / 2, 0.f);
	float y2 = min(new_center_y + new_height / 2, static_cast<float>(total_height - 1));

	return vector<float>{x1, y1, x2, y2};
}

