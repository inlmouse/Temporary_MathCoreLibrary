#include <iostream>
#include "pva_face_detector.h"

using namespace std;

int main() {
	PVAFaceDetector detector;
	detector.SetGpu(0);

	cv::Mat testImage = cv::imread("D:\\Research\\face_detection\\pva-faster-rcnn\\demo_cpp\\test1.jpg");
	detector.Run(testImage);
	detector.Run(testImage);
	vector<vector<float>> results = detector.Run(testImage);

	// print
	cout << results.size() << endl;
	for (size_t i = 0; i < results.size(); i++) {
		for (size_t j = 0; j < results[i].size(); j++) cout << results[i][j] << " ";
		cout << endl;
	}

	cout << "Done." << endl;
}

