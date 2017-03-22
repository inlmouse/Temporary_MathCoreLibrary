#include <chrono>
#include <cstdlib>
#include <memory>

#include "boost/make_shared.hpp"
#include "FaceDetection.inc.h"

//caffe::MCLC* mclc = new caffe::MCLC();

using namespace FaceInception;

int main(int argc, char* argv[])
{
	//CascadeCNN cascade("G:\\WIDER\\face_detection\\bak3\\cascade_12_memory_nobn1.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade12-_iter_490000.caffemodel",
	//                   "G:\\WIDER\\face_detection\\bak3\\cascade_24_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade24-_iter_145000.caffemodel",
	//                   "G:\\WIDER\\face_detection\\bak3\\cascade_48_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade48-_iter_225000.caffemodel");
	string model_folder = "C:\\DLFramework\\Temporary_MathCoreLibrary\\windows\\MTCNN\\model\\";
	CascadeCNN cascade(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
		model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
		model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
		model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
		model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
		0);
	//CaptureDemo(cascade);

	double min_face_size = 12;

	//ScanList("H:\\lfw\\list.txt", cascade);
	Mat image = imread("C:\\Users\\BALTHASAR\\Desktop\\2883_1.jpg");

	//Mat image = imread("G:\\WIDER\\face_detection\\pack\\1[00_00_26][20160819-181452-0].BMP");
	//Mat image = imread("D:\\face project\\FDDB\\2002/07/25/big/img_1047.jpg");

	//Mat image = imread("D:\\face project\\FDDB\\2003/01/13/big/img_1087.bmp");
	cout << image.cols << "," << image.rows << endl;
	vector<vector<Point2d>> points;
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	auto result = cascade.GetDetection(image, 12.0 / min_face_size, 0.7, true, 0.7, true, points);
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;

	cout << "===========================================================" << endl;
	points.clear();//The first run is slow because it need to allocate memory.
	p0 = std::chrono::system_clock::now();
	result = cascade.GetDetection(image, 12.0 / min_face_size, 0.7, true, 0.7, true, points);
	p1 = std::chrono::system_clock::now();
	cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;

	for (int i = 0; i < result.size(); i++) {
		//cout << "face box:" << result[i].first << " confidence:" << result[i].second << endl;
		rectangle(image, result[i].first, Scalar(255, 0, 0), 2);
		if (points.size() >= i + 1) {
			for (int p = 0; p < 5; p++) {
				circle(image, points[i][p], 2, Scalar(0, 255, 255), -1);
			}
		}
	}
	/*while (image.cols > 1000) {
		resize(image, image, Size(0, 0), 0.75, 0.75);
	}*/
	/*imshow("final", image);
	waitKey(0);*/
	imwrite("C:\\Users\\BALTHASAR\\Desktop\\output.jpg", image);
	//TestFDDBPrecision(cascade, "D:\\face project\\FDDB\\", true, true);
	//delete mclc;
	system("pause");
	return 0;
}