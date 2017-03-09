#pragma once
#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//using namespace std;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(float));

#define Version "0.8.10"
#define UpdateLog "\
Version0.5.1: 全面支持CUDA8.0和CUDNNv5；同时.NetFramework升级到4.5.2，VC++版本提升至v140；舍弃对OpenCV2.4的支持\n\
Version0.5.2: 实现弱鸡C++版本的MTCNN，但是在CUDA8.0上有未知原因bug\n\
Version0.7.1: 添加训练模型的接口；实现更高自由度的API；支持大于1batchsize的并行\n\
Version0.8.2: 全面使用MemoryDataLayer，支持全动态batchsize，预处理部分交还Caffe，因而接口参数有删减；同时解决了从路径直接读取图像和从内存获取图像前向传播结果不同的问题\n\
Version0.8.5: 添加NVIDIA NCCL多GPU并行通讯支持，支持CUDNN5.1\n\
Version0.8.6: 添加Python Layer支持\n\
Version0.8.7: 修复NCCL问题，现阶段仍使用P2P Access\n\
Version0.8.8: 支持加密prototxt\n\
Version0.8.9: 添加两步精细对齐函数，现阶段兼容原模型\n\
Version0.8.10: 添加Byte数组格式图像输入支持"

namespace CaffeSharp {
	
	public ref class CaffeModel
	{
	private:
		_CaffeModel *m_net;
		static void SetDevice(int deviceId)
		{
			_CaffeModel::SetDevice(deviceId);
		}

	public:
		
		static int DeviceCount;
		static const  System::String^ version = Version;
		static const  System::String^ updatelog = UpdateLog;
		static CaffeModel()
		{
#ifndef CPU_ONLY
			int count;
			cudaGetDeviceCount(&count);
			DeviceCount = count;
#else
			DeviceCount = 0;
#endif
		}

		CaffeModel(String ^netFile, String ^modelFile)
		{
			m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile), false);
		}

		CaffeModel(String ^netFile, String ^modelFile, int deviceId)
		{
			m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile), deviceId, false);
		}

		CaffeModel(String ^netFile, String ^modelFile, bool isencoded)
		{
			m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile), isencoded);
		}

		CaffeModel(String ^netFile, String ^modelFile, int deviceId, bool isencoded)
		{
			m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile), deviceId, isencoded);
		}

		// destructor to call finalizer
		~CaffeModel()
		{
			this->!CaffeModel();
		}

		// finalizer to release unmanaged resource
		!CaffeModel()
		{
			delete m_net;
			m_net = NULL;
		}

		void GetUsefulPart(String ^savepath)
		{
			m_net->GetUsefulPart(TO_NATIVE_STRING(savepath));
		}

		int GetInputImageWidth()
		{
			return m_net->GetInputImageWidth();
		}
		int GetInputImageHeight()
		{
			return m_net->GetInputImageHeight();
		}
		int GetInputImageChannels()
		{
			return m_net->GetInputImageChannels();
		}
		int GetInputImageBatchsize()
		{
			return m_net->GetInputImageBatchsize();
		}

		array<float>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas, String^ layerName, int DeviceId)
		{
			std::vector<std::string> datum_strings;
			for (int i = 0; i < imgDatas->Length; i++)
			{
				datum_strings.push_back(ConvertToDatum(imgDatas[i]));
			}

			FloatArray intermediate = m_net->ExtractBitmapOutputs(datum_strings, TO_NATIVE_STRING(layerName), DeviceId);
			MARSHAL_ARRAY(intermediate, outputs)
				return outputs;
		}

		array<array<float>^>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas, array<String^>^ layerNames, int DeviceId)
		{
			std::vector<std::string> datum_strings;
			for (int i = 0; i < imgDatas->Length; i++)
			{
				datum_strings.push_back(ConvertToDatum(imgDatas[i]));
			}

			std::vector<std::string> names;
			for each(String^ name in layerNames)
				names.push_back(TO_NATIVE_STRING(name));
			std::vector<FloatArray> intermediates = m_net->ExtractBitmapOutputs(datum_strings, names, DeviceId);
			auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
			for (int i = 0; i < names.size(); ++i)
			{
				auto intermediate = intermediates[i];
				MARSHAL_ARRAY(intermediate, values)
					outputs[i] = values;
			}
			return outputs;
		}

		array<float>^ ExtractByteOutputs(array<array<Byte>^>^ byteDatas, String^ layerName, int DeviceId)
		{
			std::vector<std::string> datum_strings;
			for (int i = 0; i < byteDatas->Length; i++)
			{
				datum_strings.push_back(ConvertToDatum(byteDatas[i]));
			}

			FloatArray intermediate = m_net->ExtractByteOutputs(datum_strings, TO_NATIVE_STRING(layerName), DeviceId);
			MARSHAL_ARRAY(intermediate, outputs)
				return outputs;
		}

		array<array<float>^>^ ExtractByteOutputs(array<array<Byte>^>^ byteDatas, array<String^>^ layerNames, int DeviceId)
		{
			std::vector<std::string> datum_strings;
			for (int i = 0; i < byteDatas->Length; i++)
			{
				datum_strings.push_back(ConvertToDatum(byteDatas[i]));
			}

			std::vector<std::string> names;
			for each(String^ name in layerNames)
				names.push_back(TO_NATIVE_STRING(name));
			std::vector<FloatArray> intermediates = m_net->ExtractByteOutputs(datum_strings, names, DeviceId);
			auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
			for (int i = 0; i < names.size(); ++i)
			{
				auto intermediate = intermediates[i];
				MARSHAL_ARRAY(intermediate, values)
					outputs[i] = values;
			}
			return outputs;
		}

		array<float>^ ExtractFileOutputs(array<String^>^ imageFiles, String^ layerName, int DeviceId)
		{
			std::vector<std::string> file_strings;
			for each(String^ imageFile in imageFiles)
				file_strings.push_back(TO_NATIVE_STRING(imageFile));
			FloatArray intermediate = m_net->ExtractFileOutputs(file_strings, TO_NATIVE_STRING(layerName), DeviceId);
			MARSHAL_ARRAY(intermediate, outputs)
				return outputs;
		}

		array<array<float>^>^ ExtractFileOutputs(array<String^>^ imageFiles, array<String^>^ layerNames, int DeviceId)
		{
			std::vector<std::string> file_strings;
			for each(String^ imageFile in imageFiles)
				file_strings.push_back(TO_NATIVE_STRING(imageFile));

			std::vector<std::string> names;
			for each(String^ name in layerNames)
				names.push_back(TO_NATIVE_STRING(name));

			std::vector<FloatArray> intermediates = m_net->ExtractFileOutputs(file_strings, names, DeviceId);
			auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
			for (int i = 0; i < names.size(); ++i)
			{
				auto intermediate = intermediates[i];
				MARSHAL_ARRAY(intermediate, values)
					outputs[i] = values;
			}
			return outputs;
		}

		array<float>^ ExtractVectorOutputs(array<float>^ vectorData, String^ layerName, int DeviceId)
		{
			int channels = GetInputImageChannels();
			if (vectorData->Length%channels != 0)
			{
				return gcnew array<float>(0);
			}
			else
			{
				std::vector<float> inputdata;
				for (int i = 0; i < vectorData->Length; i++)
				{
					float x = vectorData[i];
					inputdata.push_back(x);
				}
				FloatArray intermediate = m_net->ExtractVectorOutputs(inputdata, TO_NATIVE_STRING(layerName), DeviceId);
				MARSHAL_ARRAY(intermediate, outputs)
					return outputs;
			}
		}

		array<array<float>^>^ ExtractVectorOutputs(array<float>^ vectorDatas, array<String^>^ layerNames, int DeviceId)
		{
			int channels = GetInputImageChannels();
			if (vectorDatas->Length%channels != 0)
			{
				return gcnew array<array<float>^>(0);
			}
			else
			{
				std::vector<float> inputdata;
				for (int i = 0; i < vectorDatas->Length; i++)
				{
					float x = vectorDatas[i];
					inputdata.push_back(x);
				}
				std::vector<std::string> names;
				for each(String^ name in layerNames)
					names.push_back(TO_NATIVE_STRING(name));

				std::vector<FloatArray> intermediates = m_net->ExtractVectorOutputs(inputdata, names, DeviceId);
				auto outputs = gcnew array<array<float>^>(static_cast<int>(names.size()));
				for (int i = 0; i < names.size(); ++i)
				{
					auto intermediate = intermediates[i];
					MARSHAL_ARRAY(intermediate, values)
						outputs[i] = values;
				}
				return outputs;
			}
		}

		static bool Alignment(Bitmap^ imgData, array<float>^landmark, Bitmap^ dstImg)
		{
			cv::Mat mat, output;
			int a = ConvertBitmapToMat(imgData, mat);
			if (a == 0)
			{
				std::vector<float> landmarks;
				for each(float lm in landmark)
					landmarks.push_back(lm);
				if (_CaffeModel::Alignment(mat, landmarks, output))
				{
					CopyMatToBitmap(output, dstImg);
					mat.release();
					output.release();
					return true;
				}
				else
				{
					mat.release();
					output.release();
					return false;
				}

			}
			else
			{
				mat.release();
				output.release();
				return false;
			}
		}

		static array<bool>^ Alignment(array<Bitmap^>^ imgDatas, array<float>^landmarks, array<Bitmap^>^ dstImgs)
		{
			array<bool>^ results = gcnew array<bool>(imgDatas->Length);
			if (landmarks->Length == imgDatas->Length * 10)
			{
				std::vector<cv::Mat> mats(imgDatas->Length), outputs(imgDatas->Length);
				for (int i = 0; i < imgDatas->Length; i++)
				{
					int a = ConvertBitmapToMat(imgDatas[i], mats[i]);
					if (a == 0)
					{
						std::vector<float> landmark;
						for (int j = 0; j < 10; j++)
						{
							//landmark.push_back(landmarks[10 * i + j]);//maybe crash!
							float temp = landmarks[10 * i + j];
							landmark.push_back(temp);
						}
						if (_CaffeModel::Alignment(mats[i], landmark, outputs[i]))
						{
							CopyMatToBitmap(outputs[i], dstImgs[i]);
							results[i] = true;
						}
						else
						{
							dstImgs[i] = imgDatas[i];
							results[i] = false;
						}
					}
					else
					{
						dstImgs[i] = imgDatas[i];
						results[i] = false;
					}
					mats[i].release();
					outputs[i].release();
				}
			}
			else
			{
				for (size_t i = 0; i < imgDatas->Length; i++)
				{
					dstImgs[i] = imgDatas[i];
				}
				for each (bool result in results)
				{
					result = false;
				}
			}
			return results;
		}

		static array<Bitmap^>^ Align_Step1(array<Bitmap^>^ B, array<Drawing::Rectangle>^ MarginRect, array<float>^ bbox, array<float>^ headpose)
		{
			std::vector<cv::Mat> _B(B->Length);
			std::vector<float> _bbox(bbox->Length);
			std::vector<float> _headpose(headpose->Length);
			std::vector<cv::Rect2i> _MarginRect;
			for (int i = 0; i < B->Length; i++)
			{
				ConvertBitmapToMat(B[i], _B[i]);
				/*cv::imwrite("test_B.jpg", _B[0]);
				B[i]->Save("testB.jpg");*/
				for (size_t j = 0; j < 4; j++)
				{
					_bbox[4 * i + j] = bbox[4 * i + j];
				}
				for (size_t j = 0; j < 3; j++)
				{
					_headpose[3 * i + j] = headpose[3 * i + j];
				}
			}
			//Console::WriteLine("before native");
			std::vector<cv::Mat> _C = _CaffeModel::AlignStep1(_B, _bbox, _headpose, _MarginRect);
			//Console::WriteLine("after native");
			//cv::imwrite( "test_C.jpg", _C[0]);
			/*MarginRect = gcnew array<Drawing::Rectangle>(B->Length);*/
			array<Bitmap^>^ C = gcnew array<Bitmap^>(B->Length);
			for (size_t i = 0; i < B->Length; i++)
			{
				C[i] = gcnew System::Drawing::Bitmap(
					_C[i].cols, _C[i].rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				B[i] = gcnew System::Drawing::Bitmap(
					_B[i].cols, _B[i].rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				if (_C[i].channels()==4)
				{
					cvtColor(_C[i], _C[i], CV_BGRA2BGR);
				}
				if (_B[i].channels()==4)
				{
					cvtColor(_B[i], _B[i], CV_BGRA2BGR);
				}
				//Console::WriteLine("defore trans");
				CopyMatToBitmap(_C[i], C[i]);
				CopyMatToBitmap(_B[i], B[i]);
				//Console::WriteLine("after trans");
				MarginRect[i] = Drawing::Rectangle(_MarginRect[i].x, _MarginRect[i].y, _MarginRect[i].width, _MarginRect[i].height);
				_B[i].release();
				_C[i].release();
			}
			return C;
		}

		static array<Bitmap^>^ Align_Step2(array<Bitmap^>^ B, array<Bitmap^>^ C, array<float>^ ipts5, array<Drawing::Rectangle>^ MarginRect, int Width, int Height)
		{
			std::vector<cv::Mat> _B(B->Length);
			std::vector<cv::Size> _size_C(C->Length);
			std::vector<float> _ipts5(ipts5->Length);
			std::vector<cv::Rect2i> _MarginRect(MarginRect->Length);
			for (int i = 0; i < B->Length; i++)
			{
				ConvertBitmapToMat(B[i], _B[i]);
				_size_C[i] = cv::Size(C[i]->Size.Width, C[i]->Size.Height);
				_MarginRect[i] = cv::Rect2i(MarginRect[i].X, MarginRect[i].Y, MarginRect[i].Width, MarginRect[i].Height);
				for (size_t j = 0; j < 10; j++)
				{
					_ipts5[i * 10 + j] = ipts5[i * 10 + j];
				}
			}

			std::vector<cv::Mat> _F = _CaffeModel::AlignStep2(_B, _size_C, _ipts5, _MarginRect, Width, Height);
			array<Bitmap^>^ F = gcnew array<Bitmap^>(B->Length);
			for (size_t i = 0; i < B->Length; i++)
			{
				F[i] = gcnew System::Drawing::Bitmap(
					_F[i].cols, _F[i].rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				if (_F[i].channels()==4)
				{
					cvtColor(_F[i], _F[i], CV_BGRA2BGR);
				}
				CopyMatToBitmap(_F[i], F[i]);
				_B[i].release();
				_F[i].release();
			}
			return F;
		}

		static bool Train(String^ Solverpath)
		{
			return _CaffeModel::Train(TO_NATIVE_STRING(Solverpath));
		}

		static float CosineDistanceProb(array<float>^ feature1, array<float>^ feature2)
		{
			float output = 0;
			if (feature1->Length != feature2->Length)
			{
				output = -1;
			}
			else
			{
				output = innerproduct(feature1, feature2) / Math::Sqrt(innerproduct(feature1, feature1)*innerproduct(feature2, feature2));
			}
			return output;
			//return (Math::Asin(output)/Math::Acos(0)+1)/2;
		}

	private:
		cv::Mat Bitmap2Mat(System::Drawing::Bitmap^ bitmap)
		{
			IplImage* tmp;

			System::Drawing::Imaging::BitmapData^ bmData = bitmap->LockBits(System::Drawing::Rectangle(0, 0, bitmap->Width, bitmap->Height), System::Drawing::Imaging::ImageLockMode::ReadWrite, bitmap->PixelFormat);
			if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
			{
				tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 1);
				tmp->imageData = (char*)bmData->Scan0.ToPointer();
			}

			else if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 3);
				tmp->imageData = (char*)bmData->Scan0.ToPointer();
			}

			else if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
			{
				tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 4);
				tmp->imageData = (char*)bmData->Scan0.ToPointer();
			}

			bitmap->UnlockBits(bmData);

			return cv::cvarrToMat(tmp, true);
		}

		unsigned char* Bitmap2RGB(System::Drawing::Bitmap^ bmp)
		{
			int stride;
			unsigned char* res;
			System::Drawing::Imaging::BitmapData^ bmpd;
			if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed) {
				stride = bmpd->Stride;
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				return (unsigned char*)bmpd->Scan0.ToPointer();
			}
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				stride = bmp->Width + (4 - bmp->Width % 4) % 4;
				res = new unsigned char[stride * bmp->Height * 3];
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer(),
					*b, *g, *r;
				for (int offset = 0, y = 0; y < bmp->Height; ++y, offset += bmpd->Stride) {
					b = pBmp + offset + 0, g = pBmp + offset + 1, r = pBmp + offset + 2;
					for (int x = 0; x < bmpd->Width; ++x, b += 3, g += 3, r += 3) {
						//res[y * stride + x] = (unsigned char)((float)*r * 0.3f + (float)*g * 0.59f + (float)*b * 0.11f);
						//res[y * stride + x] = (unsigned char)(((s1mul0_3((int)*r)) + (s1mul0_59((int)*g)) + (s1mul0_11((int)*b))) >> 1);
						res[(y * stride + x) * 3] = *b;
						res[(y * stride + x) * 3 + 1] = *g;
						res[(y * stride + x) * 3 + 2] = *r;
					}
				}
				bmp->UnlockBits(bmpd);

			}
			return res;
		}

		/*----------------------------
		* 功能 : 将 cv::Mat? 转换为 Drawing::Bitmap?，拷贝图像数据
		*----------------------------
		* 函数 : CopyMatToBitmap
		* 访问 : public
		* 返回 : System::Drawing::Bitmap^
		* 参数 : cv::Mat * src
		*/
		static void CopyMatToBitmap(cv::Mat &src, System::Drawing::Bitmap^dst)
		{
			// ?bitmap 初始化  
			/*System::Drawing::Bitmap ^dst = gcnew System::Drawing::Bitmap(
			src.cols, src.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);*/
			/*if (src.channels()==4)
			{
				cvCvtColor(src, src, CV_BGRA2BGR);
			}
			if (src.channels()==1)
			{
				cvCvtColor(&src, &src, CV_GRAY2BGR);
			}*/
			// ?获取 bitmap 数据指针  
			System::Drawing::Imaging::BitmapData ^data = dst->LockBits(
				*(gcnew System::Drawing::Rectangle(0, 0, dst->Width, dst->Height)),
				System::Drawing::Imaging::ImageLockMode::ReadWrite,
				System::Drawing::Imaging::PixelFormat::Format24bppRgb
			);

			// 获取 cv::Mat 数据地址  
			//src->addref();
			//int x = src.channels();
			// ?复制图像数据  
			if (src.channels() == 3 && src.isContinuous()) {
				memcpy(data->Scan0.ToPointer(), src.data,
					src.rows * src.cols * src.channels());
			}
			else {
				for (int i = 0; i < src.rows * src.cols; i++) {
					byte *p = (byte *)data->Scan0.ToPointer();
					*(p + i * 3) = *(p + i * 3 + 1) = *(p + i * 3 + 2) = *(src.data + i);
				}
			}

			// 释放 cv::Mat 数据  
			src.release();

			// ?解除 bitmap 数据保护  
			dst->UnlockBits(data);

		}

		/*----------------------------
		* 功能 : 将图像格式由 System::Drawing::Bitmap 转换为 cv::Mat
		*      - 不拷贝图像数据
		*----------------------------
		* 函数 : ConvertBitmapToMat
		* 访问 : public
		* 返回 : 0：转换成功
		*
		* 参数 : bmpImg      [in]    .Net 图像
		* 参数 : cvImg       [out]   OpenCV 图像
		*/
		static int ConvertBitmapToMat(System::Drawing::Bitmap^ bmpImg, cv::Mat& cvImg)
		{
			int retVal = 0;
			int width = bmpImg->Width - bmpImg->Width % 4;
			int height = bmpImg->Height - bmpImg->Height % 4;
			Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

			// resize image
			Bitmap ^resizedBmp;
			resizedBmp = gcnew Bitmap((Image ^)bmpImg, width, height);
			resizedBmp = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);

			//锁定Bitmap数据  
			System::Drawing::Imaging::BitmapData^ bmpData = resizedBmp->LockBits(
				System::Drawing::Rectangle(0, 0, resizedBmp->Width, resizedBmp->Height),
				System::Drawing::Imaging::ImageLockMode::ReadWrite, resizedBmp->PixelFormat);

			//若cvImg非空，则清空原有数据  
			if (!cvImg.empty())
			{
				cvImg.release();
			}

			//将 bmpImg 的数据指针复制到 cvImg 中，不拷贝数据  
			if (resizedBmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)  // 灰度图像  
			{
				cvImg = cv::Mat(resizedBmp->Height, resizedBmp->Width, CV_8UC1, (char*)bmpData->Scan0.ToPointer());
			}
			else if (resizedBmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)   // 彩色图像  
			{
				cvImg = cv::Mat(resizedBmp->Height, resizedBmp->Width, CV_8UC3, (char*)bmpData->Scan0.ToPointer());
			}
			else if (resizedBmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)	//RGBA
			{
				cvImg = cv::Mat(resizedBmp->Height, resizedBmp->Width, CV_8UC4, (char*)bmpData->Scan0.ToPointer());
			}
			else
			{
				retVal = -1;
			}

			//解锁Bitmap数据  
			resizedBmp->UnlockBits(bmpData);

			/*
			bmpImg = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);*/

			return (retVal);
		}

		std::string ConvertToDatum(array<Byte>^ bytedata)
		{
			std::string datum_string;

			int width = m_net->GetInputImageWidth();
			int height = m_net->GetInputImageHeight();
			int channel = m_net->GetInputImageChannels();
			if (bytedata->Length!= channel * width * height)
			{
				return NULL;
			}
			else
			{
				datum_string.resize(channel * width * height);
				char *buff = &datum_string[0];
				int Stride = width;
				for (int c = 0; c < channel; ++c)
				{
					for (int h = 0; h < height; ++h)
					{
						int line_offset = h * Stride + c;
						for (int w = 0; w < width; ++w)
						{
							*buff++ = Convert::ToChar(bytedata[line_offset + w * channel]);
						}
					}
				}
				return datum_string;
			}
		}

		std::string ConvertToDatum(Bitmap ^imgData)
		{
			std::string datum_string;

			int width = m_net->GetInputImageWidth();
			int height = m_net->GetInputImageHeight();

			Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

			// resize image
			Bitmap ^resizedBmp;
			if (width == imgData->Width && height == imgData->Height)
			{
				resizedBmp = imgData->Clone(rc, PixelFormat::Format24bppRgb);
			}
			else
			{
				resizedBmp = gcnew Bitmap((Image ^)imgData, width, height);
				resizedBmp = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);
			}
			// get image data block
			BitmapData ^bmpData = resizedBmp->LockBits(rc, ImageLockMode::ReadOnly, resizedBmp->PixelFormat);
			pin_ptr<char> bmpBuffer = (char *)bmpData->Scan0.ToPointer();

			// prepare string buffer to call Caffe model
			datum_string.resize(3 * width * height);
			char *buff = &datum_string[0];
			for (int c = 0; c < 3; ++c)
			{
				for (int h = 0; h < height; ++h)
				{
					int line_offset = h * bmpData->Stride + c;
					for (int w = 0; w < width; ++w)
					{
						*buff++ = bmpBuffer[line_offset + w * 3];
					}
				}
			}
			resizedBmp->UnlockBits(bmpData);

			return datum_string;
		}



		static double innerproduct(array<float>^ feature1, array<float>^ feature2)
		{
			double output = 0;
			for (int i = 0; i < feature1->Length; i++)
			{
				output += feature1[i] * feature2[i];
			}
			return output;
		}

		

	};

	
}