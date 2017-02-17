// model_encryption.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <algorithm>
#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <glog/logging.h>
#include <io.h>
#include <fcntl.h>
#include <caffe/proto/caffe.pb.h>
#include <windows.h>

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.


std::string Base64Encode(const std::string &strString)
{
	size_t nByteSrc = strString.length();
	std::string pszSource = strString;

	int i = 0;
	for (i; i < nByteSrc; i++)
		if (pszSource[i] < 0 || pszSource[i] > 127)
			throw "can not encode Non-ASCII characters";

	const char *enkey = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string pszEncode(nByteSrc * 4 / 3 + 4, '\0');
	size_t nLoop = nByteSrc % 3 == 0 ? nByteSrc : nByteSrc - 3;
	int n = 0;
	for (i = 0; i < nLoop; i += 3)
	{
		pszEncode[n] = enkey[pszSource[i] >> 2];
		pszEncode[n + 1] = enkey[((pszSource[i] & 3) << 4) | ((pszSource[i + 1] & 0xF0) >> 4)];
		pszEncode[n + 2] = enkey[((pszSource[i + 1] & 0x0f) << 2) | ((pszSource[i + 2] & 0xc0) >> 6)];
		pszEncode[n + 3] = enkey[pszSource[i + 2] & 0x3F];
		n += 4;
	}

	switch (nByteSrc % 3)
	{
	case 0:
		pszEncode[n] = '\0';
		break;

	case 1:
		pszEncode[n] = enkey[pszSource[i] >> 2];
		pszEncode[n + 1] = enkey[((pszSource[i] & 3) << 4) | ((0 & 0xf0) >> 4)];
		pszEncode[n + 2] = '=';
		pszEncode[n + 3] = '=';
		pszEncode[n + 4] = '\0';
		break;

	case 2:
		pszEncode[n] = enkey[pszSource[i] >> 2];
		pszEncode[n + 1] = enkey[((pszSource[i] & 3) << 4) | ((pszSource[i + 1] & 0xf0) >> 4)];
		pszEncode[n + 2] = enkey[((pszSource[i + 1] & 0xf) << 2) | ((0 & 0xc0) >> 6)];
		pszEncode[n + 3] = '=';
		pszEncode[n + 4] = '\0';
		break;
	}

	return pszEncode.c_str();
}

std::string Base64Decode(const std::string &strString)
{
	size_t nByteSrc = strString.length();
	std::string pszSource = strString;

	const int dekey[] = {
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		62, // '+'
		-1, -1, -1,
		63, // '/'
		52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
		-1, -1, -1, -1, -1, -1, -1,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
		-1, -1, -1, -1, -1, -1,
		26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
	};

	if (nByteSrc % 4 != 0)
		throw "bad base64 string";

	std::string pszDecode(nByteSrc * 3 / 4 + 4, '\0');
	size_t nLoop = pszSource[nByteSrc - 1] == '=' ? nByteSrc - 4 : nByteSrc;
	int b[4];
	int i = 0, n = 0;
	for (i = 0; i < nLoop; i += 4)
	{
		b[0] = dekey[pszSource[i]];        b[1] = dekey[pszSource[i + 1]];
		b[2] = dekey[pszSource[i + 2]];    b[3] = dekey[pszSource[i + 3]];
		if (b[0] == -1 || b[1] == -1 || b[2] == -1 || b[3] == -1)
			throw "bad base64 string";

		pszDecode[n] = (b[0] << 2) | ((b[1] & 0x30) >> 4);
		pszDecode[n + 1] = ((b[1] & 0xf) << 4) | ((b[2] & 0x3c) >> 2);
		pszDecode[n + 2] = ((b[2] & 0x3) << 6) | b[3];

		n += 3;
	}

	if (pszSource[nByteSrc - 1] == '=' && pszSource[nByteSrc - 2] == '=')
	{
		b[0] = dekey[pszSource[i]];        b[1] = dekey[pszSource[i + 1]];
		if (b[0] == -1 || b[1] == -1)
			throw "bad base64 string";

		pszDecode[n] = (b[0] << 2) | ((b[1] & 0x30) >> 4);
		pszDecode[n + 1] = '\0';
	}

	if (pszSource[nByteSrc - 1] == '=' && pszSource[nByteSrc - 2] != '=')
	{
		b[0] = dekey[pszSource[i]];        b[1] = dekey[pszSource[i + 1]];
		b[2] = dekey[pszSource[i + 2]];
		if (b[0] == -1 || b[1] == -1 || b[2] == -1)
			throw "bad base64 string";

		pszDecode[n] = (b[0] << 2) | ((b[1] & 0x30) >> 4);
		pszDecode[n + 1] = ((b[1] & 0xf) << 4) | ((b[2] & 0x3c) >> 2);
		pszDecode[n + 2] = '\0';
	}

	if (pszSource[nByteSrc - 1] != '=' && pszSource[nByteSrc - 2] != '=')
		pszDecode[n] = '\0';

	return pszDecode;
}



std::string PwdEncode(std::string str)
{
	char keys[8] = { 0x47, 0x6C, 0x61, 0x73, 0x73, 0x73, 0x69, 0x78 };
	for (size_t i = 0; i < str.length(); i++)
	{
		str[i] = (str[i] ^ keys[i % 8]);
	}
	return str;
}

std::string PwdDecode(std::string str)
{
	char keys[8] = { 0x47, 0x6C, 0x61, 0x73, 0x73, 0x73, 0x69, 0x78 };
	for (size_t i = 0; i < str.length(); i++)
	{
		str[i] ^= keys[i % 8];
	}
	return str;
}


void encode()
{
	std::ifstream fin("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy.prototxt");
	std::vector<std::string> oristr;
	std::vector<std::string> ecdstr;
	std::string s;
	while (getline(fin, s))
	{
		std::string withoutspace="";
		for (size_t i = 0; i < s.length(); i++)
		{
			if (s[i]!=0x20)
			{
				withoutspace += s[i];
			}
		}
		withoutspace =  withoutspace + "##";
		oristr.push_back(withoutspace);
		ecdstr.push_back(Base64Encode(withoutspace));
	}
	/*std::string encode_strs = PwdEncode(strs);*/

	std::ofstream ofs("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy_encode.prototxt");
	for (size_t i = 0; i < oristr.size(); i++)
	{
		ofs << ecdstr[i] << std::endl;
	}
	//ofs.write(encode_strs.c_str(), strlen(encode_strs.c_str()));
	ofs.close();
}

void decode()
{
	std::ifstream fin("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy_encode.prototxt");
	std::vector<std::string> ecdstr;
	std::vector<std::string> decstr;
	std::string s;
	while (getline(fin, s))
	{
		ecdstr.push_back(s);
		std::string temp = Base64Decode(s);
		std::string dec = "";
		for (size_t i = 0; i < temp.size(); i++)
		{
			if (temp[i]!='\0'&&temp[i]!=0x23)
			{
				dec += temp[i];
			}
			
		}
		decstr.push_back(dec);
	}
	//std::string decode_strs = PwdDecode(strs);

	std::ofstream ofs("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy_decode.prototxt");
	std::string output = "";
	for (size_t i = 0; i < ecdstr.size(); i++)
	{
		output += (decstr[i] + '\n');
		//ofs << decstr[i]+'\n';// << std::endl;
	}
	ofs << output;
	//ofs.write(decode_strs.c_str(), strlen(decode_strs.c_str()));
	ofs.close();
	caffe::NetParameter param;
	bool success = google::protobuf::TextFormat::ParseFromString(output, &param);
}


int main()
{
	/*FILE* pFile = fopen("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy.prototxt", "rb");
	char *pBuf;
	fseek(pFile, 0, SEEK_END);
	int len = ftell(pFile);
	pBuf = new char[len];
	rewind(pFile);
	fread(pBuf, 1, len, pFile);
	pBuf[len] = 0;
	fclose(pFile);

	std::string strs(pBuf, len);*/

	//std::ifstream fin("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy.prototxt");
	//std::string strs="";
	//std::string s;
	//while (fin >> s)
	//{
	//	strs += s;
	//}
	////delete[] pBuf;
	//std::string encode_strs = Base64Encode(strs);
	//std::string decode_strs = Base64Decode(strs);
	//caffe::NetParameter param;
	//bool success = google::protobuf::TextFormat::ParseFromString(decode_strs, &param);

	//std::string outstr;
	//success = google::protobuf::TextFormat::PrintToString(param, &outstr);

	//success = google::protobuf::TextFormat::ParseFromString(outstr, &param);

	//std::ofstream ofs("D:\\Research\\FacialLandmarks\\Code\\IPBBox\\IPBBox_deploy_decode.prototxt");
	//ofs.write(decode_strs.c_str(), strlen(decode_strs.c_str()));
	//ofs.close();

	encode();
	decode();


    return 0;
}

