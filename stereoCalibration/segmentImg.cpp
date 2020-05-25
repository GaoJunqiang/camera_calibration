#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

int main()
{
	string fold_path = "./red_phone_calib/";
	cv::namedWindow("src", 0);
	cv::namedWindow("rotationImg", 0);
	for (int i=0; i<21; ++i)
	{
		string img_index;
		std::stringstream StrStm;
		StrStm<<i;
		StrStm>>img_index;
		string img_path = fold_path + img_index + ".jpg";

		cv::Mat imgSrc = imread(img_path, -1);
		
		cv::Point2f center(imgSrc.cols / 2., imgSrc.rows / 2.);
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, 180.0, 1.0);
		cv::Size src_sz = imgSrc.size();
		cv::Mat roImg;
		cv::warpAffine(imgSrc, roImg, rot_mat, src_sz);
//		imshow("src", imgSrc);
//		imshow("rotationImg", roImg);
//		waitKey(0);
		
		cv::Mat imgL = roImg(Rect(0, 0, 720, 1280));
		cv::Mat imgR = roImg(Rect(720, 0, 720, 1280));
		string img_path_L = "./L" + img_index + ".bmp";
		string img_path_R = "./R" + img_index + ".bmp";
		imwrite(img_path_L, imgL);
		imwrite(img_path_R, imgR);
	}
	
	return 0;

}
