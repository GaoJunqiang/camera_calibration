#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int getRemap()
{	
	
	double cam_k1[9] = {411.5919, 0.0, 314.2577, 0.0, 412.3891, 229.0431, 0.0, 0.0, 1.0};
	cv::Mat K1(3, 3, CV_64FC1, cam_k1);
	double cam_d1[4] = {-0.05584464104333399, -0.02347788145674064, 0.02061744418313239,
    -0.010737523144347836};
	cv::Mat D1(4, 1, CV_64FC1, cam_d1);
	
	double cam_k2[9] = {409.4971, 0.0,  329.4955, 0.0, 410.31936, 240.7883, 0.0, 0.0, 1.0};
	cv::Mat K2(3, 3, CV_64FC1, cam_k2);
	double cam_d2[4] = {-0.05948917800113634, -0.010104364177388412, -0.004568019387917628,
    0.0025937909845686435};
	cv::Mat D2(4, 1, CV_64FC1, cam_d2);
	cv::Size image_size(640, 480);
	
	double rot[9] = {0.9998804919298033, 0.003568250371239498, -0.015042255399492024,
					-0.003731607521706696, 0.9999342192519318, -0.010845841338245845,
					0.015002565231297605, 0.010900676966070026, 0.9998280343529896 };
	double trans[3] = {-0.0582809,-0.0005148, -0.000277129};
	cv::Mat R(3,3,CV_64FC1, rot);
	cv::Mat T(3,1,CV_64FC1, trans);

	
	cv::Mat R1, R2, P1, P2, Q;
    fisheye::stereoRectify(K1, D1, K2, D2, image_size, R, T, R1, R2, P1, P2,
                           Q, CV_CALIB_ZERO_DISPARITY, image_size, 0.0, 1.1);

    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    int useCalibrated = 1;
    Mat rmap[2][2];
    fisheye::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_16SC2, rmap[0][0], rmap[0][1]);
    fisheye::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_16SC2, rmap[1][0], rmap[1][1]);
    std::cout << "******** Done Calibration ********\n" << std::endl;
	
	cv::Mat imgSrc = imread("/home/devin16/my_data/big_red_buke/indoor_1/cam0/data/1552630814969836000.png");
	
	cv::Mat imgDst;
	cv::remap(imgSrc, imgDst, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	cv::imshow("dst", imgDst);
	cv::waitKey(0);
    //------------------------------save calibration result----------------------------------
    if (1)
    {
        FileStorage fs("./intrinsics.yml", CV_STORAGE_WRITE);
        if (fs.isOpened())
        {
            fs << "M1" << K1 << "D1" << D1 << "M2" << K2 << "D2" << D2;
            fs.release();
        }
        else
            cout << "Error: can not save the intrinsic parameters\n";

        fs.open("./new_intrinsics.yml", CV_STORAGE_WRITE);
        if (fs.isOpened())
        {
            fs << "P1" << P1 << "D1" << D1 << "P2" << P2 << "D2" << D2;
            fs.release();
        }
        else
            cout << "Error: can not save the intrinsic parameters\n";


        fs.open("extrinsics.yml", CV_STORAGE_WRITE);
        if (fs.isOpened())
        {
            fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
            fs.release();
        }
        else
            cout << "Error: can not save the intrinsic parameters\n";

        fs.open("remap.yml", CV_STORAGE_WRITE);
        if (fs.isOpened())
        {
            fs << "MapLx" << rmap[0][0] << "MapLy" << rmap[0][1] << "MapRx" << rmap[1][0] << "MapRy" << rmap[1][1];
            fs.release();
        }
        else
            cout << "Error: can not save the remap parameters\n";
    }
	
    return true;

}


//demo主函数
int main(int argc, char const *argv[])
{
    getRemap();

    return 0;
}

