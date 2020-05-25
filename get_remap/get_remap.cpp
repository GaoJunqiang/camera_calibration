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
	double cam_k1[9] = {414.3574706, 0.0, 321.04326, 0.0, 413.983907, 251.37605, 0.0, 0.0, 1.0};
	cv::Mat K1(3, 3, CV_64FC1, cam_k1);
	double cam_d1[5] = {-0.3613801482383776, 0.12540178633361745, 0.001237971310747468, 0.0006757107858447016, 0.0};
	cv::Mat D1(5, 1, CV_64FC1, cam_d1);
	
	double cam_k2[9] = {413.6776, 0.0, 344.8163, 0.0, 413.4748, 235.3869, 0.0, 0.0, 1.0};
	cv::Mat K2(3, 3, CV_64FC1, cam_k2);
	double cam_d2[5] = {-0.35433739478156506, 0.11378057599690586, -0.0002944728206929264, 0.00038793559097615593, 0.0};
	cv::Mat D2(5, 1, CV_64FC1, cam_d2);
	cv::Size image_size(640, 480);
	
	double rot[9] = {0.9986386716667941, -0.0022357097366940454, -0.05211338651005067,
					0.0021097643677495735, 0.9999947199255848, -0.002471642218378091,
					0.05211863722206534, 0.002358330535855008, 0.998638115601037};
	double trans[3] = {-0.05846375532468877, -0.000242116, -0.00158912};
	cv::Mat R(3,3,CV_64FC1, rot);
	cv::Mat T(3,1,CV_64FC1, trans);
	
	cv::Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    cv::stereoRectify(K1, D1, K2, D2, image_size, R, T, R1, R2, P1, P2,
                           Q, CV_CALIB_ZERO_DISPARITY, -1, image_size, &validRoi[0], &validRoi[1]);
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    int useCalibrated = 1;
    Mat rmap[2][2];
    cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_16SC2, rmap[1][0], rmap[1][1]);
    std::cout << "******** Done Calibration ********\n" << std::endl;

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

