#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

//双目鱼眼标定调用的函数，用于标定图像的读取和焦点提取
int load_image_points(cv::Size board_size, float square_size, int num_imgs, 
                       string img_dir, string leftimg_filename, string rightimg_filename, string image_format, 
                       vector<vector<Point2f> >& left_img_points, vector<vector<Point2f> >& right_img_points, 
                       vector<vector<Point3f> >& object_points, cv::Size& imageSize);


/********************************************************************××××××××××××××××××××××××××××××××××××××××
 * Function：  camera stereo calinration and evaluate calibration result using 3D reconstruction
 * Date：     2018.4.23
 * Author:    devin.gao
 * 
 * 该函数用于双目标定(正常相机)和基于双目重建的标定误差评估。标定利用opencv中的标定，函数中已将双目标定的内部参数（M1,D1,M2,D2)
 * ，外部参数（R,T,R1,R2,P1,P2,Q）立体矫正参数(MapLx,MapLy,MapRx,MapRy)存储在了intrinsics.yml，extrinsics.yml，
 * remap.yml三个文件中。评估方法是在双目标定后，利用标定参数重建评估图像上的棋盘格角点的三维坐标，然后计算相邻角点的三维距离和
 * 真实的物理距离作比较，得到标定误差。因此，在使用该函数做标定，需要采集0~n对棋盘格标定图像用作双目标定，接着采集n+1～m对棋盘格
 * 图像用作双目标定的评估，采集评估的棋盘格图像时，最好将棋盘格分布在图像的不同位置。
 * 
 * input parameters:
 *                  board_width：        棋盘格标定板角点的宽，即列数
 *                  board_height:        棋盘格标定板角点的高，即行数
 *                  square_size：        棋盘格标定板相邻角点间的物理距离，单位为mm
 *                  calib_imgs_num：     双目标定图像的数量（对数）,标定图像的序号从0开始，可以自动删除错误的标定图像对
 *                  img_dir：            标定图像所在文件夹的路径
 *                  leftimg_filename：   双目左图像的文件名，即图像序列号前的字符，例：图像名称是 left0.jpg，该参数为：left                
 *                  rightimg_filename：  双目右图像的文件名，即图像序列号前的字符，例：图像名称是 right0.jpg，该参数为：right
 *                  cali_images_format： 图像格式，0：bmp；1：jpg；2：png，如需扩展格式，只需要在在函数第8行下面新增一行代码即可
 *                  evalu_imgs_num：     评估双目标定的图像对数量，图像的文件名字和标定图像一样，序号紧接标定图像的序号
 *                  is_show_rectified：  是否显示立体矫正的图像，1：显示，0：不显示，默认参数是不显示
 *                  is_save_result：     是否保存标定参数，1：保存，0：不保存，默认参数是保存
 * output parameters：
 *                   calib_rms_error： opencv中双目标定函数（cv::stereoCalibrate）返回的rms值，表示反投影误差
 *                   calib_3D_error：  评估双目标定的三维误差
 * return value：    标定成功与否的标志 * 
 * 
 ********************************************************************/
int normalStereoCalibration(int board_width, int board_height, float square_size, int calib_imgs_num,
                             string img_dir, string leftimg_filename, string rightimg_filename, int cali_images_format, 
                             double& calib_rms_error, int evalu_imgs_num, double& calib_3D_error, 
                             int is_show_rectified=0, int is_save_result=1)
{
    if (board_width < 3 || board_height < 3 || square_size<=0 || calib_imgs_num<5) return false;
    calib_3D_error  = -1.0;
    calib_rms_error = -1.0;
    double calib_disparity_error  = -1.0;
    cv::Size board_size(board_width, board_height);
    string images_format;
    if (cali_images_format == 0) images_format = "bmp";
    else if (cali_images_format == 1) images_format = "jpg";
    else if (cali_images_format == 2) images_format = "png";
    else 
    {
        std::cout << "images format are wrong." << std::endl;
        return false;
    }
    vector<vector<Point2f> > left_img_points, right_img_points;
    vector<vector<Point3f> > object_points;
    cv::Size image_size;
    load_image_points(board_size, square_size, calib_imgs_num, 
                      img_dir, leftimg_filename, rightimg_filename, images_format,
                      left_img_points, right_img_points, object_points, image_size);

    //--------------------------------stereo calibration-----------------------------------
    if (object_points.size() < 4)
    {
        std::cout << "correct images numbers:  " << object_points.size() << "--too few calibration images" <<std::endl;
        return false;
    }
    std::cout << "\n******** Starting Calibration ********" << std::endl;
    cv::Mat K1, K2;
    cv::Mat R, T, E, F;
    cv::Mat D1, D2;
    int flags = 0;
    calib_rms_error = cv::stereoCalibrate(object_points, left_img_points, right_img_points,
                                                    K1, D1, K2, D2, image_size, R, T, E, F,
                                                    flags, cvTermCriteria(3, 20, 1e-6));
    std::cout << "#############  RMS:" << calib_rms_error << "#############" << std::endl;

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
    if (is_save_result)
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


    //------------------------evaluate stereo calibration result---------------------------
    vector<float> v3D_error;
    vector<double> disparity_aver;
    std::cout << "******** stereo calibration error ********" << std::endl;
    for (int index=calib_imgs_num; index<calib_imgs_num+evalu_imgs_num; ++index)
    {
        char left_img[100], right_img[100];
        sprintf(left_img, "%s%s%d.%s", img_dir.c_str(), leftimg_filename.c_str(), index, images_format.c_str());
        sprintf(right_img, "%s%s%d.%s", img_dir.c_str(), rightimg_filename.c_str(), index, images_format.c_str());

        cv::Mat img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
        cv::Mat img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
        if (img1.empty() || img2.empty() )
        {
            std::cout << index << ": NAN" << std::endl;
            continue;
        }
		//cv::resize(img1, img1, cv::Size(img1.cols/2,img1.rows/2), 0, 0, cv::INTER_LINEAR);
		//cv::resize(img2, img2, cv::Size(img2.cols/2,img2.rows/2), 0, 0, cv::INTER_LINEAR);
		//cv::pyrDown(img1, img1, cv::Size(img1.cols/2,img1.rows/2));
		//cv::pyrDown(img2, img2, cv::Size(img2.cols/2,img2.rows/2));
        cv::Mat recImg1, recImg2, gray1, gray2;
        remap(img1, recImg1, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
        remap(img2, recImg2, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
        cv::cvtColor(recImg1, gray1, CV_BGR2GRAY);
        cv::cvtColor(recImg2, gray2, CV_BGR2GRAY);
        bool found1 = false, found2 = false;
        Size board_size = Size(board_width, board_height);
        vector<Point2f> corners1, corners2;
        vector<cv::Point3f> v3DPoints;

        found1 = cv::findChessboardCorners(recImg1, board_size, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = cv::findChessboardCorners(recImg2, board_size, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1 && found2)
        {
            cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray1, board_size, corners1, found1);

            cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray2, board_size, corners2, found2);
        }
        else
        {
            std::cout << index << ": NAN" << std::endl;
            continue;
        }
        //stereo disparity error
        double disparity_error = 0.0;
        for (int i = 0; i < board_width * board_height; ++i)
        {
            double disparity_tmp = abs(corners1[i].y - corners2[i].y);
            disparity_error += disparity_tmp;
        }
        disparity_error = disparity_error / (board_width * board_height);
        disparity_aver.push_back(disparity_error);


        //3D reconstruction error
        double f = P1.at<double>(0, 0);
        double tx = T.at<double>(0, 0);
        double ty = T.at<double>(1, 0);
        double tz = T.at<double>(2, 0);
        for (int i = 0; i < board_width * board_height; ++i)
        {
//            double x_1 = corners1[i].x - P1.at<double>(0, 2);
//            double y_1 = corners1[i].y - P1.at<double>(1, 2);
//            double x_2 = corners2[i].x - P2.at<double>(0, 2);
//            double y_2 = corners2[i].y - P2.at<double>(1, 2);
            double y_1 = corners1[i].x - P1.at<double>(0, 2);
            double x_1 = corners1[i].y - P1.at<double>(1, 2);
            double y_2 = corners2[i].x - P2.at<double>(0, 2);
            double x_2 = corners2[i].y - P2.at<double>(1, 2);
			
            double w_z = (f * tx - x_2 * tz) / (x_2 - x_1);
            double w_x = w_z * x_1 / f;
            double w_y = w_z * y_1 / f;

            v3DPoints.push_back(cv::Point3f(w_x, w_y, w_z));
        }
        float dis_error = 0.0;
        int err_num = 0;
        //水平相邻角点距离误差
        for (int i = 0; i < board_height; ++i)
            for (int j = 0; j < board_width - 1; ++j)
            {
                float err_t = abs(sqrt((v3DPoints[i * board_width + j].x - v3DPoints[i * board_width + j + 1].x) * (v3DPoints[i * board_width + j].x - v3DPoints[i * board_width + j + 1].x) + (v3DPoints[i * board_width + j].y - v3DPoints[i * board_width + j + 1].y) * (v3DPoints[i * board_width + j].y - v3DPoints[i * board_width + j + 1].y) + (v3DPoints[i * board_width + j].z - v3DPoints[i * board_width + j + 1].z) * (v3DPoints[i * board_width + j].z - v3DPoints[i * board_width + j + 1].z)) - square_size);
                dis_error += err_t;
                err_num++;
				std::cout << "error " << i * board_width + j << ": " << err_t << std::endl;
            }
        //竖直相邻角点距离误差
        for (int i = 0; i < board_width; ++i)
        {
            for (int j = 0; j < board_height - 1; ++j)
            {
                float err_t = abs(sqrt((v3DPoints[j * board_width + i].x - v3DPoints[(j + 1) * board_width + i].x) * (v3DPoints[j * board_width + i].x - v3DPoints[(j + 1) * board_width + i].x) + (v3DPoints[j * board_width + i].y - v3DPoints[(j + 1) * board_width + i].y) * (v3DPoints[j * board_width + i].y - v3DPoints[(j + 1) * board_width + i].y) + (v3DPoints[j * board_width + i].z - v3DPoints[(j + 1) * board_width + i].z) * (v3DPoints[j * board_width + i].z - v3DPoints[(j + 1) * board_width + i].z)) - square_size);
                dis_error += err_t;
                err_num++;
                //cout << err_t << endl;
            }
        }
        v3D_error.push_back(dis_error / err_num);

        std::cout << "image." << index << "--stereo disparity error: " << disparity_error << " pixel" << "---"
                  << "3D reconstruction erro: " << dis_error / err_num << " mm" <<std::endl;
    }

    float error3d_sum = 0.0;
    for (int i=0; i<v3D_error.size(); ++i)
    {
        error3d_sum += v3D_error[i];
    }
    calib_3D_error = error3d_sum/v3D_error.size();
    std::cout << "#############  average 3D reconstruction error: " << calib_3D_error << " mm #############" << std::endl;

    float errorDisparity_sum = 0.0;
    for (int i=0; i<disparity_aver.size(); ++i)
    {
        errorDisparity_sum += disparity_aver[i];
    }
    calib_disparity_error = errorDisparity_sum/disparity_aver.size();
    std::cout << "#############  average disparity error: " << calib_disparity_error << " pixel #############\n" << std::endl;

    //-------------------------------show rectified stereo images----------------------------------
    if (is_show_rectified)
    {
        cv::Mat canvas;
        double sf;
        int w, h;
        if (!isVerticalStereo)
        {
            sf = 600. / MAX(image_size.width, image_size.height);
            w = cvRound(image_size.width * sf);
            h = cvRound(image_size.height * sf);
            canvas.create(h, w * 2, CV_8UC3);
        }
        else
        {
            sf = 300. / MAX(image_size.width, image_size.height);
            w = cvRound(image_size.width * sf);
            h = cvRound(image_size.height * sf);
            canvas.create(h * 2, w, CV_8UC3);
        }
        for (int i = 0; i < calib_imgs_num; i++)
        {
            char img_path_l[100], img_path_r[100];
            sprintf(img_path_l, "%s%s%d.%s", img_dir.c_str(), leftimg_filename.c_str(), i, images_format.c_str());
            sprintf(img_path_r, "%s%s%d.%s", img_dir.c_str(), rightimg_filename.c_str(), i, images_format.c_str());
            Mat rimg_l, rimg_r, cimg_l, cimg_r;
            Mat img_l = imread(img_path_l, 0); 
            Mat img_r = imread(img_path_r, 0);
            if (img_l.empty() || img_r.empty()) continue;
			//cv::resize(img_l, img_l, cv::Size(img_l.cols/2,img_l.rows/2), 0, 0, cv::INTER_LINEAR);
			//cv::resize(img_r, img_r, cv::Size(img_r.cols/2,img_r.rows/2), 0, 0, cv::INTER_LINEAR);
			//cv::pyrDown(img_l, img_l, cv::Size(img_l.cols/2,img_l.rows/2));
			//cv::pyrDown(img_r, img_r, cv::Size(img_r.cols/2,img_r.rows/2));
            remap(img_l, rimg_l, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
            remap(img_r, rimg_r, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
            cvtColor(rimg_l, cimg_l, COLOR_GRAY2BGR);
            cvtColor(rimg_r, cimg_r, COLOR_GRAY2BGR);
			imwrite("L.bmp", cimg_l);	
			imwrite("R.bmp", cimg_r);		

            Mat canvasPart = !isVerticalStereo ? canvas(Rect(0, 0, w, h)) : canvas(Rect(0, 0, w, h));
            resize(cimg_l, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);

            Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w, 0, w, h)) : canvas(Rect(0, h, w, h));
            resize(cimg_r, canvasPart2, canvasPart.size(), 0, 0, CV_INTER_AREA);

            if (!isVerticalStereo)
                for (int j = 0; j < canvas.rows; j += canvas.rows/20)
                    line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
            else
                for (int j = 0; j < canvas.cols; j += canvas.cols/20)
                    line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
            imshow("rectified", canvas);
            std::cout << "***image " << i << " is rectified***" << std::endl;
            char c = (char)waitKey(0);
        }
    }

    return true;

}

int load_image_points(cv::Size board_size, float square_size, int num_imgs, 
                       string img_dir, string leftimg_filename, string rightimg_filename, string image_format, 
                       vector<vector<Point2f> >& left_img_points, vector<vector<Point2f> >& right_img_points, 
                       vector<vector<Point3f> >& object_points, cv::Size& imageSize)
{
    left_img_points.clear();
    right_img_points.clear();
    object_points.clear();
    int board_n = board_size.width * board_size.height;

    cv::Mat gray1, gray2;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;
    for (int i = 0; i < num_imgs; i++)
    {
        char left_img[100], right_img[100];
        sprintf(left_img, "%s%s%d.%s", img_dir.c_str(), leftimg_filename.c_str(), i, image_format.c_str());
        sprintf(right_img, "%s%s%d.%s", img_dir.c_str(), rightimg_filename.c_str(), i, image_format.c_str());
        cv::Mat img1 = imread(left_img, -1);
        cv::Mat img2 = imread(right_img, -1);		
        if (img1.empty() || img2.empty() )
        {
			std::cout << left_img << " " << right_img << std::endl;
            std::cout << i << ". not Found corners!" << std::endl;
            continue;
        }
		//cv::resize(img1, img1, cv::Size(img1.cols/2,img1.rows/2), 0, 0, cv::INTER_LINEAR);
		//cv::resize(img2, img2, cv::Size(img2.cols/2,img2.rows/2), 0, 0, cv::INTER_LINEAR);
		//cv::pyrDown(img1, img1, cv::Size(img1.cols/2,img1.rows/2));
		//cv::pyrDown(img2, img2, cv::Size(img2.cols/2,img2.rows/2));
		if (img1.channels() == 3)
		{
			cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		}
		else
		{
			gray1 = img1.clone();
		}
		if (img2.channels() == 3)
		{
			cv::cvtColor(img2, gray2, CV_BGR2GRAY);
		}
		else
		{
			gray2 = img2.clone();
		}


        bool found1 = false, found2 = false;

        found1 = cv::findChessboardCorners(img1, board_size, corners1,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = cv::findChessboardCorners(img2, board_size, corners2,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1 && found2)
        {
            cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray1, board_size, corners1, found1);
            cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray2, board_size, corners2, found2);

            // imshow("gray1", gray1);
            // imshow("gray2", gray2);
            // char c = (char)waitKey(0);

        }
        else
        {
            std::cout << i << ". not Found corners!" << std::endl;
            continue;
        }
        vector<cv::Point3f> obj;
        for (int k = 0; k < board_size.height; ++k)
            for (int j = 0; j < board_size.width; ++j)
                obj.push_back(Point3f((float)j * square_size, (float)k * square_size, 0));

        cout << i << ". Found corners!" << endl;
        imagePoints1.push_back(corners1);
        imagePoints2.push_back(corners2);
        object_points.push_back(obj);
    }
    for (int k = 0; k < imagePoints1.size(); k++)
    {
        vector<Point2f> v1, v2;
        for (int j = 0; j < imagePoints1[k].size(); j++)
        {
            v1.push_back(Point2f(imagePoints1[k][j].x, imagePoints1[k][j].y));
            v2.push_back(Point2f(imagePoints2[k][j].x, imagePoints2[k][j].y));
        }
        left_img_points.push_back(v1);
        right_img_points.push_back(v2);
    }
    imageSize = gray1.size();
    return true;
}


//demo主函数
int main(int argc, char const *argv[])
{
    int board_width, board_height, num_imgs;
    float square_size;
    char *img_dir;
    char *leftimg_filename;
    char *rightimg_filename;
    char *out_file;

    static struct poptOption options[] = {
        {"board_width", 'w', POPT_ARG_INT, &board_width, 0, "Checkerboard width", "NUM"},
        {"board_height", 'h', POPT_ARG_INT, &board_height, 0, "Checkerboard height", "NUM"},
        {"square_size", 's', POPT_ARG_FLOAT, &square_size, 0, "Checkerboard square size", "NUM"},
        {"num_imgs", 'n', POPT_ARG_INT, &num_imgs, 0, "Number of checkerboard images", "NUM"},
        {"img_dir", 'd', POPT_ARG_STRING, &img_dir, 0, "Directory containing images", "STR"},
        {"leftimg_filename", 'l', POPT_ARG_STRING, &leftimg_filename, 0, "Left image prefix", "STR"},
        {"rightimg_filename", 'r', POPT_ARG_STRING, &rightimg_filename, 0, "Right image prefix", "STR"},
        POPT_AUTOHELP{NULL, 0, 0, NULL, 0, NULL, NULL}};

    POpt popt(NULL, argc, argv, options, 0);
    int c;
    while ((c = popt.getNextOpt()) >= 0)
    {
    }
    double calib_rms_error = 0.0; 
    double calib_3D_error = 0.0;
    normalStereoCalibration(board_width, board_height, square_size, num_imgs,
                             img_dir, leftimg_filename, rightimg_filename, 0, 
                             calib_rms_error, 20, calib_3D_error, 
                             1);

    return 0;
}
