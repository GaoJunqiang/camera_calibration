#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "popt_pp.h"
using namespace std;
using namespace cv;

int CalibMono(int board_width, int board_height, float square_length, int image_count,
              string img_dir, string img_filename, string cali_images_format, double& calib_rms_error, 
			  int is_show_rectified=0, int is_save_result=1); 

int main(int argc, char const *argv[])
{
    int board_width, board_height, num_imgs;
    float square_length;
    char *img_dir;
    char *img_filename;
    char* img_format;

    static struct poptOption options[] = {
        {"board_width", 'w', POPT_ARG_INT, &board_width, 0, "Checkerboard width", "NUM"},
        {"board_height", 'h', POPT_ARG_INT, &board_height, 0, "Checkerboard height", "NUM"},
        {"square_size", 's', POPT_ARG_FLOAT, &square_length, 0, "Checkerboard square size", "NUM"},
        {"num_imgs", 'n', POPT_ARG_INT, &num_imgs, 0, "Number of checkerboard images", "NUM"},
        {"img_dir", 'd', POPT_ARG_STRING, &img_dir, 0, "Directory containing images", "STR"},
        {"img_name", 'i', POPT_ARG_STRING, &img_filename, 0, "image prefix", "STR"},
        {"img_format", 'f', POPT_ARG_STRING, &img_format, 0, "Checker image format", "NUM"},
        POPT_AUTOHELP{0, 0, 0, 0, 0, 0, 0}};

    POpt popt(NULL, argc, argv, options, 0);
    int c;
    while ((c = popt.getNextOpt()) >= 0)
    {
    }
    
    std::cout << "1\n";
    std::cout << string(img_dir) << " " << string(img_filename) << std::endl;
    double calib_rms_error = 0.0;
    int calib_flag = CalibMono(board_width, board_height, square_length, num_imgs, string(img_dir), string(img_filename), string(img_format), calib_rms_error, 1, 1);
    return calib_flag;
    
}

int CalibMono(int board_width, int board_height, float square_length, int image_count,
              string img_dir, string img_filename, string cali_images_format, double& calib_rms_error, 
			  int is_show_rectified, int is_save_result) 
{
    Size board_size = Size(board_width, board_height);
    Size square_size(square_length, square_length);
    if (cali_images_format[0] != '.')
    {
    	cali_images_format = string(".") + cali_images_format;
    }
    cout<<"开始提取角点………………"<<endl;
    
    vector<Point2f> corners;
    vector < vector < Point2f > >  corners_Seq; 
    vector<Mat> image_Seq;
    int successImageNum = 0;
    
	cv::namedWindow("corner", 0);
    int count = 0;
    for( int i = 0; i!=image_count; i++)
    {
        cout<<"Frame #"<<i<<"..."<<endl;
        string imageFileName;
        std::stringstream StrStm;
        StrStm << i;
        StrStm >> imageFileName;
        imageFileName += cali_images_format;
        // imageFileName += ".png";
		string calib_image_path = img_dir + img_filename + imageFileName;
        cv::Mat image = imread(calib_image_path, 1);
		if (image.empty())
		{
			cout<<"can not find image" << i << ": " << calib_image_path << endl;
			continue;
		}
        //imshow("image",image);
        //waitKey(100);
        /* 提取角点 */
        Mat imageGray;
        cvtColor(image, imageGray , CV_RGB2GRAY);
        bool patternfound = findChessboardCorners(image, board_size, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+
            CALIB_CB_FAST_CHECK );
        if (!patternfound)
        {
            cout<<"can not find chessboard corners!\n";
            continue;
            exit(1);
        }
        else
        {
            /* 亚像素精确化 */
            cornerSubPix(imageGray, corners, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            /* 绘制检测到的角点并保存 */
            Mat imageTemp = image.clone();
            for (unsigned int j = 0; j < corners.size(); j++)
            {
                circle(imageTemp, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
            }

            cout<<"Frame corner#"<<i<<"...end"<<endl;
            imshow("corner",imageTemp);
            waitKey(0);

            count = count + corners.size();
            successImageNum = successImageNum + 1;
            corners_Seq.push_back(corners);
        }
        image_Seq.push_back(image);
    }
    cout<<"角点提取完成！\n";
    /************************************************************************
           摄像机定标
    *************************************************************************/
    cout<<"开始定标………………"<<endl;

    vector < vector < Point3f > >  object_Points;        /****  保存定标板上角点的三维坐标   ****/

    Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));  /*****   保存提取的所有角点   *****/
    vector<int>  point_counts;
    /* 初始化定标板上角点的三维坐标 */
    for (int t = 0; t<successImageNum; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i<board_size.height; i++)
        {
            for (int j = 0; j<board_size.width; j++)
            {
                /* 假设定标板放在世界坐标系中z=0的平面上 */
                Point3f tempPoint;
                tempPoint.y = i*square_size.height;
                tempPoint.x = j*square_size.width;
                tempPoint.z = 0;
                tempPointSet.push_back(tempPoint);
            }
        }
        object_Points.push_back(tempPointSet);
    }
    for (int i = 0; i< successImageNum; i++)
    {
        point_counts.push_back(board_size.width*board_size.height);
    }
    /* 开始定标 */
    Size image_size = image_Seq[0].size();
    cv::Mat intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
    cv::Mat distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
    std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
    cv::calibrateCamera(object_Points, corners_Seq, image_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vectors, translation_vectors);
    
    Mat mapx = Mat(image_size,CV_32FC1);
    Mat mapy = Mat(image_size,CV_32FC1);
    Mat R = Mat::eye(3,3,CV_32F);
    cv::Size image_size2(image_size.width, image_size.height);
	cv::Mat intrinsic_matrix2 = intrinsic_matrix.clone();
	//intrinsic_matrix2.at<double>(0,0) = intrinsic_matrix2.at<double>(0,0) / 3.0;
	//intrinsic_matrix2.at<double>(1,1) = intrinsic_matrix2.at<double>(1,1) / 3.0;
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, 
	 				R,intrinsic_matrix2, image_size, CV_16SC2, mapx,mapy);
    cout<<"定标完成！\n";
    

    if (is_save_result)
    {
    	std::cout << "开始保存标定结果" << std::endl;
		cv::FileStorage fs("./intrinsics.yml", CV_STORAGE_WRITE);
		if (fs.isOpened())
		{
		    fs << "M1" << intrinsic_matrix << "D1" << distortion_coeffs;
		    fs.release();
		}
		else
		    cout << "Error: can not save the intrinsic parameters\n";
		    
		FileStorage fs2;
		fs2.open("./remap.yml", CV_STORAGE_WRITE);
		if( fs2.isOpened() )
		{
			fs2 << "Mapx" << mapx<< "Mapy" <<  mapy;
			fs2.release();
		}
		else
			cout << "Error: can not save the remap parameters\n";
		cv::FileStorage fs3("./new_intrinsics.yml", CV_STORAGE_WRITE);
		if (fs3.isOpened())
		{
		    fs3 << "M1" << intrinsic_matrix2;
		    fs3.release();
		}
		else
		    cout << "Error: can not save the intrinsic parameters\n";
		std::cout << "保存标定完成" << std::endl;
	}
	

	

    cout<<"开始评价定标结果"<<endl;
    double total_err = 0.0; 
    double err = 0.0;
    vector<Point2f>  image_points2; 

    for (int i=0;  i<successImageNum;  i++)
    {
        vector<Point3f> tempPointSet = object_Points[i];
        //通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
        projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
        // 计算新的投影点和旧的投影点之间的误差
        vector<Point2f> tempImagePoint = corners_Seq[i];
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
        for (size_t i = 0 ; i != tempImagePoint.size(); i++)
        {
            image_points2Mat.at<Vec2f>(0,i) = Vec2f(image_points2[i].x, image_points2[i].y);
            tempImagePointMat.at<Vec2f>(0,i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err/=  point_counts[i];
    }
    calib_rms_error = total_err / image_count;
    cout<<"总体平均误差："<< calib_rms_error <<"像素"<<endl;
    cout<<"评价完成！"<<endl;

	if (is_show_rectified)
	{
		cv::namedWindow("uimage", 0);
		for (int i = 0 ; i != successImageNum ; i++)
		{
		    cout<<"Frame #"<<i<<"..."<<endl;
		    Mat t = image_Seq[i].clone();
		    cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT);
		
			string imageFileName;
		    std::stringstream StrStm;
		    StrStm<<i+1;
		    StrStm>>imageFileName;
		    imageFileName = "./uimages/" + imageFileName + "_d.jpg";
		    imwrite(imageFileName,t);
		    imshow("uimage",t);
		    char key = waitKey(0);
			if (key == 'q') break;

		}
    }

    return 1;
}
