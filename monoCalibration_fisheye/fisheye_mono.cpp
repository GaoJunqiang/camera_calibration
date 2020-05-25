//============================================================================
// Name        : fisheyestereo.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

int main() {
	ofstream fout("caliberation_result.txt");  /**    保存定标结果的文件     **/

	    /************************************************************************
	           读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
	    *************************************************************************/
	    cout<<"开始提取角点………………"<<endl;
        int image_count=  26;                    /****    图像数量     ****/
        Size board_size = Size(9,6);            /****    定标板上每行、列的角点数       ****/
	    vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
	    vector < vector < Point2f > >  corners_Seq;    /****  保存检测到的所有角点       ****/
	    vector<Mat>  image_Seq;
	    int successImageNum = 0;                /****   成功提取角点的棋盘图数量    ****/
	   // Size square_size = Size(108,108);
        Size square_size = Size(2.7,2.7);
		cv::namedWindow("corner", 0);
	    int count = 0;
        string tempsub = "/home/ai/workspace/tan/test/testOpenCV/4_video/build/calibration/data7/";
	    for( int i = 0; i != image_count ; i++)
	    {
	        cout<<"Frame #"<<i<<"..."<<endl;
	        string imageFileName;
	        std::stringstream StrStm;
            StrStm<<i;
	        StrStm>>imageFileName;
            imageFileName += ".jpg";
            // imageFileName += ".png";
			string calib_mage_path = tempsub+imageFileName;
            cv::Mat image = imread(calib_mage_path, 1);
			if (image.empty())
			{
				cout<<"can not find image" << i << ": " << calib_mage_path << endl;
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
	                circle( imageTemp, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
	            }
	            string imageFileName;
	            std::stringstream StrStm;
	            StrStm<<i+1;
	            StrStm>>imageFileName;
	            imageFileName = "./corners/"+imageFileName+"_corner.jpg";
	            imwrite(imageFileName,imageTemp);
	            cout<<"Frame corner#"<<i<<"...end"<<endl;
	            imshow("corner",imageTemp);
	            waitKey(100);

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
	    int flags = 0;
	    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	    flags |= cv::fisheye::CALIB_CHECK_COND;
	    flags |= cv::fisheye::CALIB_FIX_SKEW;
        //flags |= cv::fisheye::CALIB_FIX_K3 ;
        //flags |= cv::fisheye::CALIB_FIX_K4;
	    fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
	    cout<<"定标完成！\n";

	    /************************************************************************
	           对定标结果进行评价
	    *************************************************************************/
	    cout<<"开始评价定标结果………………"<<endl;
	    double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
	    double err = 0.0;                        /* 每幅图像的平均误差 */
	    vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

	    cout<<"每幅图像的定标误差："<<endl;
	    cout<<"每幅图像的定标误差："<<endl<<endl;
	    for (int i=0;  i<successImageNum;  i++)
	    {
	    	cout<<"0123   "<<i<< " "<<object_Points.size()<<endl;
	        vector<Point3f> tempPointSet = object_Points[i];
	        cout<<"0123   0"<<i<<endl;
	        /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
	        fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
	        /* 计算新的投影点和旧的投影点之间的误差*/
	        cout<<"0123   1"<<i<<endl;
	        vector<Point2f> tempImagePoint = corners_Seq[i];
	        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
	        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
	        cout<<"0123   2"<<i<<endl;
	        for (size_t i = 0 ; i != tempImagePoint.size(); i++)
	        {
	            image_points2Mat.at<Vec2f>(0,i) = Vec2f(image_points2[i].x, image_points2[i].y);
	            tempImagePointMat.at<Vec2f>(0,i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
	        }
	        cout<<"0123   3"<<i<<endl;
	        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
	        total_err += err/=  point_counts[i];
	        cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
	        cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
	        cout<<"123"<<endl;
	    }
	    cout<<"1234"<<endl;
	    cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;
	    cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;
	    cout<<"评价完成！"<<endl;

	    /************************************************************************
	           保存定标结果
	    *************************************************************************/
		cv::FileStorage fs("./intrinsics.yml", CV_STORAGE_WRITE);
        if (fs.isOpened())
        {
            fs << "M1" << intrinsic_matrix << "D1" << distortion_coeffs;
            fs.release();
        }
        else
            cout << "Error: can not save the intrinsic parameters\n";

	    cout<<"开始保存定标结果………………"<<endl;
	    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

	    cout<<"相机内参数矩阵："<<endl;
	    cout<<intrinsic_matrix<<endl;
	    cout<<"畸变系数：\n";
	    cout<<distortion_coeffs<<endl;
	    for (int i=0; i<image_count; i++)
	    {
	        cout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;
	        cout<<rotation_vectors[i]<<endl;

	        /* 将旋转向量转换为相对应的旋转矩阵 */
	        Rodrigues(rotation_vectors[i],rotation_matrix);
	        cout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;
	        cout<<rotation_matrix<<endl;
	        cout<<"第"<<i+1<<"幅图像的平移向量："<<endl;
	        cout<<translation_vectors[i]<<endl;
	    }
	    cout<<"完成保存"<<endl;
	    cout<<endl;


	    /************************************************************************
	           显示定标结果
	    *************************************************************************/
	    Mat mapx = Mat(image_size,CV_32FC1);
	    Mat mapy = Mat(image_size,CV_32FC1);
	    Mat R = Mat::eye(3,3,CV_32F);

	    cout<<"保存矫正图像"<<endl;

	    //cv::Size image_size2(image_size.width*1.3, image_size.height*1.3);
		cv::Mat intrinsic_matrix2 = intrinsic_matrix.clone();
		intrinsic_matrix2.at<double>(0,0) = intrinsic_matrix2.at<double>(0,0) / 1.5;
		intrinsic_matrix2.at<double>(1,1) = intrinsic_matrix2.at<double>(1,1) / 1.5;
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, 
		 				R,intrinsic_matrix2, image_size, CV_16SC2, mapx,mapy);

		cv::FileStorage fs1("./newIntrinsics.yml", CV_STORAGE_WRITE);
        if (fs1.isOpened())
        {
            fs1 << "M1" << intrinsic_matrix2 << "D1" << distortion_coeffs;
            fs1.release();
        }
		
		// fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
		// 	getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs,
		// 			image_size, 1, image_size, 0), image_size, CV_32FC1, mapx,mapy);

        FileStorage fs2;
        fs2.open("remap.yml", CV_STORAGE_WRITE);
        if( fs2.isOpened() )
        {
        	fs2 << "Mapx" << mapx<< "Mapy" <<  mapy;
        	fs2.release();
        }
        else
        	cout << "Error: can not save the remap parameters\n";
		
		// cv::undistortImage(InputArray distorted, OutputArray undistorted, InputArray K, 
		// InputArray D, InputArray Knew=cv::noArray(), const Size& new_size=Size())
		cv::namedWindow("uimage", 0);
	    for (int i = 0 ; i != successImageNum ; i++)
	    {
	        cout<<"Frame #"<<i<<"..."<<endl;
	        //fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
	        //fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,
	        //   getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0),image_size,CV_32FC1,mapx,mapy);
	        Mat t = image_Seq[i].clone();
	        cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT);
			//cv::fisheye::undistortImage (image_Seq[i], t, intrinsic_matrix, distortion_coeffs, intrinsic_matrix2);
			
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
	    cout<<"保存结束"<<endl;

	    /************************************************************************
	           测试一张图片
	    *************************************************************************/
	    if (0)
	    {
	        cout<<"TestImage ..."<<endl;			
			string test_path = tempsub + "img20.bmp";
			cv::Mat imgSrc = imread(test_path);
			Mat imageGray;
	        cvtColor(imgSrc, imageGray , CV_RGB2GRAY);
			vector<Point2f> test_corners;
	        bool patternfound = findChessboardCorners(imageGray, board_size, test_corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+
	            CALIB_CB_FAST_CHECK );
	        if (!patternfound)
	        {
	            cout<<"test image can not find chessboard corners!\n";
	        }
	        else
	        {
	            /* 亚像素精确化 */
	            cornerSubPix(imageGray, test_corners, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	            /* 绘制检测到的角点并保存 */
	            Mat imageTemp = imgSrc.clone();
	            for (unsigned int j = 0; j < corners.size(); j++)
	            {
	                circle( imageTemp, test_corners[j], 10, Scalar(0,0,255), 2, 8, 0);
	            }
	            imshow("corner",imageTemp);
	            waitKey(100);
	        }
	    }
	
	
	    return 0;
}
