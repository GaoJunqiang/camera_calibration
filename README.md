# camera_calibration
 normal, fisheye, mono, stereo camera calibration

/*------------------相机标定程序----------------------------
包括： 
    1、monoCalibration_fisheye:    单目鱼眼相机标定
    2、stereoCalibration_fisheye： 双目鱼眼相机标定
    3、stereoCalibration：         双目相机标定（普通相机）
    4、monoCalibration：           单目相机标定（普通相机）
    5、get_remap:                  通过已有的双目标定内外参数，得到立体矫正的remap文件。

标定程序安装编译详见各分支文件

需要安装依赖库popt: sudo apt-get install libpopt-dev
