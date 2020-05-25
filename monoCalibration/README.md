1 题目：单目相机高精度标定和评估

2 安装： 
       本代码依赖于opencv3.x，已经在opencv3.2版本上测试通过

3 运行
      安装依赖项：(1) opencv3.x  (2) sudo apt-get install libpopt-dev
      编译: mkdir build 
            cd build 
            cmake .. 
            make  
     
      执行：
          /mono_calibration -w <标定板角点列数> -h <标定板角点行数> -s <标定板相邻角点间距离> -n <标定图像数量> -d <标定图像的路径> -i <图像名字> -f <图像格式>
          注意：如果图像路径或者名字字符串是空的话，可以写成 -i "" -d ""; 图像格式： bmp jpg png等opencv支持的格式
      例如： ./mono_calibration -w 8 -h 6 -s 30 -n 30 -d ../supernode/ -i img -f png

