// Harris2.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<opencv2\opencv.hpp>
#include <iostream>
//#include <opencv2\imgproc\types_c.h> 

using namespace cv;
using namespace std;

//全局变量
Mat src, src_gray;
double myHarrisThrehold = 10000000000;
double myHarrisThrehold_Gauss = 1000000000000;
Mat harrisResponse(Mat& ixx, Mat& iyy, Mat& ixy, int wsize);
void mixP(Mat& point, Mat& img, int psize);
Mat LocalMaxValue(Mat& img, int wsize);
Mat computeImage(Mat& ix, Mat& iy, int wsize, int para);
void sobelGradient(Mat& img, Mat& dst, int para);
void myHarrisCorner(Mat& srcImg);
void myHarrisCorner_Gauss(Mat& srcImg);


int main()
{
	src = imread("1.jpg", 1);
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage:  <Input image>" << endl;
		return -1;
	}
	namedWindow("src_image",1);
	imshow("src_image", src);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	imshow("src_gray", src_gray);
	myHarrisCorner(src);
	myHarrisCorner_Gauss(src);
	waitKey(0);
	return 0;
}

void myHarrisCorner(Mat& srcImg) {
	Mat image, src_grayImg, Ix, Iy, I_xx, I_yy, I_xy, R, filter_R, result;
	cvtColor(srcImg, src_grayImg, COLOR_BGR2GRAY);
	image = srcImg.clone();

	int wsize = 3;//窗口大小

	sobelGradient(src_grayImg, Ix, 1);

	sobelGradient(src_grayImg, Iy, 2);

	I_xx = computeImage(Ix, Iy, wsize, 1);

	I_yy = computeImage(Ix, Iy, wsize, 2);

	I_xy = computeImage(Ix, Iy, wsize, 4);

	//compute the R value
	R = computeImage(Ix, Iy, wsize, 3);

	filter_R = LocalMaxValue(R, 10);
	imshow("fr_NoGauss", filter_R);

	mixP(filter_R, image, 2);
	imshow("NoGauss", image);

}

void myHarrisCorner_Gauss(Mat& srcImg) {
	Mat image, src_grayImg, Ix, Iy, I_xx, I_yy, I_xy, Gaussian_xx, Gaussian_yy, Gaussian_xy, R, filter_R, result;
	cvtColor(srcImg, src_grayImg, COLOR_BGR2GRAY);
	image = srcImg.clone();

	int wsize = 3;//窗口大小

	sobelGradient(src_grayImg, Ix, 1);

	sobelGradient(src_grayImg, Iy, 2);

	I_xx = computeImage(Ix, Iy, wsize, 1);
	GaussianBlur(I_xx, Gaussian_xx, Size(3, 3), 0, 0);

	I_yy = computeImage(Ix, Iy, wsize, 2);
	GaussianBlur(I_yy, Gaussian_yy, Size(3, 3), 0, 0);

	I_xy = computeImage(Ix, Iy, wsize, 4);
	GaussianBlur(I_xy, Gaussian_xy, Size(3, 3), 0, 0);
	//compute the R value
	R = harrisResponse(Gaussian_xx, Gaussian_yy, Gaussian_xy, wsize);

	filter_R = LocalMaxValue(R, 10);
	//imshow("响应图R", R);
	imshow("fr_Gauss", filter_R);

	mixP(filter_R, image, 2);
	imshow("Gauss", image);
}

void sobelGradient(Mat& img, Mat& dst, int para) {
	dst = Mat::zeros(img.size(), CV_64FC1);

	if (para == 1)
		Sobel(img, dst, CV_64FC1, 0, 1, 3);
	else if (para == 2)
		Sobel(img, dst, CV_64FC1, 1, 0, 3);
}

Mat computeImage(Mat& ix, Mat& iy, int wsize, int para) {

	Mat I_xx, I_yy, I_xy, r;
	I_xx = Mat::zeros(ix.size(), CV_64FC1);
	I_yy = Mat::zeros(ix.size(), CV_64FC1);
	r = Mat::zeros(ix.size(), CV_64FC1);
	I_xy = Mat::zeros(ix.size(), CV_64FC1);

	for (int i = wsize / 2; i < (ix.rows - wsize / 2); i++)
		for (int j = wsize / 2; j < (ix.cols - wsize / 2); j++) {
			//compute A B C, A = Ix * Ix, B = Iy * Iy, C = Ix * Iy
			double A = 0;
			double B = 0;
			double C = 0;
			for (int ii = i - wsize / 2; ii <= (i + wsize / 2); ii++)
				for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++) {
					double xx = ix.at<double>(ii, jj);
					double yy = iy.at<double>(ii, jj);
					A += xx * xx;
					B += yy * yy;
					C += xx * yy;
				}
			double p = A + B;
			double q = A * B - C * C;

			I_xx.at<double>(i, j) = A;
			I_yy.at<double>(i, j) = B;
			I_xy.at<double>(i, j) = C;
			double rr = q - 0.06 * p * p;

			if (rr > myHarrisThrehold) {
				r.at<double>(i, j) = rr;
			}

		}
	switch (para) {

	case 1: return I_xx; break;
	case 2: return I_yy; break;
	case 3: return r; break;
	case 4: return I_xy; break;
	}

}

Mat LocalMaxValue(Mat& img, int wsize) {
	Mat result;
	result = Mat::zeros(img.size(), CV_64F);

	//find local maxima of R
	for (int i = wsize / 2; i < (img.rows - wsize / 2); i++)
		for (int j = wsize / 2; j < (img.cols - wsize / 2); j++) {
			double origin = img.at<double>(i, j);
			bool found = false;
			for (int ii = i - wsize / 2; ii <= (i + wsize / 2) && found == false; ii++)
				for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++)
					if (origin < img.at<double>(ii, jj)) {
						origin = 0;
						found = true;
						break;
					}
			if (origin == 0)
				result.at<double>(i, j) = 0;
			else
				result.at<double>(i, j) = 255;
		}

	return result;
}

void mixP(Mat& point, Mat& img, int psize) {

	for (int i = psize; i < img.rows - psize; i++)
		for (int j = psize; j < img.cols - psize; j++) {
			if (point.at<double>(i, j) != 0) {
				for (int ii = i - psize; ii <= i + psize; ii++)
					for (int jj = j - psize; jj <= j + psize; jj++) {
						img.at<Vec3b>(ii, jj)[0] = 0;
						img.at<Vec3b>(ii, jj)[1] = 0;
						img.at<Vec3b>(ii, jj)[2] = 255;

					}
			}
		}
}

Mat harrisResponse(Mat& ixx, Mat& iyy, Mat& ixy, int wsize) {

	Mat result;
	result = Mat::zeros(ixx.size(), CV_64FC1);


	for (int i = wsize / 2; i < (ixx.rows - wsize / 2); i++)
		for (int j = wsize / 2; j < (ixx.cols - wsize / 2); j++) {
			//compute A B C, A = Ix * Ix, B = Iy * Iy, C = Ix * Iy
			double A = 0;
			double B = 0;
			double C = 0;
			for (int ii = i - wsize / 2; ii <= (i + wsize / 2); ii++)
				for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++) {
					double xx = ixx.at<double>(ii, jj);
					double yy = iyy.at<double>(ii, jj);
					double xy = ixy.at<double>(ii, jj);
					A += xx;
					B += yy;
					C += xy;
				}
			double p = A + B;
			double det = A * B - C * C;

			double rr = det - 0.06 * p * p;
			//result1.at<double>(i, j) = rr;
			
			if (rr > myHarrisThrehold_Gauss) {
				result.at<double>(i, j) = rr;
			}

		}
	//imshow("R", result1);
	return result;
}