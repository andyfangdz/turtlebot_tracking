#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

using namespace cv;
using namespace std;

int Uxt=255,Vxt=255,Unt=0,Vnt=0,Yxt=255,Ynt=0;
int Ux,Un,Vx,Vn,Yx,Yn;
Mat frame,YUV;
Mat channels[3],tresh[3];
vector <Point> hits,bestHits;
void onTrackBarChange(int,void*)
{
	/*Ux=((double)Uxt)/1000-1;
	Un=((double)Unt)/1000-1;
	Vx=((double)Vxt)/1000-1;
	Vn=((double)Vnt)/1000-1;*/
	Ux=Uxt,Un=Unt,Vx=Vxt,Vn=Vnt;
	//Ux=60;Un=30;Vx=220;Vn=180;
}
void creatTrackBars()
{
	//createTrackbar("1000*(Y+1)(min)","Track Bars",&Ynt,255,onTrackBarChange);
	//createTrackbar("1000*(Y+1)(max)","Track Bars",&Yxt,255,onTrackBarChange);
	createTrackbar("1000*(U+1)(min)","Track Bars",&Unt,255,onTrackBarChange);
	createTrackbar("1000*(U+1)(max)","Track Bars",&Uxt,255,onTrackBarChange);
	createTrackbar("1000*(V+1)(min)","Track Bars",&Vnt,255,onTrackBarChange);
	createTrackbar("1000*(V+1)(max)","Track Bars",&Vxt,255,onTrackBarChange);
}
void processRange(int lowBoundU,int highBoundU,int lowBoundV,int highBoundV,Mat pro[])
{
	vector <uint8_t> xx,yy;
	hits.clear();
	bestHits.clear();
	for(int i=0;i<pro[0].rows;i++)
	{
		for(int j=0;j<pro[0].cols;j++)
		{
			//double valueY=pro[0].at<double>(i,j);
			uint8_t valueU=pro[1].at<uint8_t>(i,j);
			uint8_t valueV=pro[2].at<uint8_t>(i,j);
			//cout<<valueU<<valueV;
			if(valueU>=lowBoundU&&valueU<=highBoundU&&valueV>=lowBoundV&&valueV<=highBoundV)
			{
				Point nhit;
				nhit.x=j;
				nhit.y=i;
				hits.push_back(nhit);
			}
		}
	}
	//imshow("U Channel",pro[1]);
	//printf("%lf",&pro[0].at<double>(pro[0].rows/2,pro[0].cols/2));
	//printf("%lf",&pro[1].at<double>(pro[0].rows/2,pro[0].cols/2));
	//printf("%lf",&pro[2].at<double>(pro[0].rows/2,pro[0].cols/2));
	cout<<"Selected "<<hits.size()<<" hit(s)."<<endl;
	if(hits.size()==0)return;
	double sumx=0,sumy=0,avax,avay,sx,sy;
	for(int i=0;i<hits.size();i++)
	{
		sumx+=hits[i].x;
		sumy+=hits[i].y;
		xx.push_back(hits[i].x);
		yy.push_back(hits[i].y);
	}
	avax=sumx/pro[0].cols;
	avay=sumy/pro[0].rows;
	sumx=0,sumy=0;
	for(int i=0;i<hits.size();i++)
	{
		sumx+=(hits[i].x)*(hits[i].x);
		sumy+=hits[i].y*hits[i].y;
	}
	sumx/=pro[0].cols;
	sumy/=pro[0].rows;
	sx=sqrt(avax*avax-sumx);
	sy=sqrt(avay*avay-sumy);
	//sort(xx.begin(),xx.end());
	//sort(yy.begin(),yy.end());
	nth_element( xx.begin(), xx.begin()+xx.size()/2,xx.end() );
	nth_element( yy.begin(), yy.begin()+yy.size()/2,yy.end() );
	double medianx=xx[xx.size()/2],mediany=yy[yy.size()/2];
	cout<<"============Statistic Data=========="<<endl;
	cout<<"Standard Error for X-axis:"<<sx<<endl;
	cout<<"Standard Error for Y-axis:"<<sy<<endl;
	cout<<"Average for X-axis:"<<avax<<endl;
	cout<<"Average for Y-axis:"<<avay<<endl;
	cout<<"Square of Average for X-axis:"<<avax*avax<<endl;
	cout<<"Square of Average for Y-axis:"<<avay*avay<<endl;
	cout<<"Average of Square for X-axis:"<<sumx<<endl;
	cout<<"Average of Square for Y-axis:"<<sumy<<endl;
	cout<<"Median of X-axis:"<<medianx<<endl;
	cout<<"Median of Y-axis:"<<mediany<<endl;
	cout<<"N of X-axis:"<<pro[0].cols<<endl;
	cout<<"N of Y-axis:"<<pro[0].rows<<endl;
	cout<<"=========Statistic Data End========="<<endl;

	for(int i=0;i<hits.size();i++)
	{
		if(fabs(hits[i].x-medianx<=sx)&&fabs(hits[i].y-mediany<=sy))bestHits.push_back(hits[i]);
	}
	cout<<"Selected "<<bestHits.size()<<" best hit(s)."<<endl;
}
void drawRect()
{
	RotatedRect box = minAreaRect(Mat(bestHits));
	Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; ++i)
	{
	    line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, CV_AA);
	}
}

void getRange()
{
	VideoCapture capture(-1);
	Mat framee,f[3],tmp;
	while(true)
	{
		capture>>framee;
		cvtColor(framee, tmp, CV_RGB2YCrCb);
		//split(framee,f);
		int cn = tmp.channels();
		Scalar_<uint8_t> yuvPixel;
		yuvPixel.val[0] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 0]; // Y
		yuvPixel.val[1] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 1]; // U
		yuvPixel.val[2] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 2]; // V
		cout<<"Current Locked Y:"<<(int)yuvPixel.val[0]<<endl;
		cout<<"Current Locked U:"<<(int)yuvPixel.val[1]<<endl;
		cout<<"Current Locked V:"<<(int)yuvPixel.val[2]<<endl;
		Point v[4],ff[2];
		//ff[0].x=320;ff[0].y=0;
		//ff[1].x=320;ff[1].y=480;
		v[0].x=320;v[0].y=0;
		v[1].x=320;v[1].y=480;
		v[2].x=0;v[2].y=240;
		v[3].x=640;v[3].y=240;
		//line(framee,ff[0],ff[1],CV_RGB(255, 0, 0));
		line(framee,v[0],v[1],CV_RGB(0, 0, 255));
		line(framee,v[2],v[3],CV_RGB(0, 255, 0));
		imshow("Grab Center YUV",framee);
		if(waitKey(30) >= 0) break;
	}
	cvtColor(framee, framee, CV_RGB2YCrCb);
	split(framee,f);
	int cn = tmp.channels();
	Scalar_<uint8_t> yuvPixel;
	yuvPixel.val[0] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 0]; // Y
	yuvPixel.val[1] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 1]; // U
	yuvPixel.val[2] = tmp.data[(tmp.rows/2)*tmp.cols*cn + (tmp.cols/2)*cn + 2]; // V
	Uxt=yuvPixel.val[1]+10;;
	Unt=yuvPixel.val[1]-10;
	Vxt=yuvPixel.val[2]+10;
	Vnt=yuvPixel.val[2]-10;
	cout<<"Locked Y Range:"<<Yxt<<"--"<<Ynt<<endl;
	cout<<"Locked U Range:"<<Uxt<<"--"<<Unt<<endl;
	cout<<"Locked V Range:"<<Vxt<<"--"<<Vnt<<endl;
	waitKey();
	capture.release();
	destroyWindow("Grab Center YUV");
}
int main()
{

	getRange();
	VideoCapture capture(-1);
		if(!capture.isOpened())
		{
			cout<<"Failed to open camera."<<endl;
			return -1;
		}
	onTrackBarChange(0,0);
	namedWindow("Track Bars",1);
	creatTrackBars();
	while (true)
	{
		capture>>frame;
		cvtColor(frame, YUV, CV_RGB2YCrCb);
		split(YUV,channels);
		processRange(Un,Ux,Vn,Vx,channels);
		if(bestHits.size()>0)
		{
			drawRect();
		}
		imshow("Main View",frame);
        if(waitKey(30) >= 0) break;
	}
}
