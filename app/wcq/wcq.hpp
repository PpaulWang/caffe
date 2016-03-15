/*
 * wcq.hpp
 *
 *  Created on: Dec 10, 2015
 *      Author: wcq
 */

#ifndef TOOLS_WCQ_HPP_
#define TOOLS_WCQ_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using namespace cv;

namespace wcq{
	void print_blob(caffe::Blob<float>* blob){
		std::ofstream fout("out.txt");
		for(int i=0;i<blob->num();i++){
			for(int j=0;j<blob->channels();j++){
				for(int k=0;k<blob->height();k++){
					for(int w=0;w<blob->width();w++){
						fout<<blob->data_at(i,j,k,w)<<" ";
					}fout<<std::endl;
				}fout<<std::endl;
			}fout<<std::endl;
		}
		fout.close();
	}
    void Resize(Mat &image,int row,int col){
        Size dsize=Size(row,col);
        Mat image2=Mat(dsize,CV_32S);
        resize(image,image2,dsize);
        image=image2;
    }
    void submean(std::vector<std::vector<std::vector<float> > > &mean,Mat &image){
        Resize(image,mean[0].size(),mean[0][0].size());

        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                for(int k=0;k<3;k++){
                    image.at<Vec<uchar,3> >(i,j)[k]-=mean[k][i][j];
                }
            }
        }
    }

	//void
	void image2blob(caffe::Blob<float>* blob,std::vector<std::string>& filename,int st1=0,int st2=1,int st3=2){
		BlobProto temp;
		std::vector<std::vector<std::vector<float> > > mean_val(3);
		std::ifstream ffin("mean.txt");
		for(int i=0;i<mean_val.size();i++){
			mean_val[i].resize(256);
			for(int j=0;j<mean_val[i].size();j++){
				mean_val[i][j].resize(256);
				for(int k=0;k<mean_val[i][j].size();k++){
					ffin>>mean_val[i][j][k];
					//std::cout<<mean_val[i][j][k]<<std::endl;
				}
			}
		}
		ffin.close();
		temp.Clear();
		temp.set_num(blob->num());
		temp.set_channels(blob->channels());
		temp.set_height(blob->height());
		temp.set_width(blob->width());
		int height=blob->height();
		int width=blob->width();
		for(int i=0;i<blob->num();i++){
            Mat img =imread(filename[i],CV_LOAD_IMAGE_COLOR);\
			LOG(ERROR)
			<<"[loading... ] "<<filename[i];
            submean(mean_val,img);

            Resize(img,height,width);
            int st[3]={st1,st2,st3};
			for(int k=0;k<3;k++){
				for(int i=0;i<height;i++)
                for(int j=0;j<width;j++){
                    temp.add_data((float)img.at<Vec<uchar,3> >(i,j)[st[k]]);
				}
            }
		}
		blob->FromProto(temp,0);
		//print_blob(blob);
		return ;
	}


}

#endif /* TOOLS_WCQ_HPP_ */
