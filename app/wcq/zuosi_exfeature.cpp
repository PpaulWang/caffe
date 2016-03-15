/*
 * zuosi_exfeature.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: wcq
 */

/*
 * ex_extract_features.cpp
 *
 *  Created on: Dec 9, 2015
 *      Author: wcq
 */
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "wcq.hpp"
#include "/usr/include/opencv2/highgui/highgui.hpp"

using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;


template<typename Dtype>
int feature_extraction_pipeline(std::string pretrained_binary_proto,std::string feature_extraction_proto,
		std::vector<std::string> blob_names,std::vector<std::string> dataset_names,
		int num_mini_batches
		, char** argv,char *file_list);




int main(int argc, char** argv) {
	if(argc<3){
		LOG(ERROR)<<
		"config storepath";
		return 1;
	}


	std::ifstream fin(argv[1]);
	std::cout<<"using CPU \n";
	std::string model_name,net_name;
	int feature_num;
	std::vector<std::string> blob_names;
	std::vector<std::string> dataset_names;
	char file_list[100];
	int num_mini_batches;
	fin>>model_name;
	fin>>net_name;
	std::cout<<model_name<<" "<<net_name<<std::endl;
	fin>>feature_num;
	blob_names.resize(feature_num);
	dataset_names.resize(feature_num);
	for(int i=0;i<feature_num;i++){
		fin>>blob_names[i];
		fin>>dataset_names[i];
	}
	fin>>num_mini_batches;
	fin>>file_list;
	fin.close();

	return feature_extraction_pipeline<float>(model_name,net_name,blob_names,dataset_names,num_mini_batches,argv,file_list);

}


template<typename Dtype>
int feature_extraction_pipeline(std::string pretrained_binary_proto,std::string feature_extraction_proto,
		std::vector<std::string> blob_names,std::vector<std::string> dataset_names,
		int num_mini_batches
		, char** argv,char* file_list) {
	::google::InitGoogleLogging(argv[0]);
	int len=strlen(argv[2]);
	std::string store_path(argv[2],len);

	std::ofstream fout;

	Caffe::set_mode(Caffe::CPU);

	shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
	std::cout<< feature_extraction_net->num_inputs() << " " << feature_extraction_net->num_outputs()  << std::endl;

	size_t num_features = blob_names.size();

	for (size_t i = 0; i < num_features; i++) {
		CHECK(feature_extraction_net->has_blob(blob_names[i]))
        		<< "Unknown feature blob name " << blob_names[i]
				<< " in the network " << feature_extraction_proto;
	}
	const int input_num=1;
	const int kMaxKeyStrLength = 100;
	char key_str[kMaxKeyStrLength];
	std::vector<Blob<float>*> input_vec(1);
	input_vec[0]=new Blob<float>;
	std::vector<int> image_indices(num_features, 0);
	std::ifstream fin(file_list);
	std::vector<std::string> filenames(input_num);
	LOG(ERROR)<< "Extracting features ...";
	for (int batch_index = 0; batch_index < num_mini_batches ; ++batch_index) {
		input_vec[0]->Reshape(input_num,3,227,227);
		for(int i=0;i<input_num;i++)fin>>filenames[i];
        wcq::image2blob(input_vec[0],filenames);
		feature_extraction_net->Forward(input_vec);
		for (int i = 0; i < num_features; ++i) {
			if(batch_index)
				fout.open((store_path+blob_names[i]+".out").c_str(),std::fstream::app);
			else
				fout.open((store_path+blob_names[i]+".out").c_str());
			const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);
			int batch_size = feature_blob->num();
			int Height=feature_blob->height();
			int Weith=feature_blob->width();
			int Channels=feature_blob->channels();
			int number=Height*Weith*Channels;
			for (int n = 0; n < batch_size; ++n) {
				snprintf(key_str, kMaxKeyStrLength, "%06d",image_indices[i]+1);
				Dtype temp=0;
				fout<<key_str<<" "<<number<<std::endl;
				for(int c=0;c<Channels;c++){
					for(int h=0;h<Height;h++){
						for(int w=0;w<Weith;w++){
							temp+=feature_blob->data_at(n,c,h,w);
							snprintf(key_str,kMaxKeyStrLength,"%2.5f",feature_blob->data_at(n,c,h,w));
							fout<<key_str<<" ";
						}
					}
				}fout<<'\n';
				fout<<temp<<std::endl;
				std::string a;

				++image_indices[i];
				if (image_indices[i] % 1000 == 0) {
					LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
					" query images for feature blob " << blob_names[i];
				}
			}  // for (int n = 0; n < batch_size; ++n)
			fout.close();
		}  // for (int i = 0; i < num_features; ++i)
	}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	// write the last batch
	for (int i = 0; i < num_features; ++i) {
		LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
			" query images for feature blob " << blob_names[i];
	}
	fin.close();
	LOG(ERROR)<< "Successfully extracted the features!";
	return 0;
}




