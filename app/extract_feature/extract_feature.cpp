#include<caffe/classifier.hpp>
#include "DIY_time.h"
#include <fstream>
using std::string;
using namespace caffe;
int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto file" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  caffe::Classifier classifier(model_file, trained_file, mean_file);
  std::ifstream fin(argv[4]);
  string file;
  std::vector<string> featureNames;
  const int kMaxKeyStrLength=10;
  char key_str[kMaxKeyStrLength];

  for(int i=5;i<argc;i++)featureNames.push_back(argv[i]);
  int num_features=featureNames.size();
  std::vector<int> image_indices(num_features,0);
  bool beginFlag=false;
  TimeMachine TIM;
  std::ofstream fout;
  while(fin>>file){
      std::cout << "---------- Prediction for "
                << file << " ----------" << std::endl;
      TIM.begin();
      cv::Mat img = cv::imread(file, -1);
      TIM.end();
      TIM.print_time();
      CHECK(!img.empty()) << "Unable to decode image " << file;
      TIM.begin();
      std::vector<Prediction> predictions = classifier.Classify(img);
      TIM.end();
      TIM.print_time();
      for (int bat = 0; bat < num_features; ++bat) {
          if(beginFlag)
              fout.open((featureNames[bat]+".out").c_str(),std::fstream::app);
          else
              fout.open((featureNames[bat]+".out").c_str()),beginFlag=true;
          const boost::shared_ptr<std::vector<float> > feature = classifier.getFeature(featureNames[bat]);


          snprintf(key_str, kMaxKeyStrLength, "%06d",image_indices[bat]+1);
          float temp=0;
          fout<<key_str<<" "<<feature->size()<<std::endl;
          for(int i=0;i<feature->size();i++){

              temp+=feature->at(i);
              snprintf(key_str,kMaxKeyStrLength,"%2.5f",feature->at(i));
              fout<<key_str<<" ";

          }fout<<'\n';
          fout<<temp<<std::endl;
          std::string a;

          ++image_indices[bat];
          if (image_indices[bat] % 1000 == 0) {
              LOG(ERROR)<< "Extracted features of " << image_indices[bat] <<
              " query images for feature blob " << featureNames[bat];
          }

          fout.close();
      }  // for (int i = 0; i < num_features; ++i)


  }
  fin.close();
}
