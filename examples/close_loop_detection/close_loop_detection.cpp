#include<caffe/classifier.hpp>
#include<iostream>
#include<fstream>
#include<cstring>
#include<algorithm>
#include<string>
#include<map>
#include<boost/shared_ptr.hpp>


using namespace caffe;
using std::string;


typedef std::vector<float> vecf;
typedef boost::shared_ptr<vecf> vecfPtr;
//const int N=100000;


const int numberN=10;
const int L=30;
const int perIter=300;
const float inf=1e9;


class Picture{

private:


    class pictureDistanceId{
    private:
        int _id;
        float dist;
    public:
        pictureDistanceId(){}
        pictureDistanceId(int id,float dist):_id(id),dist(dist){}
        bool operator <(const pictureDistanceId & topPi)const{
            return dist<topPi.dist;
        }
        int id(){
            return _id;
        }
        float dis(){
            return dist;
        }
    };


    int closestPictureId;
    int num;
    float dist;
    std::vector<bool> _canChose;
    std::vector<pictureDistanceId> pictureDisId;
    vecfPtr _feature;

public:
    Picture(){}
    Picture(int num,vecfPtr feature):num(num),_feature(feature){
        pictureDisId.clear();
        _canChose.clear();
    }
    void process(){
        _canChose.resize(num,0);
        int len=std::min<int>(numberN,(int)pictureDisId.size());
        partial_sort(pictureDisId.begin(),pictureDisId.begin()+len,pictureDisId.end());
        for(int i=0;i<len;i++){
            setCanChose(pictureDisId[i].id()-L,pictureDisId[i].id()+L);
        }
        closestPictureId=pictureDisId[0].id();
        dist=pictureDisId[0].dis();
        if(pictureDisId.size()<numberN||num%perIter==0){
            setCanChose(0,num-1);
        }
    }
    void setCanChose(int l,int r){
        l=std::max(0,l);
        r=std::min(num-1,r);
        for(int i=l;i<=r;i++)_canChose[i]=1;
    }
    void push(int _id,float _dist){
        pictureDisId.push_back(pictureDistanceId(_id,_dist));
    }
    bool canChose(int id){
        if(id>=num)return false;
        return _canChose[id];
    }
    vecfPtr feature(){
        return _feature;
    }
    int closestPicture(){
        return closestPictureId;
    }
    float dis(){
        return dist;
    }
};
typedef boost::shared_ptr<Picture> PicturePtr;


bool oula = true;

class Pictures{

private:

    std::vector<PicturePtr> pictures;
    int _maxPoint;

public:
    PicturePtr pictureAt(int idx){
        return pictures[idx];
    }
    size_t size(){
        return pictures.size();
    }

    Pictures(){
        pictures.clear();
        _maxPoint=-1;
    }
    int maxPoint(){return _maxPoint;}
    float get_dist(int x,int y){
        if(x==_maxPoint||y==_maxPoint)return inf;
        if(oula)return calculate_oula(x,y);
        return calculate_manhadun(x,y);
    }
    float calculate_oula(int x,int y){
        float ret=0;
        vecfPtr featureX=pictures[x]->feature();
        vecfPtr featureY=pictures[y]->feature();
        int feature_len=featureX->size();
        CHECK_EQ(featureX->size(),featureY->size());
        for(int i=0;i<feature_len;i++){
            float delta=featureX->at(i)-featureY->at(i);
            ret+=delta*delta;
        }
        return sqrt(ret);
    }

    float calculate_manhadun(int x,int y){
        float ret=0;
        vecfPtr featureX=pictures[x]->feature();
        vecfPtr featureY=pictures[y]->feature();
        int feature_len=featureX->size();
        for(int i=0;i<feature_len;i++){
            float delta=featureX->at(x)-featureY->at(y);
            ret+=fabs(delta);
        }
        return ret;
    }
    void add(PicturePtr tp){
        tp->push(_maxPoint,inf);
        pictures.push_back(tp);
    }

};

class CloseLoopDetecter{
private:
    boost::shared_ptr<Classifier> classifer;
    boost::shared_ptr<Pictures> pictures;


public:
    CloseLoopDetecter(){}
    CloseLoopDetecter(string cnnNetName,string cnnNetParameter,string meanFile){
        classifer=boost::shared_ptr<Classifier>(
                    new Classifier(cnnNetName,cnnNetParameter,meanFile));
        pictures=boost::shared_ptr<Pictures>(new Pictures());
    }
    int getClosestPicture(int num){

        int delta=80;
        for(int j=0;j+delta<num;j++){
            if(num&&pictures->pictureAt(num-1)->canChose(j))
            pictures->pictureAt(num)->push(j,pictures->get_dist(num,j));
        }
        pictures->pictureAt(num)->process();
        return pictures->pictureAt(num)->closestPicture();

    }
    std::pair<int,float> getClosePoint(cv::Mat img){
        classifer->Classify(img);
        int num=pictures->size();
        pictures->add(PicturePtr(new Picture(num,classifer->getFeature("fc6"))));
        int ret_first=getClosestPicture(num);
        if(ret_first==-1){
            return std::make_pair<int,float>(-1,inf);
        }
        float ret_second=pictures->pictureAt(ret_first)->dis();
        return std::make_pair(ret_first,ret_second);
    }

};

int main(int argc,char** argv){
    if(argc<5){
        std::cerr<<"usage: use "<<argv[0]<<
                   " modelprototxt parameterfile meanfile filelist storepath"<<std::endl;
        return -1;
    }
    ::google::InitGoogleLogging(argv[0]);

    string ModelFile(argv[1]);
    string ModelParameter(argv[2]);
    string MeanFile(argv[3]);
    string FileList(argv[4]);
    std::ofstream fout(argv[5]);
    boost::shared_ptr<CloseLoopDetecter> closeLoopDetecter(
                new CloseLoopDetecter(ModelFile,ModelParameter,MeanFile));
    std::ifstream fin(FileList.c_str());

    string fileName;
    std::vector<string> fileNames;
    std::map<string,int> file2Id;
    while(fin>>fileName){

        fileNames.push_back(fileName);
        file2Id[fileName]=fileNames.size()-1;
        cv::Mat img=cv::imread(fileName,-1);
        std::pair<int,float> closePictureInfo=closeLoopDetecter->getClosePoint(img);
        std::cout<<fileName<<" "<<fileNames.size()-1<<" "<<closePictureInfo.first<<std::endl;
        if(~closePictureInfo.first)
            fout<<fileName<<" "<<fileNames.size()-1<<" "<<
                  fileNames[closePictureInfo.first]<<" "<<closePictureInfo.first<<
                  " "<<closePictureInfo.second<<std::endl;
        else
            fout<<fileName<<" "<<fileNames.size()-1<<" "<<
                  "None -1 "<<inf<<std::endl;
    }
    fin.close();
    fout.close();
	return 0;
}

