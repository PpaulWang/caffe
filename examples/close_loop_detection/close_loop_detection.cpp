#include<caffe/classifier.hpp>
#include<iostream>
#include<cstring>
#include<algorithm>
#include<string>
#include<boost/shared_ptr.hpp>


using namespace caffe;
using std::string;


typedef std::vector<float> vecf;
typedef boost::shared_ptr<vecf> vecf_ptr;
//const int N=100000;


const int numberN=10;
const int L=3;
const int perIter=300;
const float inf=1e9;


class Picture{

private:


    class topPic{
    private:
        int id;
        float dist;
    public:
        topPic(){}
        topPic(int id,float dist):id(id),dist(dist){}
        bool operator <(const topPic & topPi)const{
            return dist<topPi.dist;
        }
        int ID(){
            return id;
        }
        float dis(){
            return dist;
        }
    };


    int topone;
    int num;
    double dist;
    std::vector<bool> canChose;
    std::vector<topPic> toppic;
    vecf_ptr _feature;

public:
    Picture(){}
    Picture(int num,vecf_ptr feature):num(num),_feature(feature){
        toppic.clear();
        canChose.clear();
    }
    void process(){
        canChose.resize(num,0);
        int len=std::min<int>(numberN,(int)toppic.size());
        partial_sort(toppic.begin(),toppic.begin()+len,toppic.end());
        for(int i=0;i<len;i++){
            SET(toppic[i].ID()-L,toppic[i].ID()+L);
        }
        topone=toppic[0].ID();
        dist=toppic[0].dis();
        if(toppic.size()<numberN||num%perIter==0){
            canChose.resize(num,1);
        }
    }
    void SET(int l,int r){
        l=std::max(0,l);
        r=std::min(num-1,r);
        for(int i=l;i<=r;i++)canChose[i]=1;
    }
    void push(int _id,float _dist){
        toppic.push_back(topPic(_id,_dist));
    }
    bool CANCHOSE(int id){
        if(id>=num)return false;
        return canChose[id];
    }
    vecf_ptr feature(){
        return _feature;
    }
    int closestPicture(){
        return topone;
    }
};
typedef boost::shared_ptr<Picture> PicturePtr;


bool oula = true;

class Features{

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

    Features(){
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
        vecf_ptr featureX=pictures[x]->feature();
        vecf_ptr featureY=pictures[y]->feature();
        int feature_len=featureX->size();
        for(int i=0;i<feature_len;i++){
            float delta=featureX->at(x)-featureY->at(y);
            ret+=delta*delta;
        }
        return sqrt(ret);
    }

    float calculate_manhadun(int x,int y){
        float ret=0;
        vecf_ptr featureX=pictures[x]->feature();
        vecf_ptr featureY=pictures[y]->feature();
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
    boost::shared_ptr<Features> features;


public:
    CloseLoopDetecter(){}
    CloseLoopDetecter(string cnnNetName,string meanFile,string cnnNetParameter){
        classifer=boost::shared_ptr<Classifier>(
                    new Classifier(cnnNetName,cnnNetParameter,meanFile));
        features=boost::shared_ptr<Features>(new Features());
    }
    int get_closest_point(vecf_ptr featurePtr){

        int num=features->size();
        features->add(PicturePtr(new Picture(num,featurePtr)));

        int delta=80;
        for(int j=0;j+delta<num;j++){
            if(num&&features->pictureAt(num-1)->CANCHOSE(j))
            features->pictureAt(num)->push(j,features->get_dist(num,j));
        }
        features->pictureAt(num)->process();
        return features->pictureAt(num)->closestPicture();
    }

};

int main(){
    boost::shared_ptr<CloseLoopDetecter> closeLoopDetecter(new CloseLoopDetecter());

	return 0;
}

