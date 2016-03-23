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

public:
    Picture(){}
    Picture(int num):num(num){
        toppic.clear();
        canChose.clear();
    }
    void prepare(){
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
};
typedef boost::shared_ptr<Picture> PicturePtr;


bool oula = true;

class Features{
public:

    Features(){
        features.clear();
        _maxPoint=-1;
    }
    int maxPoint(){return _maxPoint;}
    float get_dist(int x,int y){
        if(x==_maxPoint||y==_maxPoint)return inf;
        if(oula)return calculate_oula(x,y);
        return calculate_manhadun(x,y);
    }
    float calculate_oula(int x,int y){
        int ret=0;
        int feature_len=features[0]->size();
        for(int i=0;i<feature_len;i++){
            int delta=features[x]->at(x)-features[y]->at(y);
            ret+=delta*delta;
        }
        return sqrt(ret);
    }

    float calculate_manhadun(int x,int y){
        int ret=0;
        int feature_len=features[0]->size();
        for(int i=0;i<feature_len;i++){
            int delta=features[x]->at(x)-features[y]->at(y);
            ret+=fabs(delta);
        }
        return ret;
    }
    void push_back(vecf_ptr tp){
        features.push_back(tp);
    }

private:

    std::vector<vecf_ptr> features;
    int _maxPoint;

};

class CloseLoopDetecter{
public:
    CloseLoopDetecter(){}
    CloseLoopDetecter(string cnnNetName,string meanFile,string cnnNetParameter){
        classifer=boost::shared_ptr<Classifier>(
                    new Classifier(cnnNetName,cnnNetParameter,meanFile));
        features=boost::shared_ptr<Features>(new Features());
    }
    void get_closest_point(vecf_ptr featurePtr){

        int num=pictures.size();
        features->push_back(featurePtr);
        pictures.push_back(PicturePtr(new Picture(num)));
        pictures[num]->push(features->maxPoint(),inf);

        int delta=80;
        for(int j=0;j+delta<num;j++){
            if(pictures[num-1]->CANCHOSE(j))
            pictures[num]->push(j,features->get_dist(num,j));
        }
        pictures[num]->prepare();

    }
private:
    boost::shared_ptr<Classifier> classifer;
    boost::shared_ptr<Features> features;
    std::vector<PicturePtr> pictures;

};

int main(){
	

	return 0;
}

