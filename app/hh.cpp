#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
#include<vector>
//
using namespace std;
const int N=3000;
vector<string> str;

int main(){
    freopen("pic_list.txt","r",stdin);
	freopen("out.txt","w",stdout);
	string tp;
	while(cin>>tp){
        str.push_back("/home/wcq/Documents/caffe-master/examples/images1/"+tp);
	}
	sort(str.begin(),str.end());
	unique(str.begin(),str.end());
	for(vector<string>::iterator i=str.begin();i!=str.end();++i)
		cout<<*i<<endl;
	cout<<str.size()<<endl;
	return 0;
}
