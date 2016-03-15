#include<time.h>
#include<cstdio>
timespec DIFF(timespec& a,timespec& b){
    timespec ret;
    if(b.tv_nsec<a.tv_nsec){
        ret.tv_nsec=(1000000000+b.tv_nsec-a.tv_nsec);
        ret.tv_sec=(b.tv_sec-1-a.tv_sec);
    }
    else{
        ret.tv_nsec=(b.tv_nsec-a.tv_nsec);
        ret.tv_sec=(b.tv_sec-a.tv_sec);
    }
    return ret;
}

struct TimeMachine{
	timespec Start,End,Temp;
	TimeMachine(){}
	void begin(){
		clock_gettime(CLOCK_MONOTONIC,&Start);
	}
	void end(){
		clock_gettime(CLOCK_MONOTONIC,&End);
	}
	void print_time(){
		Temp=DIFF(Start,End);
		printf("%d.%09d\n",(int)Temp.tv_sec,(int)Temp.tv_nsec);
	}
};
