
extern "C"
{
#include "RaspiStill.h"
}

#include "facialrecognizer.h"

int main(int argc, char *argv[])
{
	FacialRecognizer* rec = 0;
	rec = new FacialRecognizer();
    rec->train("facerec_custom.txt");

	rec->test();
    
}
