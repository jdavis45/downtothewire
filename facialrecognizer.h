#ifndef FACIALRECOGNIZER_H
#define FACIALRECOGNIZER_H

#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

class FacialRecognizer
{
public:
    FacialRecognizer();

    void init();
    int train(const std::string& trainXml);
    int detect(cv::Mat& image, std::string& identifier);
    void test();

    void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';');

    cv::Ptr<cv::FaceRecognizer> model;

    std::vector<cv::Mat> _images;
    std::vector<int> _labels;
    std::map<int,std::string> _names;
    cv::Mat _testSample;
    int _testLabel;

};




#endif // FACIALRECOGNIZER_H
