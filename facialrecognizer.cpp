
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include "facialrecognizer.h"

using namespace cv;
using namespace std;

FacialRecognizer::FacialRecognizer()
{
    _names[10] = "Tim";
    _names[11] = "John";
    _names[1] = "Bryan";
    _names[2] = "Ryan";
    _names[3] = "John";
    _names[4] = "Dave";
    _names[5] = "Mike";
    _names[6] = "Tom";
    _names[7] = "Jan";
    _names[8] = "Tiny";
    _names[9] = "Jorge";

}

void FacialRecognizer::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int FacialRecognizer::train(const std::string& trainXml)
{
    // Get the path to your CSV.

    // These vectors hold the images and corresponding labels.
    //vector<Mat> images;
    //vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(trainXml, _images, _labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << trainXml << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(_images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = _images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    _testSample = _images[_images.size() - 1];
    _testLabel = _labels[_labels.size() - 1];

    _images.pop_back();
    _labels.pop_back();
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      cv::createEigenFaceRecognizer(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidennce threshold, call it with:
    //
    //      cv::createEigenFaceRecognizer(10, 123.0);
    //
    model = createEigenFaceRecognizer();
   // model = createLBPHFaceRecognizer();
   // model = createFisherFaceRecognizer();
    model->train(_images, _labels);
    // The following line predicts the label of a given
    // test image:
}

int FacialRecognizer::detect(Mat &image, string &identifier)
{
    double confidence = 0;
    int predictedLabel = -1;
    model->predict(image,predictedLabel,confidence);
    printf("label %d conf %f\n",predictedLabel,confidence);
    if(_names.find(predictedLabel) != _names.end())
    {
        if (confidence<10)
        {
            identifier = "Unknown";
        }
        else
        {
            identifier = _names[predictedLabel];
        }
    }
    else
    {
        identifier = "None";
    }

}

void FacialRecognizer::test()
{
    std::string id;
    detect(_testSample,id);
    printf("found %s\n",id.c_str());
}

/*static Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}*/

