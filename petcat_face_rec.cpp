#include <opencv2/opencv.hpp>
#include <opencv2/face/facerec.hpp>
#include <iostream>
#include <experimental/filesystem>
#include <string.h>

namespace fs = std::experimental::filesystem;

void faceDetection(std::vector<cv::Mat>& images) {
    cv::CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml");

    for(auto& image : images) {
        cv::Mat grey;
        cv::cvtColor(image, grey, CV_BGR2GRAY);
        cv::equalizeHist(grey, grey);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(grey, faces, 1.3, 3, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
        if(faces.size()>0) {
            cv::Mat face_roi = grey(faces[0]);
            image = face_roi;
        }
    }
}

void prepareTrainingData(std::vector<cv::Mat>& images, std::vector<int> labels, fs::path path_to_show) {
    cv::Mat temp;

    for(const auto& entry : fs::directory_iterator(path_to_show)) {
        const auto dirname = entry.path();
        fs::path p(dirname);
        std::error_code ec;
        int l=0;
        if(fs::is_directory(p,ec)) {
            const auto dname = entry.path().filename().string();
            std::string sub = dname.substr(1);
            int label = std::stoi(sub) -1;
            
            for(const auto& dir_entry : fs::directory_iterator(p)) {
                const auto img_name = dir_entry.path().string();
                temp = cv::imread(img_name);
                images.push_back(temp);
                labels.push_back(l);
            }
        }
    }
}   

int main(int argc, char** argv)
{
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    fs::path path_to_show("./training");
    prepareTrainingData(images, labels, path_to_show);
    faceDetection(images);

    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::createLBPHFaceRecognizer(1,8,8,8,10.0);
    model->train(images, labels);
    model->save("cat_faces.xml");

    int size = images.size();
    int label =0;
    std::string s;
    for(auto& image : images) {
        s = std::to_string(label);
        cv::imshow(s, image);
        cv::waitKey(0);
        cv::destroyWindow(s);
    }
    return 0;
}