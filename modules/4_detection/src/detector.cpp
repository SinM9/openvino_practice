#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <inference_engine.hpp>

using namespace cv;
using namespace InferenceEngine;

Detector::Detector() {
    Core ie;

    // Load deep learning network into memory
    auto net = ie.ReadNetwork(utils::fs::join(DATA_FOLDER, "face-detection-0104.xml"),
                              utils::fs::join(DATA_FOLDER, "face-detection-0104.bin"));
    InputInfo::Ptr inputInfo = net.getInputsInfo()["image"];
    inputInfo->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo->setLayout(Layout::NHWC);
    inputInfo->setPrecision(Precision::U8);
    outputName = net.getOutputsInfo().begin()->first;

    // Initialize runnable object on CPU device
    ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");

    // Create a single processing thread
    req = execNet.CreateInferRequest();
}


void Detector::detect(const cv::Mat& image,
                      float nmsThreshold,
                      float probThreshold,
                      std::vector<cv::Rect>& boxes,
                      std::vector<float>& probabilities,
                      std::vector<unsigned>& classes) {
    std::vector<size_t> dims = {1, (size_t)image.channels(), (size_t)image.rows, (size_t)image.cols};
    Blob::Ptr input = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, dims, Layout::NHWC), image.data);

    req.SetBlob("image", input);
    req.Infer();

    float* output = req.GetBlob(outputName)->buffer();
    int size = req.GetBlob(outputName)->size()/7;

    int indx;
    for (int i = 0; i < size; i = ++i) {
        indx = i * 7;
        if (output[indx + 2] > probThreshold) {
            int xmin = static_cast<int>(output[indx + 3] * image.cols);
            int ymin = static_cast<int>(output[indx + 4] * image.rows);
            int xmax = static_cast<int>(output[indx + 5] * image.cols);
            int ymax = static_cast<int>(output[indx + 6] * image.rows);
            Rect rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            boxes.push_back(rect);

            probabilities.push_back(output[indx + 2]);
            classes.push_back(output[indx + 1]);
        }
    }
    std::vector<unsigned> indices;
    nms(boxes, probabilities, nmsThreshold, indices);
    size = boxes.size();

    int k, j= 0;
    int dist;
    for (int i = 0; i < size; ++i) {
        if (indices[k] != i) {
            dist = i - j;
            boxes.erase(boxes.begin() + dist);
            probabilities.erase(probabilities.begin() + dist);
            classes.erase(classes.begin() + dist);
            j++;
        } else {
            k++;
        }
    }
 }

void nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& probabilities,
         float threshold, std::vector<unsigned>& indices){

    size_t n = boxes.size();
    std::set<size_t> remainingIndices;
    for (size_t i = 0; i < n; i++) {
        remainingIndices.insert(i);
    }
    while (!remainingIndices.empty()) {
        size_t indMaxProb = 0;
        float maxProb = 0.0f;
        for (auto i : remainingIndices) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                indMaxProb = i;
            }
        }

        remainingIndices.erase(indMaxProb);
        indices.push_back(indMaxProb);

        for (auto i : remainingIndices) {
            if (iou(boxes[indMaxProb], boxes[i]) > threshold) {
                remainingIndices.erase(i);
            }
        }
    }
}	

float iou(const cv::Rect& a, const cv::Rect& b) {
    float Intersection = (a & b).area();
    float Union = a.area() + b.area();
    return Intersection / (Union - Intersection);
}
