#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>

// ------------------------------
// Global configuration
// ------------------------------
// Adjust NUM_CLASSES to match your model (e.g. COCO has 80 classes).
const int NUM_CLASSES   = 7;
const int NUM_CHANNELS  = 4 + 1 + NUM_CLASSES; // [cx, cy, log_w, log_h, obj_conf, cls0,...,clsN]
const float CONF_THRESH = 0.3f;
const float NMS_THRESH  = 0.6f; 
const std::vector<int> STRIDES = {8, 16, 32};

struct Detection {
    float bbox[4]; // [x1, y1, x2, y2]
    float obj_conf;
    float cls_conf;
    int   class_id;
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float computeIoU(const Detection &a, const Detection &b) {
    float xx1 = std::max(a.bbox[0], b.bbox[0]);
    float yy1 = std::max(a.bbox[1], b.bbox[1]);
    float xx2 = std::min(a.bbox[2], b.bbox[2]);
    float yy2 = std::min(a.bbox[3], b.bbox[3]);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    float areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    return inter / (areaA + areaB - inter);
}

// Per-class Non-Maximum Suppression (NMS)
std::vector<Detection> nms(const std::vector<Detection>& dets, float nms_thresh) {
    std::vector<Detection> result;
    // Group detections by class id.
    std::unordered_map<int, std::vector<Detection>> classMap;
    for (const auto &d : dets)
        classMap[d.class_id].push_back(d);
    
    for (auto &pair : classMap) {
        auto &dlist = pair.second;
        // Sort by score: (obj_conf * cls_conf)
        std::sort(dlist.begin(), dlist.end(), [](const Detection &a, const Detection &b) {
            return (a.obj_conf * a.cls_conf) > (b.obj_conf * b.cls_conf);
        });
        while (!dlist.empty()) {
            Detection best = dlist.front();
            result.push_back(best);
            dlist.erase(dlist.begin());
            dlist.erase(std::remove_if(dlist.begin(), dlist.end(), [&](const Detection &d) {
                return computeIoU(best, d) > nms_thresh;
            }), dlist.end());
        }
    }
    // Optionally sort all final detections by score descending.
    std::sort(result.begin(), result.end(), [](const Detection &a, const Detection &b) {
        return (a.obj_conf * a.cls_conf) > (b.obj_conf * b.cls_conf);
    });
    return result;
}

// Process the outputs from the network to extract detections.
// Here we assume each output has shape [1, NUM_CHANNELS, grid_h, grid_w]
// and the order of outputs corresponds to STRIDES order.
std::vector<Detection> processOutputs(const std::vector<cv::Mat>& outputs, int input_size) {
    std::vector<Detection> detections;

    if (outputs.size() != STRIDES.size()) {
        std::cerr << "Warning: Number of output layers (" << outputs.size() 
                  << ") does not match expected (" << STRIDES.size() << ")." << std::endl;
    }

    // Loop over each output scale.
    for (size_t i = 0; i < outputs.size(); ++i) {
        int stride = STRIDES[i];
        int grid_w = input_size / stride;
        int grid_h = input_size / stride;
        int num_cells = grid_w * grid_h;

        const cv::Mat &out = outputs[i];
        // Expected output shape is [1, NUM_CHANNELS, grid_h, grid_w].
        if (out.dims != 4 || out.size[1] != NUM_CHANNELS) {
            std::cerr << "Unexpected output dimensions in layer " << i << std::endl;
            continue;
        }
        const float* data = reinterpret_cast<const float*>(out.data);

        // Loop over each grid cell.
        for (int idx = 0; idx < num_cells; idx++) {
            float cx    = data[0 * num_cells + idx];
            float cy    = data[1 * num_cells + idx];
            float log_w = data[2 * num_cells + idx];
            float log_h = data[3 * num_cells + idx];
            float obj_conf = sigmoid(data[4 * num_cells + idx]);

            float max_cls = -1.0f;
            int cls_id = -1;
            // Examine class scores.
            for (int c = 5; c < 5 + NUM_CLASSES; ++c) {
                float cls_score = sigmoid(data[c * num_cells + idx]);
                if (cls_score > max_cls) {
                    max_cls = cls_score;
                    cls_id = c - 5;
                }
            }

            float score = obj_conf * max_cls;
            if (score < CONF_THRESH)
                continue;

            int x_idx = idx % grid_w;
            int y_idx = idx / grid_w;
            float adjusted_cx = (cx + x_idx) * stride;
            float adjusted_cy = (cy + y_idx) * stride;
            float w = std::exp(log_w) * stride;
            float h = std::exp(log_h) * stride;

            // Convert center coordinates to bounding box coordinates.
            float x1 = adjusted_cx - w / 2.0f;
            float y1 = adjusted_cy - h / 2.0f;
            float x2 = adjusted_cx + w / 2.0f;
            float y2 = adjusted_cy + h / 2.0f;

            Detection det;
            det.bbox[0] = x1;
            det.bbox[1] = y1;
            det.bbox[2] = x2;
            det.bbox[3] = y2;
            det.obj_conf = obj_conf;
            det.cls_conf = max_cls;
            det.class_id = cls_id;
            detections.push_back(det);
        }
    }
    return detections;
}

cv::Mat preprocessImage(const cv::Mat &img, int input_size) {
    // Resize input image to (input_size x input_size)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));

    // Convert from BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and scale to [0, 1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // Normalize the image using mean and std
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    float std_vals[3]  = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_vals[i]) / std_vals[i];
    }
    cv::merge(channels, rgb);

    // cv::Mat vis;
    // // This scales the minimum value in rgb to 0 and the maximum to 255.
    // cv::normalize(rgb, vis, 0, 255, cv::NORM_MINMAX);
    // // Convert to 8-bit so PNG can display it properly
    // vis.convertTo(vis, CV_8U);
    // // Convert back to BGR for standard image viewing (optional)
    // cv::cvtColor(vis, vis, cv::COLOR_RGB2BGR);
    
    // std::string normalizedOutputPath = "normalized_image.png";
    // if (!cv::imwrite(normalizedOutputPath, vis)) {
    //     std::cerr << "Failed to save normalized image to " << normalizedOutputPath << std::endl;
    // } else {
    //     std::cout << "Normalized image saved to " << normalizedOutputPath << std::endl;
    // }
    // ------------------------------------------------------------------------

    // Create a 4D blob (shape: [1, 3, input_size, input_size])
    cv::Mat blob = cv::dnn::blobFromImage(rgb);
    return blob;
}


// Draw detection boxes and scores on the image.
void drawDetections(cv::Mat &image, const std::vector<Detection> &detections, float conf_thresh = 0.5f) {
    for (const auto &det : detections) {
        float finalScore = det.obj_conf * det.cls_conf;
        if (finalScore < conf_thresh)
            continue;
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);
        cv::putText(image, cv::format("%.2f", finalScore), 
                    cv::Point(x1, std::max(y1 - 10, 0)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path-to-onnx-model> <path-to-image> [input_size]" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    int input_size = 512; // default input size
    if (argc > 3)
        input_size = std::stoi(argv[3]);

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    // For visualization, resize without normalization.
    cv::Mat visImg;
    cv::resize(image, visImg, cv::Size(input_size, input_size));

    // Preprocess image (normalization & blob creation)
    cv::Mat blob = preprocessImage(image, input_size);

    // Load ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Failed to load ONNX model: " << modelPath << std::endl;
        return -1;
    }
    net.setInput(blob);

    // Obtain output layer names and run forward pass.
    std::vector<std::string> outputNames = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    net.forward(outputs, outputNames);

    // Process outputs to extract raw detections.
    std::vector<Detection> detections = processOutputs(outputs, input_size);

    // Apply NMS.
    std::vector<Detection> finalDetections = nms(detections, NMS_THRESH);

    // Print detections.
    std::cout << "Final Detections:" << std::endl;
    for (const auto &det : finalDetections) {
        float finalScore = det.obj_conf * det.cls_conf;
        std::cout << "Class: " << det.class_id << ", Score: " << finalScore
                  << ", BBox: [" << det.bbox[0] << ", " << det.bbox[1]
                  << ", " << det.bbox[2] << ", " << det.bbox[3] << "]" << std::endl;
    }

    // Draw detections on the resized image.
    drawDetections(visImg, finalDetections, 0.5f);

    // Save result image with detections drawn.
    std::string outFile = "detection_result.jpg";
    if (!cv::imwrite(outFile, visImg)) {
        std::cerr << "Failed to save detection result to " << outFile << std::endl;
        return -1;
    }
    std::cout << "Detection result saved to " << outFile << std::endl;

    return 0;
} 