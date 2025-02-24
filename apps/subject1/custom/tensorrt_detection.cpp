#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cassert>
#include <chrono>

// TensorRT and CUDA includes
#include "NvInfer.h"
#include "cuda_runtime_api.h"

// Macro for CUDA error checking
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                               \
    if(err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while(0)

// ------------------------------
// Global configuration
// ------------------------------
// Adjust NUM_CLASSES to match your model (e.g. COCO has 80 classes).
const int NUM_CLASSES   = 7;
const int NUM_CHANNELS  = 4 + 1 + NUM_CLASSES; // [cx, cy, log_w, log_h, obj_conf, cls0,...,clsN]
const float CONF_THRESH = 0.3f;
const float NMS_THRESH  = 0.6f; 
const std::vector<int> STRIDES = {8, 16, 32};

// Structure to hold detection.
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
// Each output is expected to have shape [1, NUM_CHANNELS, grid_h, grid_w]
// and the order corresponds to the STRIDES defined above.
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

// Preprocess the image: resize, convert from BGR to RGB, scale to [0,1],
// and normalize using mean and std (ImageNet values).
// The resulting blob has shape [1,3,input_size,input_size].
cv::Mat preprocessImage(const cv::Mat &img, int input_size) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));

    // For inference, we need a normalized blob.
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float, scale to [0, 1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // Normalize: subtract mean and divide by std.
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    float std_vals[3]  = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_vals[i]) / std_vals[i];
    }
    cv::merge(channels, rgb);

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

// Simple logger for TensorRT (required by the runtime).
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log messages with severity of warning or higher
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

//
// Main function: Load the TensorRT engine file, process image, run inference,
// post-process outputs, and draw detections.
//
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path-to-tensorrt-engine> <path-to-image> [input_size]" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    std::string imagePath = argv[2];
    int input_size = 512; // default input size
    if (argc > 3)
        input_size = std::stoi(argv[3]);

    // Load image.
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }
    // For visualization, resize without normalization.
    cv::Mat visImg;
    cv::resize(image, visImg, cv::Size(input_size, input_size));

    // Preprocess image (normalization & blob creation).
    cv::Mat blob = preprocessImage(image, input_size);
    // Ensure the blob is continuous.
    assert(blob.isContinuous());
    
    // -------------------------------------------------------------------------
    // Load the serialized TensorRT engine.
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return -1;
    }
    engineFile.seekg(0, std::ifstream::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    // Create runtime and deserialize engine.
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TRT runtime." << std::endl;
        return -1;
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        runtime->destroy();
        return -1;
    }
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        engine->destroy();
        runtime->destroy();
        return -1;
    }
    // -------------------------------------------------------------------------
    // Allocate GPU buffers for all bindings.
    int nbBindings = engine->getNbBindings();
    std::vector<void*> buffers(nbBindings, nullptr);
    // To store host output buffers for outputs.
    std::vector<float*> hostOutputBuffers(nbBindings, nullptr);
    
    // Create a CUDA stream.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Set up the buffers (we assume one input and the rest outputs).
    int inputIndex = -1;
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        // Compute total number of elements.
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; j++)
            vol *= dims.d[j];
        size_t bindingSize = vol * sizeof(float);
        // Allocate device memory.
        CUDA_CHECK(cudaMalloc(&buffers[i], bindingSize));
        if (engine->bindingIsInput(i)) {
            inputIndex = i;
        } else {
            // For outputs, also allocate host memory.
            hostOutputBuffers[i] = new float[vol];
        }
    }
    if (inputIndex == -1) {
        std::cerr << "No input binding found in engine!" << std::endl;
        return -1;
    }

    // Copy input data from blob to device.
    size_t inputVolume = 1;
    nvinfer1::Dims inDims = engine->getBindingDimensions(inputIndex);
    for (int j = 0; j < inDims.nbDims; j++)
        inputVolume *= inDims.d[j];
    size_t inputSizeBytes = inputVolume * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], blob.data, inputSizeBytes, cudaMemcpyHostToDevice, stream));
    
    // Run inference.
    if (!context->enqueueV2(buffers.data(), stream, nullptr)) {
        std::cerr << "Failed to run inference!" << std::endl;
        return -1;
    }

    // Copy outputs from device to host.
    std::vector<cv::Mat> outputs; // To store output tensors as cv::Mat.
    for (int i = 0; i < nbBindings; ++i) {
        if (engine->bindingIsInput(i))
            continue;
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; j++)
            vol *= dims.d[j];
        size_t bindingSize = vol * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(hostOutputBuffers[i], buffers[i], bindingSize, cudaMemcpyDeviceToHost, stream));
        // Create a cv::Mat header for the output (using dims for shape).
        // Assume dims are in the order [1, channels, grid_h, grid_w].
        int sizes[4] = {dims.d[0], dims.d[1], dims.d[2], dims.d[3]};
        cv::Mat output(4, sizes, CV_32F, hostOutputBuffers[i]);
        outputs.push_back(output.clone()); // clone to decouple from host buffer memory.
    }
    // Synchronize the stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Process outputs to extract raw detections.
    std::vector<Detection> detections = processOutputs(outputs, input_size);
    
    // Apply Non-Maximum Suppression.
    std::vector<Detection> finalDetections = nms(detections, NMS_THRESH);

    // Print detections.
    std::cout << "Final Detections:" << std::endl;
    for (const auto &det : finalDetections) {
        float finalScore = det.obj_conf * det.cls_conf;
        std::cout << "Class: " << det.class_id << ", Score: " << finalScore
                  << ", BBox: [" << det.bbox[0] << ", " << det.bbox[1]
                  << ", " << det.bbox[2] << ", " << det.bbox[3] << "]" << std::endl;
    }

    // Draw detections on the visualization image.
    drawDetections(visImg, finalDetections, 0.5f);

    // Save result image with detections drawn.
    std::string outFile = "tensorrt_detection_result.jpg";
    if (!cv::imwrite(outFile, visImg)) {
        std::cerr << "Failed to save detection result to " << outFile << std::endl;
        return -1;
    }
    std::cout << "Detection result saved to " << outFile << std::endl;

    // Cleanup: free device and host memory, destroy stream, context, engine, runtime.
    for (int i = 0; i < nbBindings; ++i) {
        CUDA_CHECK(cudaFree(buffers[i]));
        if (!engine->bindingIsInput(i) && hostOutputBuffers[i])
            delete[] hostOutputBuffers[i];
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
} 