#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

// Configuration
const int NUM_CLASSES   = 7;
const int NUM_CHANNELS  = 12;  // [cx, cy, log_w, log_h, obj_conf, cls0,...,cls6]
const float CONF_THRESH = 0.3f;
const float NMS_THRESH  = 0.6f;
const std::vector<int> STRIDES = {8, 16, 32};

// Structure to hold a detection
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

// Simple per-class NMS
std::vector<Detection> nms(const std::vector<Detection>& dets, float nms_thresh) {
    std::vector<Detection> result;
    std::unordered_map<int, std::vector<Detection>> classMap;
    for (const auto &d : dets)
        classMap[d.class_id].push_back(d);
    for (auto &pair : classMap) {
        auto &dlist = pair.second;
        // Sort detections by score (objectness * class score) in descending order
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
    // Optionally sort final detections by score
    std::sort(result.begin(), result.end(), [](const Detection &a, const Detection &b) {
        return (a.obj_conf * a.cls_conf) > (b.obj_conf * b.cls_conf);
    });
    return result;
}

// Main DeepStream parser function
extern "C"
bool CustomParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    std::vector<Detection> detections;

    // Loop over each output layer (stage)
    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        float* output = static_cast<float*>(outputLayersInfo[i].buffer);
        int stride = STRIDES[i];
        int grid_w = networkInfo.width / stride;
        int grid_h = networkInfo.height / stride;
        int num_cells = grid_w * grid_h;
        // Data is in CHW order: channels first then spatial.
        // For each grid cell, reconstruct the prediction.
        for (int idx = 0; idx < num_cells; idx++) {
            // Get raw values (note: for channel j, value = output[j * num_cells + idx])
            float cx     = output[0 * num_cells + idx];
            float cy     = output[1 * num_cells + idx];
            float log_w  = output[2 * num_cells + idx];
            float log_h  = output[3 * num_cells + idx];
            // Apply sigmoid to objectness (channel 4) and class scores (channels 5 to 11)
            float obj_conf = sigmoid(output[4 * num_cells + idx]);
            float max_cls  = -1.0f;
            int   cls_id   = -1;
            for (int c = 5; c < 5 + NUM_CLASSES; c++) {
                float cls_score = sigmoid(output[c * num_cells + idx]);
                if (cls_score > max_cls) {
                    max_cls = cls_score;
                    cls_id  = c - 5;
                }
            }
            float score = obj_conf * max_cls;
            if (score < CONF_THRESH)
                continue;

            // Compute grid cell coordinates
            int x_idx = idx % grid_w;
            int y_idx = idx / grid_w;
            // Adjust predictions: add grid offsets and multiply by stride
            float adjusted_cx = (cx + x_idx) * stride;
            float adjusted_cy = (cy + y_idx) * stride;
            float w = std::exp(log_w) * stride;
            float h = std::exp(log_h) * stride;
            // Convert center coordinates to box corners
            float x1 = adjusted_cx - w / 2.0f;
            float y1 = adjusted_cy - h / 2.0f;
            float x2 = adjusted_cx + w / 2.0f;
            float y2 = adjusted_cy + h / 2.0f;

            // Save detection
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

    // Apply Non-Maximum Suppression (NMS)
    std::vector<Detection> final_dets = nms(detections, NMS_THRESH);

    // Convert final detections to DeepStream format
    for (const auto &det : final_dets) {
        NvDsInferObjectDetectionInfo obj;
        obj.left  = det.bbox[0];
        obj.top   = det.bbox[1];
        obj.width = det.bbox[2] - det.bbox[0];
        obj.height= det.bbox[3] - det.bbox[1];
        obj.detectionConfidence = det.obj_conf * det.cls_conf;
        obj.classId = det.class_id;
        objectList.push_back(obj);
    }

    return true;
}
