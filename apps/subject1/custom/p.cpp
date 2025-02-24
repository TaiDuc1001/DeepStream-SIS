#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <numeric>
#include "nvdsinfer_custom_impl.h"

// Enable or disable debug prints:
// Set ENABLE_DEBUG to 0 to turn off debug; to 1 to enable them.
#ifndef ENABLE_DEBUG
#define ENABLE_DEBUG 1
#endif

#if ENABLE_DEBUG
    #define DEBUG_MSG(...) do { std::cout << __VA_ARGS__; } while(0)
#else
    #define DEBUG_MSG(...) ((void)0)
#endif

// Constants and configuration (using 12 channels per grid cell)
const int NUM_CLASSES     = 7;
const int NUM_CHANNELS    = 12;   // [center_x, center_y, log_w, log_h, obj_conf, class0, class1, class2, class3, class4, class5, class6]
const float CONF_THRESH   = 0.3f;  // threshold for filtering detections (product of confidences)
const float NMS_THRESH    = 0.6f; // NMS IoU threshold
const std::vector<int> STRIDES = {8, 16, 32};

// Updated Detection structure: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
struct Detection {
    float bbox[4];
    float obj_conf;
    float class_conf;
    int class_id;
};

// Minimal tensor implementation for basic operations.
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape; // stored as [batch, dim1, dim2]
};

// Stage holds one prediction tensor (assumed in CHW order)
struct Stage {
    int H;
    int W;
    int stride;
    std::vector<float> data;
};

//////////////////////  HELPER FUNCTIONS  //////////////////////

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void apply_sigmoid(Tensor &t, int start_channel) {
    int N = t.shape[1];
    int channels = t.shape[2];
    DEBUG_MSG("Debug: apply_sigmoid() applying sigmoid to channels " 
              << start_channel << " to " << channels - 1 
              << " for " << N << " predictions." << std::endl);
    for (int i = 0; i < N; i++) {
        int base = i * channels;
        for (int j = start_channel; j < channels; j++) {
            t.data[base + j] = sigmoid(t.data[base + j]);
        }
    }
}

float compute_iou(const Detection &a, const Detection &b) {
    float x1 = std::max(a.bbox[0], b.bbox[0]);
    float y1 = std::max(a.bbox[1], b.bbox[1]);
    float x2 = std::min(a.bbox[2], b.bbox[2]);
    float y2 = std::min(a.bbox[3], b.bbox[3]);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    float area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    return inter / (area_a + area_b - inter);
}

Tensor flatten_stage(const Stage &stage) {
    Tensor t;
    t.shape = {1, NUM_CHANNELS, stage.H * stage.W};
    t.data  = stage.data;
    DEBUG_MSG("Debug: flatten_stage() called for Stage with H = " << stage.H 
              << ", W = " << stage.W 
              << ", stride = " << stage.stride 
              << ", shape = " << t.shape[0] << ", " << t.shape[1] << ", " << t.shape[2]
              << ", total elements = " << stage.data.size() << std::endl);
    DEBUG_MSG("Debug: First 5 elements of flattened stage: ");
    for (int i = 0; i < std::min((int)t.data.size(), 5); i++)
        DEBUG_MSG(t.data[i] << " ");
    DEBUG_MSG(std::endl);
    return t;
}

// Concatenates a vector of tensors along axis 2.
Tensor cat_axis2(const std::vector<Tensor> &tensors) {
    DEBUG_MSG("Debug: cat_axis2() called with " << tensors.size() << " tensors." << std::endl);
    if (tensors.empty()) {
        DEBUG_MSG("Debug: cat_axis2() received empty tensor list." << std::endl);
        return Tensor();
    }

    int batch = tensors[0].shape[0];
    int channels = tensors[0].shape[1];
    // Ensure all tensors have matching batch and channels
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape[0] != batch || tensors[i].shape[1] != channels) {
            DEBUG_MSG("Debug: cat_axis2() tensors have different batch or channel dimensions." << std::endl);
            return Tensor();
        }
    }

    int total = 0;
    for (const auto &t : tensors)
        total += t.shape[2];
    Tensor out;
    out.shape = {batch, channels, total};
    out.data.reserve(batch * channels * total);
    for (int c = 0; c < channels; c++) {
        for (const auto &t : tensors) {
            for (int n = 0; n < t.shape[2]; n++) {
                out.data.push_back(t.data[c * t.shape[2] + n]);
            }
        }
    }
    DEBUG_MSG("Debug: cat_axis2() concatenated tensor shape: ["
              << out.shape[0] << ", " << out.shape[1] << ", " << out.shape[2] << "]" << std::endl);
    DEBUG_MSG("Debug: First 5 values of concatenated tensor (channel 0): ");
    for (int i = 0; i < std::min(out.shape[2], 5); i++)
        DEBUG_MSG(out.data[i] << " ");
    DEBUG_MSG(std::endl);
    return out;
}

// Permutes a tensor from shape [1, channels, N] to [1, N, channels].
Tensor permute_0_2_1(const Tensor &in) {
    Tensor out;
    int b = in.shape[0];
    int channels = in.shape[1];
    int N = in.shape[2];
    out.shape = {b, N, channels};
    out.data.resize(b * N * channels);
    for (int ch = 0; ch < channels; ch++)
        for (int n = 0; n < N; n++)
            out.data[n * channels + ch] = in.data[ch * N + n];
    
    DEBUG_MSG("Debug: permute_0_2_1() input shape: ["
              << in.shape[0] << ", " << in.shape[1] << ", " << in.shape[2] << "]"
              << " -> output shape: ["
              << out.shape[0] << ", " << out.shape[1] << ", " << out.shape[2] << "]" << std::endl);
    DEBUG_MSG("Debug: First 5 values after permutation (first prediction): ");
    for (int j = 0; j < std::min(channels, 5); j++)
        DEBUG_MSG(out.data[j] << " ");
    DEBUG_MSG(std::endl);
    return out;
}

// Applies softmax activation to channels from start_channel to the end for each prediction.
void apply_softmax(Tensor &t, int start_channel) {
    int N = t.shape[1];         // Number of predictions
    int channels = t.shape[2];    // Total number of channels per prediction
    int num_class_channels = channels - start_channel; // Number of class channels

    DEBUG_MSG("Debug: apply_softmax() applying softmax to channels " 
              << start_channel << " to " << channels - 1 
              << " for " << N << " predictions." << std::endl);
    for (int i = 0; i < N; i++) {
        int base = i * channels;
        // Find the maximum value for the class channels (for numerical stability)
        float max_val = t.data[base + start_channel];
        for (int j = 1; j < num_class_channels; j++) {
            int idx = base + start_channel + j;
            if (t.data[idx] > max_val) {
                max_val = t.data[idx];
            }
        }
        // Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int j = 0; j < num_class_channels; j++) {
            int idx = base + start_channel + j;
            t.data[idx] = std::exp(t.data[idx] - max_val);
            sum_exp += t.data[idx];
        }
        // Normalize to obtain probabilities
        for (int j = 0; j < num_class_channels; j++) {
            int idx = base + start_channel + j;
            t.data[idx] /= sum_exp;
        }
        // Debug output for the first few predictions
        if (i < 3) {
            DEBUG_MSG("Debug: softmax on prediction " << i << ": ");
            for (int j = 0; j < num_class_channels; j++) {
                DEBUG_MSG(t.data[base + start_channel + j] << " ");
            }
            DEBUG_MSG(std::endl);
        }
    }
}

// Concatenates tensors along axis 1.
Tensor cat_axis1(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        DEBUG_MSG("Debug: cat_axis1() received empty tensor list." << std::endl);
        return Tensor();
    }

    int batch = tensors[0].shape[0];
    int D = tensors[0].shape[2];
    // Ensure all tensors have the same batch and depth
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape[0] != batch || tensors[i].shape[2] != D) {
            DEBUG_MSG("Debug: cat_axis1() tensors have different batch or depth dimensions." << std::endl);
            return Tensor();
        }
    }

    int total = 0;
    for (const auto &t : tensors)
        total += t.shape[1];
    Tensor out;
    out.shape = {batch, total, D};
    out.data.reserve(batch * total * D);
    for (const auto &t : tensors) {
        out.data.insert(out.data.end(), t.data.begin(), t.data.end());
    }
    DEBUG_MSG("Debug: cat_axis1() concatenated tensor shape: ["
              << out.shape[0] << ", " << out.shape[1] << ", " << out.shape[2] << "]" << std::endl);
    DEBUG_MSG("Debug: First 5 values of cat_axis1 tensor: ");
    for (int i = 0; i < std::min(total, 5); i++) {
        DEBUG_MSG(out.data[i] << " ");
    }
    DEBUG_MSG(std::endl);
    return out;
}

// Generates a mesh grid tensor with x and y coordinates
Tensor meshgrid_tensor(int H, int W) {
    Tensor grid;
    grid.shape = {1, H * W, 2};
    grid.data.reserve(H * W * 2);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            grid.data.push_back(static_cast<float>(j));
            grid.data.push_back(static_cast<float>(i));
        }
    }
    DEBUG_MSG("Debug: meshgrid_tensor() grid tensor shape: ["
              << grid.shape[0] << ", " << grid.shape[1] << ", " << grid.shape[2] << "]" << std::endl);
    return grid;
}

// Generates a stride tensor with stride value repeated for each grid cell
Tensor create_stride_tensor(int H, int W, int stride) {
    Tensor stride_tensor;
    stride_tensor.shape = {1, H * W, 1};
    stride_tensor.data.reserve(H * W * 1);
    for (int i = 0; i < H * W; ++i) {
        stride_tensor.data.push_back(static_cast<float>(stride));
    }
    DEBUG_MSG("Debug: create_stride_tensor() stride tensor shape: ["
              << stride_tensor.shape[0] << ", " << stride_tensor.shape[1] << ", " << stride_tensor.shape[2] << "]" << std::endl);
    return stride_tensor;
}

// Adjusts predictions using the grid and stride tensors.
void adjust_predictions(Tensor &pred, const Tensor &grid, const Tensor &strides) {
    int total = pred.shape[1];
    int channels = pred.shape[2];  // expected layout: [cx, cy, log_w, log_h, obj_conf, class0...class6]
    DEBUG_MSG("Debug: Adjusting predictions for total predictions: " << total << std::endl);
    for (int i = 0; i < total; i++) {
        int base = i * channels;
        int grid_base = i * 2;
        float s = strides.data[i];
        float orig_cx = pred.data[base + 0];
        float orig_cy = pred.data[base + 1];
        float orig_log_w = pred.data[base + 2];
        float orig_log_h = pred.data[base + 3];

        pred.data[base + 0] = (pred.data[base + 0] + grid.data[grid_base + 0]) * s;
        pred.data[base + 1] = (pred.data[base + 1] + grid.data[grid_base + 1]) * s;
        pred.data[base + 2] = std::exp(pred.data[base + 2]) * s;
        pred.data[base + 3] = std::exp(pred.data[base + 3]) * s;
        
        if (i < 3) {
            DEBUG_MSG("Debug: Prediction " << i << " adjustment:" << std::endl);
            DEBUG_MSG("\tOriginal: cx = " << orig_cx << ", cy = " << orig_cy 
                      << ", log_w = " << orig_log_w << ", log_h = " << orig_log_h << std::endl);
            DEBUG_MSG("\tGrid: (" << grid.data[grid_base] << ", " << grid.data[grid_base + 1] 
                      << "), stride: " << s << std::endl);
            DEBUG_MSG("\tAdjusted: cx = " << pred.data[base + 0] << ", cy = " << pred.data[base + 1]
                      << ", w = " << pred.data[base + 2] << ", h = " << pred.data[base + 3] << std::endl);
        }
    }
}

// Converts predictions from center coordinates and size to corner coordinates.
void convert_to_box_corners(Tensor &pred) {
    int total = pred.shape[1];
    int channels = pred.shape[2];
    DEBUG_MSG("Debug: Converting predictions to box corners for " << total << " predictions." << std::endl);
    for (int i = 0; i < total; i++) {
        int base = i * channels;
        float cx = pred.data[base + 0];
        float cy = pred.data[base + 1];
        float w  = pred.data[base + 2];
        float h  = pred.data[base + 3];
        float orig_cx = cx, orig_cy = cy, orig_w = w, orig_h = h;
        pred.data[base + 0] = cx - w / 2.0f;
        pred.data[base + 1] = cy - h / 2.0f;
        pred.data[base + 2] = cx + w / 2.0f;
        pred.data[base + 3] = cy + h / 2.0f;
        if (i < 3) {
            DEBUG_MSG("Debug: Box conversion for prediction " << i << ":" << std::endl);
            DEBUG_MSG("\tCenter: (" << orig_cx << ", " << orig_cy 
                      << "), Size: (" << orig_w << ", " << orig_h << ")" << std::endl);
            DEBUG_MSG("\tBox: (" << pred.data[base + 0] << ", " << pred.data[base + 1] 
                      << ", " << pred.data[base + 2] << ", " << pred.data[base + 3] << ")" << std::endl);
        }
    }
}

// Updated filter_detections() to correctly compute the maximum class confidence and predicted class.
std::vector<std::vector<float>> filter_detections(const Tensor &pred, float score_thresh) {
    std::vector<std::vector<float>> detections;
    int total = pred.shape[1];
    int channels = pred.shape[2]; // expected: [x1, y1, x2, y2, obj_conf, class0, class1, class2, class3, class4, class5, class6]
    DEBUG_MSG("Debug: Filtering detections with score threshold: " << score_thresh << std::endl);
    
    for (int i = 0; i < total; i++) {
        int base = i * channels;
        float x1 = pred.data[base + 0];
        float y1 = pred.data[base + 1];
        float x2 = pred.data[base + 2];
        float y2 = pred.data[base + 3];
        float obj_conf = pred.data[base + 4];
        
        // Determine the maximum class score and its corresponding class index.
        float cls_conf = -1.0f;
        int cls_id = -1;
        for (int ch = 5; ch < 12; ch++) {
            float class_score = pred.data[base + ch];
            if (class_score > cls_conf) {
                cls_conf = class_score;
                cls_id = ch - 5; // Adjust to have class IDs 0-6.
            }
        }
        
        float score = obj_conf * cls_conf;
        if (score >= score_thresh) {
            detections.push_back({x1, y1, x2, y2, obj_conf, cls_conf, static_cast<float>(cls_id)});
            DEBUG_MSG("Debug: Detection accepted (index " << i << "): "
                      << "x1 = " << x1 << ", y1 = " << y1 
                      << ", x2 = " << x2 << ", y2 = " << y2 
                      << ", obj_conf = " << obj_conf 
                      << ", cls_conf = " << cls_conf 
                      << ", score = " << score 
                      << ", class_id = " << cls_id << std::endl);
        } else {
            if (i < 3) {
                DEBUG_MSG("Debug: Detection rejected (index " << i << "): "
                          << "score = " << score << std::endl);
            }
        }
    }
    
    DEBUG_MSG("Debug: Total detections after filtering: " << detections.size() << std::endl);
    return detections;
}

// Main decode function which simulates the anchor-free decoupled head pipeline.
std::vector<std::vector<float>> anchor_free_decoupled_head_decode(
    const std::vector<Stage> &stages,
    std::pair<int, int> original_shape,
    float score_thresh)
{
    DEBUG_MSG("Debug: Starting anchor_free_decoupled_head_decode()" << std::endl);
    std::vector<Tensor> flat_tensors;
    for (size_t i = 0; i < stages.size(); i++) {
        DEBUG_MSG("Debug: Processing stage " << i << std::endl);
        flat_tensors.push_back(flatten_stage(stages[i]));
    }
    Tensor pred_cat = cat_axis2(flat_tensors);
    Tensor pred_tensor = permute_0_2_1(pred_cat);
        
    // Debug: Print a few raw values from the output tensor before applying any activation.
    std::cout << "First prediction raw values:" << std::endl;
    for (int i = 0; i < 12; i++) {
        std::cout << pred_tensor.data[i] << " ";
    }
    std::cout << std::endl;

    // Apply sigmoid only on channels 4 and onwards (if that's what Python does)
    apply_sigmoid(pred_tensor, 4);

    // Debug: Check the values after activation.
    std::cout << "First prediction after sigmoid:" << std::endl;
    for (int i = 0; i < 12; i++) {
        std::cout << pred_tensor.data[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Tensor> grid_tensors, stride_tensors;
    for (size_t i = 0; i < stages.size(); i++) {
        DEBUG_MSG("Debug: Creating grid and stride for stage " << i << std::endl);
        grid_tensors.push_back(meshgrid_tensor(stages[i].H, stages[i].W));
        stride_tensors.push_back(create_stride_tensor(stages[i].H, stages[i].W, stages[i].stride));
    }
    Tensor grids = cat_axis1(grid_tensors);
    Tensor strides = cat_axis1(stride_tensors);
    
    adjust_predictions(pred_tensor, grids, strides);
    convert_to_box_corners(pred_tensor);
    
    {
        int channels = pred_tensor.shape[2];
        int total = pred_tensor.shape[1];
        DEBUG_MSG("Debug: Sum of class values (channels 5 to 11) for first few predictions:" << std::endl);
        int num_to_sum = std::min(total, 5);
        for (int i = 0; i < num_to_sum; i++) {
            int base = i * channels;
            float sigmoid_class_sum = 0.0f;
            for (int ch = 5; ch < 12; ch++) {
                sigmoid_class_sum += pred_tensor.data[base + ch];
            }
            DEBUG_MSG("Prediction " << i << ": sum(class values) = " << sigmoid_class_sum << std::endl);
        }
    }
    
    DEBUG_MSG("Debug: anchor_free_decoupled_head_decode() final prediction sample:" << std::endl);
    {
        int channels = pred_tensor.shape[2];
        int total = pred_tensor.shape[1];
        int num_to_print = std::min(total, 5);
        for (int i = 0; i < num_to_print; i++) {
            int base = i * channels;
            DEBUG_MSG("Prediction " << i << ": ");
            for (int j = 0; j < channels; j++) {
                DEBUG_MSG(pred_tensor.data[base + j] << " ");
            }
            DEBUG_MSG(std::endl);
        }
    }
    
    auto detections = filter_detections(pred_tensor, score_thresh);
    DEBUG_MSG("Debug: anchor_free_decoupled_head_decode() produced " << detections.size() << " detections." << std::endl);
    return detections;
}

// -------------------------------------------------------------------------
// Helper structure and function for coordinate trick based NMS

struct Box {
    float x1, y1, x2, y2;
};

float compute_iou_boxes(const Box &a, const Box &b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

// -------------------------------------------------------------------------
// Batched NMS using the coordinate trick.
// Mimics the Python _batched_nms_coordinate_trick():
std::vector<Detection> batched_nms_coordinate_trick(const std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) {
        return std::vector<Detection>();
    }
    
    // Find the maximum coordinate value among all detection boxes.
    float max_coordinate = 0.0f;
    for (const auto &det : detections) {
        for (int i = 0; i < 4; i++) {
            max_coordinate = std::max(max_coordinate, det.bbox[i]);
        }
    }
    
    // For each detection, add a class-dependent offset to each coordinate.
    // This ensures boxes from different classes do not overlap.
    std::vector<Box> mod_boxes;
    mod_boxes.reserve(detections.size());
    for (const auto &det : detections) {
        float offset = det.class_id * (max_coordinate + 1.0f);
        Box b;
        b.x1 = det.bbox[0] + offset;
        b.y1 = det.bbox[1] + offset;
        b.x2 = det.bbox[2] + offset;
        b.y2 = det.bbox[3] + offset;
        mod_boxes.push_back(b);
    }
    
    // Compute detection scores as product of objectness and class confidence.
    std::vector<float> scores;
    scores.reserve(detections.size());
    for (const auto &det : detections) {
        scores.push_back(det.obj_conf * det.class_conf);
    }
    
    // Build a list of indices sorted by detection score in descending order.
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,...
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    
    // Perform NMS on the modified boxes.
    std::vector<int> keep;
    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);
        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); i++) {
            int idx = indices[i];
            float iou = compute_iou_boxes(mod_boxes[current], mod_boxes[idx]);
            if (iou <= iou_threshold) {
                remaining.push_back(idx);
            }
        }
        indices = remaining;
    }
    
    // Return the original detections corresponding to kept indices.
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) {
        out.push_back(detections[idx]);
    }
    return out;
}

// -------------------------------------------------------------------------
// Batched NMS using a per-class (vanilla) approach.
// Mimics the Python _batched_nms_vanilla():
std::vector<Detection> batched_nms_vanilla(const std::vector<Detection>& detections, float iou_threshold) {
    std::unordered_map<int, std::vector<Detection>> class_map;
    for (const auto &det : detections) {
        class_map[det.class_id].push_back(det);
    }
    
    std::vector<Detection> keep;
    // Process each class independently.
    for (auto &pair : class_map) {
        int cls_id = pair.first;
        auto &dets = pair.second;
        // Sort detections for the class by score in descending order.
        std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
            return (a.obj_conf * a.class_conf) > (b.obj_conf * b.class_conf);
        });
        while (!dets.empty()) {
            Detection current = dets.front();
            keep.push_back(current);
            dets.erase(dets.begin());
            // Remove all detections with IoU > iou_threshold with the current box.
            dets.erase(std::remove_if(dets.begin(), dets.end(),
                [&](const Detection &d) {
                    float iou = compute_iou(current, d);
                    return iou > iou_threshold;
                }),
                dets.end());
        }
    }
    // Sort the final detections by score in descending order.
    std::sort(keep.begin(), keep.end(), [](const Detection &a, const Detection &b) {
        return (a.obj_conf * a.class_conf) > (b.obj_conf * b.class_conf);
    });
    return keep;
}

// -------------------------------------------------------------------------
// Batched NMS wrapper that mimics the Python batched_nms() behavior.
// Here we choose the coordinate trick if the number of detections is relatively small,
// or the per-class (vanilla) approach if there are many detections.
std::vector<Detection> batched_nms(const std::vector<Detection>& detections, float iou_threshold) {
    if (detections.size() > 4000) {
        return batched_nms_vanilla(detections, iou_threshold);
    } else {
        return batched_nms_coordinate_trick(detections, iou_threshold);
    }
}

//////////////////////  DEEPSTREAM PARSE FUNCTION  //////////////////////

extern "C"
bool CustomParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    DEBUG_MSG("Debug: Starting NvDsInferParseCustomYolox()" << std::endl);
    std::vector<Stage> stages;
    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        float* output = static_cast<float*>(outputLayersInfo[i].buffer);
        DEBUG_MSG("Debug: Processing output layer " << i << std::endl);
        int stride = STRIDES[i];
        int grid_size_w = networkInfo.width / stride;
        int grid_size_h = networkInfo.height / stride;
        if (grid_size_w != grid_size_h) {
            DEBUG_MSG("Debug: Grid size mismatch: " << grid_size_w << " != " << grid_size_h << std::endl);
            return false;
        }
        int grid_size = grid_size_w;
        DEBUG_MSG("Debug: Layer " << i << " grid size: " << grid_size 
                  << ", stride: " << stride << std::endl);
        Stage stage;
        stage.H = grid_size;
        stage.W = grid_size;
        stage.stride = stride;
        int num_elems = NUM_CHANNELS * grid_size * grid_size;
        stage.data.resize(num_elems);
        std::memcpy(stage.data.data(), output, num_elems * sizeof(float));
        DEBUG_MSG("Debug: Stage " << i << " data copied, total elements: " << num_elems << std::endl);
        if (num_elems >= 5) {
            DEBUG_MSG("Debug: First 5 elements of stage " << i << ": "
                      << stage.data[0] << ", " << stage.data[1] << ", " << stage.data[2]
                      << ", " << stage.data[3] << ", " << stage.data[4] << std::endl);
        } else {
            DEBUG_MSG("Debug: Stage " << i << " has less than 5 elements." << std::endl);
        }
        stages.push_back(stage);
    }
    
    // Decode predictions using the anchor-free decoupled head method.
    auto decoded = anchor_free_decoupled_head_decode(stages, {networkInfo.width, networkInfo.height}, CONF_THRESH);
    
    // Convert decoded detections into Detection objects.
    std::vector<Detection> detections;
    DEBUG_MSG("Debug: Converting decoded detections to Detection objects." << std::endl);
    for (const auto &d : decoded) {
        if (d.size() < 7)
            continue;
        Detection det;
        for (int j = 0; j < 4; j++)
            det.bbox[j] = d[j];
        det.obj_conf   = d[4];
        det.class_conf = d[5];
        det.class_id   = static_cast<int>(d[6]);
        detections.push_back(det);
        DEBUG_MSG("Debug: Detection object: x1 = " << det.bbox[0]
                  << ", y1 = " << det.bbox[1]
                  << ", x2 = " << det.bbox[2]
                  << ", y2 = " << det.bbox[3]
                  << ", obj_conf = " << det.obj_conf
                  << ", cls_conf = " << det.class_conf
                  << ", class_id = " << det.class_id << std::endl);
    }
    
    // Apply class-specific NMS.
    std::vector<Detection> final_dets = batched_nms(detections, NMS_THRESH);
    
    // Convert to DeepStream object detection format and add debug printouts.
    DEBUG_MSG("Debug: Converting final detections to DeepStream object format." << std::endl);
    for (const auto &det : final_dets) {
        NvDsInferObjectDetectionInfo obj;
        obj.left = det.bbox[0];
        obj.top = det.bbox[1];
        obj.width = det.bbox[2] - det.bbox[0];
        obj.height = det.bbox[3] - det.bbox[1];
        obj.detectionConfidence = det.class_conf * det.obj_conf;
        obj.classId = det.class_id;
        objectList.push_back(obj);
        std::cout << "Debug: Added detection - left: " << obj.left
                  << ", top: " << obj.top
                  << ", width: " << obj.width
                  << ", height: " << obj.height
                  << ", detectionConfidence: " << obj.detectionConfidence
                  << ", classId: " << obj.classId << std::endl;
    }
    
    std::cout << "Debug: Total number of final detections: " << objectList.size() << std::endl;
    return true;
}