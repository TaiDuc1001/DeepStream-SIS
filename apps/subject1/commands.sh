GST_DEBUG=3 \
# gst-launch-1.0 filesrc location=videos/DongKhoi_MacThiBuoi.MOV ! qtdemux ! h265parse ! video/x-h265,stream-format=byte-stream ! filesink location=videos/DongKhoi_MacThiBuoi.h265
# gst-launch-1.0 filesrc location=videos/DongKhoi_MacThiBuoi.MOV ! decodebin ! videoconvert ! x264enc ! h264parse ! video/x-h264,stream-format=byte-stream ! filesink location=videos/DongKhoi_MacThiBuoi.h264
GST_DEBUG=3 python3 deepstream_custom_binding_test.py /home/nvidia/deepstream_python_apps/apps/subject1/videos/DongKhoi_MacThiBuoi.h265
GST_DEBUG=3 python3 deepstream_custom_binding_test.py /opt/nvidia/deepstream/deepstream-6.2/samples/streams/sample_1080p_h265.mp4
# gst-launch-1.0 \
#   filesrc location=/home/nvidia/deepstream_python_apps/apps/subject1/videos/DongKhoi_MacThiBuoi.MOV ! \
#   qtdemux name=demux \
#   demux.video_0 ! queue ! h265parse ! video/x-h265,stream-format=byte-stream ! \
#   filesink location=/home/nvidia/deepstream_python_apps/apps/subject1/videos/DongKhoi_MacThiBuoi.h265

python3 deepstream_test_3.py -i file:///home/nvidia/deepstream_python_apps/apps/subject1/videos/DongKhoi_MacThiBuoi.MOV
ffmpeg -i DongKhoi_MacThiBuoi.MOV -c:v libx265 -profile:v main -pix_fmt yuv420p -c:a copy DongKhoi_MacThiBuoi.mp4

g++ -std=c++11 onnx_detection.cpp -o onnx_detection `pkg-config --cflags --libs opencv4`
./onnx_detection ../models/detection_yolox_tiny_best.onnx /home/nvidia/deepstream_python_apps/apps/subject1/norm_frame_0.png 512

g++ -std=c++11 -O2 tensorrt_detection.cpp -o tensorrt_detection \
  `pkg-config --cflags --libs opencv4` \
  -I/usr/local/cuda/include \
  -I/usr/include/aarch64-linux-gnu \
  -L/lib/aarch64-linux-gnu -lnvinfer -lnvinfer_plugin -lcudart

./tensorrt_detection ../models/detection_yolox_tiny_best.onnx_b1_gpu0_fp16.engine /home/nvidia/deepstream_python_apps/apps/subject1/norm_frame_0.png 512
