cd settings/parser
make
cd -
GST_DEBUG=3 python3 main.py \
-i \
file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/DongKhoi_MacThiBuoi.mp4 \
file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/RachBungBinh_NguyenThong_2.mp4 \
file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/TranHungDao_NguyenVanCu.mp4 \
file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/TranKhacChan_TranQuangKhai.mp4
# file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/tracking/DongKhoi_MacThiBuoi.mp4 \
# file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/tracking/RachBungBinh_NguyenThong_2.mp4 \
# file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/tracking/TranHungDao_NguyenVanCu.mp4 \
# file:///home/nvidia/deepstream_python_apps_subject2/apps/subject1/input/tracking/TranKhacChan_TranQuangKhai.mp4 \