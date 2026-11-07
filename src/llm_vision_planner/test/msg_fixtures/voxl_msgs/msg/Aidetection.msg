# Test fixture shaped like ModalAI's voxl_msgs/Aidetection. The bag tool reads
# the real definition from src/voxl_msgs/msg when it runs on recorded bags.
int64 timestamp_ns
uint32 class_id
int32 frame_id
string class_name
string cam
float32 class_confidence
float32 detection_confidence
float32 x_min
float32 y_min
float32 x_max
float32 y_max
