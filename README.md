## 车道线检测与跟随


####  / bin 可执行文件目录

* "image_talker.py":

读取摄像头图像。并且发布到 '/Image' topic, 大小为 （60 80 ），发布频率为30Hz。

* "main.py":

车道线的检测算法调用，以及PID的运算。

* "pid_node.py":

PID的实现，在"main.py"中被调用。

#### /src/lane_detection 车道线检测图像处理过程库。


处理过程可以参考 [github.com/georgesung/advanced_lane_detection](https://github.com/georgesung/advanced_lane_detection)

