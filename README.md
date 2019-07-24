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

---

### 请注意：

1、本项目并不是开箱即用的，你需要一步一步的利用`/src/lane_detection`文件夹下的文件进行参数的测试，大部分文件内都是可以直接运行的，例如：

`src\lane_detection\thresholding_main.py` 颜色阈值参数的调整：

```python
if __name__ == '__main__':


	img_file = '/home/ubuntu/catkin_ws/src/lane_detection/test/test1.jpg'
	img = calibration_main.undistort_image(img_file, Visualization=False)
	all_combined, abs_bin, mag_bin, dir_bin, hls_bin = Threshold().combined_thresh(img)
	plt.subplot(3, 3, 1)
	plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 2)
	plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 3)
	plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 4)
	plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 5)
	plt.imshow(img)
	plt.subplot(3, 3, 6)
	plt.imshow(all_combined, cmap='gray', vmin=0, vmax=1)
	plt.tight_layout()
	plt.show()

```

2、如何使用:

（1）运行图像发布节点：

图像大小为 60 x 80，发布频率为30Hz。
```bash
roscore

rosrun lane_detection image_talker.py 
```
（2）运行车道线检测与跟随节点：

这个节点会进行车道线的检测，并且计算偏移误差，利用PID进行矫正控制，也就是说PID的参数也需要校准。

TODO：使用MPC进行控制。

```bash
roslaunch lane_detection lane_detection.launch
```
