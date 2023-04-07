1.在当前目录执行python ./main.py.
2.参数设置：
test_dir：测试集图像文件夹，在main.py中设置该参数为测试集的路径.
segments_dir：分割结果图像文件夹，在main.py中设置该参数为要保存图像的路径.
model_trained：设置为True，即使用已经训练好的模型
3.prediction.txt会生成在test_dir对应的文件夹中.
4.test.ipynb与test.png为测试一张网络图片的识别效果

使用库：
cudatoolkit               10.2.89
matplotlib                3.5.0
numpy                   1.21.2
opencv-python           4.5.5.62
python                   3.9.7
pytorch                  1.10.1
