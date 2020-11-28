# text_classifier_textcnn 文本分类代码
文本分类模型训练及验证代码：training.py
代码运行的环境要求：joblib 0.17.0,Keras  2.3.1,tensorflow 1.14.0
如果运行代码出错提示没有安装某个包，就在命令行输入以下命令 pip install 包名==版本号 -i https://pypi.tuna.tsinghua.edu.cn/simple
配置参数在Config类中设定。
data文件夹下为样例数据，可根据实际需要更改数据路径
如果输入文本有标题和正文，则填写title和content相关信息，如果只有正文，则title_name为None
如果有两个输出，填写label1和label2相关信息，如果只有一个输出，label2_name为None
do_train,do_eval,do_test至少有一个为True
如果需要在之前训练的模型上微调，则指定模型的路径trained_model_dir
