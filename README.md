# biggan-paddle
biggan的paddle实现

直接运行python train.py就可以开始训练

主训练函数如下：
train(cifar10_generator, epoch_num=args.epoch_num, batch_size=BATCH_SIZE, use_gpu=True, load_model=args.load_model, model_path = args.model_path, n_class=10, draw=args.draw, saveimg=args.saveimg) 


建议在有图形显示的系统用如下命令训练：
python train.py  --draw True --saveimg True 


因为水平有限，复现的还很不到位。
另外还可以通过百度aistudio的项目直接复现，aistudio提供特斯拉V100的gpu卡，用起来贼爽！
https://aistudio.baidu.com/aistudio/projectdetail/861109


