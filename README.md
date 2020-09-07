# biggan-paddle
论文复现：biggan的paddle实现

直接运行python train.py就可以开始训练

主训练函数如下：
train(cifar10_generator, epoch_num=args.epoch_num, batch_size=args.batch_size, use_gpu=True, load_model=args.load_model, model_path = args.model_path, n_class=10, draw=args.draw, saveimg=args.saveimg) 

运行参数设置如下：
    parser.add_argument('--load_model', type=bool, default=False,
    help='if load_model or not')
parser.add_argument('--draw', type=bool, default=False,
    help='if show imgs')
parser.add_argument('--saveimg', type=bool, default=False,
    help='if save imgs to file')
parser.add_argument('--model_path', type=str, default="./output",
    help='where to load or save model')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_num', type=int, default=1,

    

建议在有图形显示的系统用如下命令训练：
python train.py  --draw True --saveimg True 


因为水平有限，复现的还很不到位。
另外还可以通过百度aistudio的项目直接运行复现，aistudio提供特斯拉V100的gpu卡，用起来贼爽！
项目复现地址：https://aistudio.baidu.com/aistudio/projectdetail/861109


