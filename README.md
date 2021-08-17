# karman-2d

该仓库复现了论文["Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers"](http://arxiv.org/abs/2007.00016) by Kiwon Um, Robert Brand, Yun (Raymond) Fei, Philipp Holl, Nils Thuerey中的一个
不稳定流的例子。

## 使用说明
**需要的依赖包**
- [Pytorch](https://pytorch.org/);1.9.0
- [PhiFlow](https://github.com/tum-pbs/PhiFlow);

**数据的生成**
在当前目录下执行下面语句可以分别生成训练数据和测试数据:
```
make Ref_Data     #生成NON,SOL所需的训练数据,存储到Reference-Data文件夹下
make Pre_Data     #生成Pre(beta=0)所需的训练数据,存储到Pre-Data文件夹下
make Pre_Data_Sr  #生成Pre(beta=1)所需的训练数据,存储到Pre-Data-Sr文件夹下
make Test_Data    #生成测试数据,存储到Test-Data文件夹下
```

**训练模型**
```
make NON          #训练NON模型
make PRO_NON      #训练ProNON模型
make PRE          #训练PRE模型(通过修改Makefile文件中的input可以指定测试数据为Pre_Data_Sr)
make SOL          #训练SOL模型
```
可以通过修改Makefile中的相关参数来设置训练的超参数。

**测试模型**
```
make Source_Test #计算无矫正迭代的误差
make Apply       #计算矫正迭代的误差(通过设定model_path可以指定希望使用的矫正模型的地址，从而使用该模型)
```

**生成图像结果**
```
make Generate_Imag  #生成密度速度的热力图
```
设置type可以指定使用的模型类型，可以选择的有Ref(生成参考结果)、Source(生成无矫正迭代结果)和Model(生成矫正迭代结果)。其中若type为Model则还需要设置model_path来指定希望使用模型的地址。
