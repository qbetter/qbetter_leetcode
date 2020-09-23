一，手推Logistic回归
logistic回归是概率型非线性回归，研究二分类结果y与一些影响因素(x1,x2...,xd)之间关系的一种多变量分析方法。
logistic回归基于线性分类$θ^TX+b$,并使用sigmoid函数将线性函数做非线性映射到（0，1）空间中去。于是有假设函数$H_θ(x)$,表示的是x发生的几率。若$H_θ(x)$值大于0.5则表示是正样本否则是负样本。
1，对于样本集D={(x1,y1),(x2,y2),...,(xn,yn)}有n个样本，每个样本有x = (x1;x2;...;xd) d个维度，y={y=0,y=1}。在对其进行线性分类的时候有：
$$f(θ,b)=θ_1*x_1+θ_2*x_2+...θ_d*x_d    $$
$$f(θ,b)=θ^Tx = f(z) \tag 1 $$
其中θ,b是要训练的参数。
2，对于二分类来说，f(x)是一个实数，现在欲将其映射到(0,1)的区间上。于是logistic回归的假设为：
$$ y=h_\theta(x)=g(\theta^Tx)= \frac{1}{1+e^{-\theta^Tx}}  \tag{2}$$

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjAxMzU4NzA5?x-oss-process=image/format,png)

sigmoid的好处是理论上可以任何数投影到(0,1)之间。

3，损失函数
通过损失函数来计算样本预测的准确率。
~~由公式(2)可以得到~~ 
$d$ ln\frac{y}{1-y}={\theta^Tx}  \tag{3}$$
~~将y视作后验概率，那么
$d$y=p(y=1|x)$$
$d$1-y=p(y=0|x)$$~~ 
~~那么根据公式(2)，公式(3)可以转变为：
$d$p(y=1|x)=\frac{e^{\theta^Tx}}{1+e^{\theta^Tx}} \tag{4}$$
$d$p(y=0|x)=\frac{1}{1+e^{\theta^Tx}} \tag{5}$$~~ 

将y视作后验概率。那么样本x在条件$θ$下y发生的概率是：
$$p(y=1|x;\theta)=h_\theta(z) \tag{6}$$
y不发生的概率是：
$$p(y=0|x;\theta)=1-h_\theta(z) \tag{7}$$
将(6)(7)合并，就得到样本被预测正确的概率：
$$p(y|x;\theta)=h_\theta(z)^y(1-h_\theta(z))^{1-y}\tag{8}$$

4，极大似然估计
再得到样本的预测概率之后，再使用极大似然法来估计 $\theta,b$。
极大似然估计是令每个样本属于其真实标记的概率越大越好。
由于各观测样本之间相互独立，他们的联合分布为各边缘分布的乘积，因此似然函数为：
$$L(\theta,b)=\quad\prod_{i=1}^n[h_\theta(z)^y(1-h_\theta(z))^{1-y}]$$
为了简化两边取对数并加上符号，变成求取最小值。转变后的公式如下:
$$J(\theta,b)=-ln(L(\theta,b))=- \sum_{i=1}^n[yln(h_\theta(z)) +(1-y_i)*ln(1-h_\theta(z))]  \tag{9}$$
由此得到了LR的损失函数，该函数又被称为**交叉熵**，最大似然损失函数，log损失函数。
4，当然还需要加入正则项来防止过拟合。
$$J(\theta,b)=- \sum_{i=1}^n[yln(h_\theta(z)) +(1-y_i)*ln(1-h_\theta(z))] +\frac{1}{2n}\sum_{i=1}^n \theta_i^2 \tag{9}$$
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjAxNjU3NDYz?x-oss-process=image/format,png)

下面就可以使用随机梯度下降等优化方法来求解最优的参数theta了。

二，正则项小议
L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。
损失函数以均方误差，其加上L1,L2的损失函数为：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjAyOTA0OTIw?x-oss-process=image/format,png)
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjAyOTI4MzU0?x-oss-process=image/format,png)
一般回归分析中回归w表示特征的系数，从上式可以看到正则化项是对系数做了处理（限制）。L1正则化和L2正则化的说明如下：

*** L1正则化是指权值向量w中各个元素的绝对值之和，通常表示为||w||1
*** L2正则化是指权值向量w中各个元素的平方和然后再求平方根（可以看到Ridge回归的L2正则化项有平方符号），通常表示为||w||2


L1和L2正则化的直观理解
假设有如下带L1正则化的损失函数： 
$$ J = J_0 + \alpha \sum_w{|w|}  $$
求解J0的过程可以画出等值线，同时L1正则化的函数L也可以在w1w2的二维平面上画出来。如下图：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjAzNzU0NjMx?x-oss-process=image/format,png)

图中等值线是J0的等值线，黑色方形是L函数的图形。在图中，当$j_0$等值线与L图形首次相交的地方就是最优解。上图中$j_0$与L在L的一个顶点处相交，这个顶点就是最优解。
$j_0$与这些角接触的机率会远大于与L其它部位接触的机率，而在这些角上，会有很多权值等于0，那么对于多维特征中也会存在更多的特征为0。这就是为什么L1正则化可以产生稀疏模型，进而可以用于特征选择。

同样的假设有如下带L2正则化的损失函数： 
$$ J = J_0 + \alpha \sum_w{|w|^2}  $$
可以画出他们在二维平面上的图形，如下：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTEwMjA0MDQ2MDcz?x-oss-process=image/format,png)

二维平面下L2正则化的函数图形是个圆，与方形相比，被磨去了棱角。因此$J_0$与L相交时使得${w_1}$或$w_2$等于零的机率小了许多，这就是为什么L2正则化不具有稀疏性的原因。
因为在最小化损失函数的时候加上了L2正则化因此最终的最优解会是的参数量和参数本身都不会太大，最终达到抑制过拟合的作用。

L1，L2总结：
>* L1正则化是指权值向量w中各个元素的绝对值之和，通常表示为$\sum{||w||}$
>* 使用L1能够得到数据的稀疏特征
>* L2正则化是指权值向量w中各个元素的平方和然后再求平方根（可以看到Ridge回归的L2正则化项有平方符号），通常表示为$\sum{||w||^2}$
>* 使用L2能够比较好的抑制模型过拟合

其他比较好的博文：
**L1,L2：**
http://blog.csdn.net/jinping_shi/article/details/52433975
**LR:**
http://www.cnblogs.com/GuoJiaSheng/p/3928160.html
http://blog.csdn.net/xierhacker/article/details/53316138
