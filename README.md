# ensemble
1.首先训练各源域至目标域的分类器
run.py中执行tr.train函数

2.优化分类器权重
由于优化部分代码，暂且还在做另一个实验，此处，给出本文中权重如下：
B为源域时：D/K/E的权重为 0.4/0.5/0.1
K为源域时：B/E/D的权重为 0.5/0.15/0.35
D为源域时：B/E/K权重为为 0.4/0.3/0.3
E为源域时：B/D/K的权重为 0.45/0.4/0.15

3.集成各分类器权重完成多源任务
run.py中执行ensemble函数
