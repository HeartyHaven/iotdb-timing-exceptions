import sys

import numpy as np

from DataGenerator.numerical_data_generator import generator
data=generator(1,0.1,2.5,0,0.5,100)#生成有一定数值特征的随机数据序列（数值类）
print(data)
from DataGenerator.text_data_generator import text_generator
data1=text_generator(8,20,33,0.1,50)#生成有一定统计学特征的随机文本数据（字符串类）
print(data1)
from FeatureCalculator.numerical_data_feature_calculator import statistic
res=statistic(np.array(data))
print(res)#根据所给数据生成响应的数值特征
from FeatureCalculator.text_data_feature_calculator import cal_feature
res1=cal_feature(data1)#根据文本列表所给数据生成统计特征
print(res1)