import utils.hotfeat as hotfeat
import torch

#随机生成一个张量：
feats=torch.rand(16,32,88,88) # 四维，batchsize,channel,h,w
#生成张量热力图：
path=hotfeat.feature_vis(feats, 'test')
print('已保存至',path)