
require 'nn'
require 'inn'
require 'cutorch'
require 'cudnn'
require 'image'
matio=require 'matio'
model=torch.load('rgb_model.t7')
require 'mattorch'

w=model.modules[1].modules[1].weight

--size of 64x10x11x11
print '==> visualizing ConvNet filter'
print('Layer 1 filters:')

w=w:double()
matio.save('rgb_filter.mat',w)

print('saved filter')
