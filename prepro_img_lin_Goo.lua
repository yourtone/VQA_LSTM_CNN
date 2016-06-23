-- Modified by Yuetan Lin (2016/06/18 15:41)
require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'image'
require 'xlua'
require 'hdf5'
npy4th = require 'npy4th'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_npy_train','s1_GoogLeNet_1024_train.npy','path to the npy file')
cmd:option('-input_npy_test','s1_GoogLeNet_1024_test.npy','path to the npy file')
cmd:option('-out_name', 'data_img.h5', 'output name')

opt = cmd:parse(arg)
print(opt)

-- open the mdf5 file

local feat_train=npy4th.loadnpy(opt.input_npy_train)
local feat_test=npy4th.loadnpy(opt.input_npy_test)
print('Train feature size:')
print(feat_train:size())
print('Test feature size:')
print(feat_test:size())

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()

print('save image feature to: '..opt.out_name)