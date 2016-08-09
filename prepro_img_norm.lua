-- Modified by Yuetan Lin (2016/06/18 22:06)
require 'nn';
require 'torch';
require 'nngraph';
require 'optim';
require 'misc.netdef';
require 'cutorch';
require 'cunn';
require 'hdf5';
require 'misc.Bilstm';
require 'misc.Zigzag';
cjson=require('cjson');
LSTM=require 'misc.LSTM';

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-subset', false, 'true: use subset, false: use all dataset')
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-CNNmodel', 'VGG16', 'CNN model')
cmd:option('-layer', 30, 'layer number')
cmd:option('-num_region_width', 14, 'number of image regions in the side of width')
cmd:option('-num_region_height', 14, 'number of image regions in the side of heigth')
cmd:option('-imdim', 512, 'image feature dimension')

-- Model parameter settings
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
nhimage = opt.imdim

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local input_img_name
if opt.CNNmodel == 'VGG19' then
    input_img_name = string.format('s%d_%s_l%d_d%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim)
elseif opt.CNNmodel == 'GoogLeNet' then
    input_img_name = string.format('s%d_%s_d%d',opt.split,opt.CNNmodel,opt.imdim)
elseif opt.CNNmodel == 'VGG16' or opt.CNNmodel == 'VGG19R' then
    input_img_name = string.format('s%d_%s_l%d_d%dx%dx%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim,opt.num_region_height,opt.num_region_width)
else
    print('CNN model name error')
end
if opt.subset then
    input_img_name = 'sub_' .. input_img_name
end
input_img_h5 = 'data_img_' .. input_img_name .. '.h5'
output_img_h5 = 'data_img_' .. input_img_name .. 'norm.h5'

print('Loading: ', input_img_h5)
-------------------------------------
-- Train
-------------------------------------
print('Load train image feature ...')
h5_file = hdf5.open(input_img_h5, 'r')
feat_train = h5_file:read('/images_train'):all()
h5_file:close()

-- Normalize the image feature
print('Normalize train image feature ...')
if opt.img_norm == 1 then
  local nm = torch.norm(feat_train, 2, 2)
  nm[nm:eq(0)]=1e-5
  if opt.CNNmodel == 'VGG19' or opt.CNNmodel == 'GoogLeNet' then
    feat_train=torch.cdiv(feat_train, torch.repeatTensor(nm,1,nhimage)):float()
  elseif opt.CNNmodel == 'VGG16' or opt.CNNmodel == 'VGG19R' then
    feat_train=torch.cdiv(feat_train, torch.repeatTensor(nm,1,nhimage,1,1)):float()
  end
end
print('Assert no NaN item ...')
assert(torch.sum(feat_train:ne(feat_train))==0)

print('Write train image feature ...')
h5_file = hdf5.open(output_img_h5, 'w')
h5_file:write('/images_train', feat_train:float())
h5_file:close()
feat_train = nil
collectgarbage()

-------------------------------------
-- Test
-------------------------------------
print('Load test image feature ...')
h5_file = hdf5.open(input_img_h5, 'r')
feat_test = h5_file:read('/images_test'):all()
h5_file:close()

-- Normalize the image feature
print('Normalize test image feature ...')
if opt.img_norm == 1 then
  local nm = torch.norm(feat_test, 2, 2)
  nm[nm:eq(0)]=1e-5
  if opt.CNNmodel == 'VGG19' or opt.CNNmodel == 'GoogLeNet' then
    feat_test=torch.cdiv(feat_test, torch.repeatTensor(nm,1,nhimage)):float()
  elseif opt.CNNmodel == 'VGG16' or opt.CNNmodel == 'VGG19R' then
    feat_test=torch.cdiv(feat_test, torch.repeatTensor(nm,1,nhimage,1,1)):float()
  end
end
print('Assert no NaN item ...')
assert(torch.sum(feat_test:ne(feat_test))==0)

print('Write test image feature ...')
h5_file = hdf5.open(output_img_h5, 'a')
h5_file:write('/images_test', feat_test:float())
h5_file:close()
feat_test = nil
collectgarbage()
