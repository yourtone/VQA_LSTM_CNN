-- Modified by Yuetan Lin (2016/07/20 16:03)
require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'image'
require 'hdf5'
require 'xlua'
cjson = require('cjson')
npy4th = require 'npy4th'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-subset', false, 'true: use subset, false: use all dataset')
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')
cmd:option('-num_ans', 1000, 'number of top answers for the final classifications')
cmd:option('-CNNmodel', 'GoogLeNet', 'CNN model')
cmd:option('-dim', 1024, 'image feature dimension')
--cmd:option('-input_npy_trainval','VQA-GoogLeNet-1000.npy','path to the train+val npy file')
--cmd:option('-input_npy_test','VQA-test2015-GoogLeNet-1000.npy','path to the test npy file')
--cmd:option('-im_path_trainval','trainval_im_paths.txt','path to the train+val impath file')
--cmd:option('-im_path_test','test_im_paths.txt','path to the test impath file')

opt = cmd:parse(arg)
print(opt)

local input_npy_trainval = string.format('/home/deepnet/lyt/vqa/feature/VQA/VQA-GoogLeNet-%d.npy',opt.dim)
local input_npy_test = string.format('/home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-GoogLeNet-%d.npy',opt.dim)
local im_path_trainval = string.format('/home/deepnet/lyt/vqa/feature/VQA/trainval_im_paths.txt')
local im_path_test = string.format('/home/deepnet/lyt/vqa/feature/VQA/test_im_paths.txt')

-- load all image features
local trainval_npy=npy4th.loadnpy(input_npy_trainval)
local test_npy
if opt.split == 1 then
  test_npy=trainval_npy:clone()
else
  test_npy=npy4th.loadnpy(input_npy_test)
end
print('TrainVal feature size:')
print(trainval_npy:size())
print('Test feature size:')
print(test_npy:size())

-- load impath index
local trainval_imlist = {}
local test_imlist = {}
local f = io.open(im_path_trainval,'r')
assert(f ~= nil)
local line = f:read('*line')
while line ~= nil do
  table.insert(trainval_imlist, line)
  line = f:read('*line')
end
f:close()
if opt.split == 1 then
  test_imlist=trainval_imlist
else
  f = io.open(im_path_test,'r')
  assert(f ~= nil)
  line = f:read('*line')
  while line ~= nil do
    table.insert(test_imlist, line)
    line = f:read('*line')
  end
  f:close()
end

-- swap index and value
local trainval_indlist = {}
local sz=#trainval_imlist
--print(string.format('Total number of train+val image: %d', sz))
for i = 1,sz do
  trainval_indlist[trainval_imlist[i]] = i
end
-- E.g. trainval_indlist['train2014/COCO_train2014_000000487025.jpg'] = 1
local test_indlist = {}
sz=#test_imlist
--print(string.format('Total number of test image: %d', sz))
for i = 1,sz do
  test_indlist[test_imlist[i]] = i
end

local input_json
local out_name
if opt.subset then
    input_json = string.format('data_prepro_sub_s%d.json',opt.split)
    out_name = string.format('data_img_sub_s%d_%s_d%d.h5',opt.split,opt.CNNmodel,opt.dim)
else
    input_json = string.format('data_prepro_s%d.json',opt.split)
    out_name = string.format('data_img_s%d_%s_d%d.h5',opt.split,opt.CNNmodel,opt.dim)
end

-- open the json file
local f = io.open(input_json, 'r')
local text = f:read()
f:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
  table.insert(train_list, imname)
end
local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
  table.insert(test_list, imname)
end

local ndims=opt.dim
local sz=#train_list
local feat_train=torch.Tensor(sz,ndims)
print(string.format('actual processing %d train image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_train[{i,{}}]=trainval_npy[{trainval_indlist[train_list[i]],{}}]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)

local sz=#test_list
local feat_test=torch.Tensor(sz,ndims)
print(string.format('actual processing %d test image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_test[{i,{}}]=test_npy[{test_indlist[test_list[i]],{}}]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)
collectgarbage()

local train_h5_file = hdf5.open(out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()
print('save image feature to: '..out_name)
