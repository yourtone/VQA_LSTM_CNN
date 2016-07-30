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
cmd:option('-input_json','data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-input_npy_train','VQA-train2014-VGG16-snake196x512.npy','path to the train npy file')
cmd:option('-input_npy_val','VQA-val2014-VGG16-snake196x512.npy','path to the val npy file')
cmd:option('-input_npy_test','VQA-test2015-VGG16-snake196x512.npy','path to the test npy file')
cmd:option('-im_path_train','train_im_paths.txt','path to the train impath file')
cmd:option('-im_path_val','val_im_paths.txt','path to the val impath file')
cmd:option('-im_path_test','test_im_paths.txt','path to the test impath file')

cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Train+Val and test on Test, 3: train on Train+Val and test on Test-dev')
cmd:option('-dim1', 196, 'number of image features per image')
cmd:option('-dim2', 512, 'image feature dimension')

cmd:option('-out_name', 'data_img.h5', 'output name')

opt = cmd:parse(arg)
print(opt)

-- load all image features
local train_npy=npy4th.loadnpy(opt.input_npy_train)
local val_npy=npy4th.loadnpy(opt.input_npy_val)
local test_npy
if opt.split == 1 then
  test_npy=val_npy:clone()
else
  train_npy=torch.cat(train_npy,val_npy,1)
  test_npy=npy4th.loadnpy(opt.input_npy_test)
end
print('Training feature size:')
print(train_npy:size())
print('Testing feature size:')
print(test_npy:size())

-- load impath index
local train_imlist = {}
local test_imlist = {}
local f = io.open(opt.im_path_train,'r')
assert(f ~= nil)
local line = f:read('*line')
while line ~= nil do
  table.insert(train_imlist, line)
  line = f:read('*line')
end
f:close()
if opt.split == 1 then
  f = io.open(opt.im_path_val,'r')
  assert(f ~= nil)
  line = f:read('*line')
  while line ~= nil do
    table.insert(test_imlist, line)
    line = f:read('*line')
  end
  f:close()
else
  f = io.open(opt.im_path_val,'r')
  assert(f ~= nil)
  line = f:read('*line')
  while line ~= nil do
    table.insert(train_imlist, line)
    line = f:read('*line')
  end
  f:close()

  f = io.open(opt.im_path_test,'r')
  assert(f ~= nil)
  line = f:read('*line')
  while line ~= nil do
    table.insert(test_imlist, line)
    line = f:read('*line')
  end
  f:close()
end

-- swap index and value
local train_indlist = {}
local sz=#train_imlist
--print(string.format('Total number of train image: %d', sz))
for i = 1,sz do
  train_indlist[train_imlist[i]] = i
end
-- E.g. train_indlist['train2014/COCO_train2014_000000487025.jpg'] = 1
local test_indlist = {}
sz=#test_imlist
--print(string.format('Total number of test image: %d', sz))
for i = 1,sz do
  test_indlist[test_imlist[i]] = i
end

-- open the json file
local f = io.open(opt.input_json, 'r')
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

local ndim1=opt.dim1
local ndim2=opt.dim2
local sz=#train_list
local feat_train=torch.Tensor(sz,ndim1,ndim2)
print(string.format('actual processing %d train image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_train[i]=train_npy[train_indlist[train_list[i]]]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)

local sz=#test_list
local feat_test=torch.Tensor(sz,ndim1,ndim2)
print(string.format('actual processing %d test image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_test[i]=test_npy[test_indlist[test_list[i]]]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)
collectgarbage()

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()
print('save image feature to: '..opt.out_name)
