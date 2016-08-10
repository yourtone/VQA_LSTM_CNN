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
cmd:option('-CNNmodel', 'VGG16', 'CNN model')
cmd:option('-layer', 30, 'layer number')
cmd:option('-num_region', 196, 'number of image features per image')
cmd:option('-dim', 512, 'image feature dimension')

cmd:option('-input_npy_train','/home/deepnet/lyt/vqa/feature/VQA/VQA-train2014-VGG16-snake196x512.npy','path to the train npy file')
cmd:option('-input_npy_val','/home/deepnet/lyt/vqa/feature/VQA/VQA-val2014-VGG16-snake196x512.npy','path to the val npy file')
cmd:option('-input_npy_test','/home/deepnet/lyt/vqa/feature/VQA/VQA-test2015-VGG16-snake196x512.npy','path to the test npy file')
cmd:option('-im_path_train','/home/deepnet/lyt/vqa/dataset/data/VQA/done/train2014_im_path.txt','path to the train impath file')
cmd:option('-im_path_val','/home/deepnet/lyt/vqa/dataset/data/VQA/done/val2014_im_path.txt','path to the val impath file')
cmd:option('-im_path_test','/home/deepnet/lyt/vqa/dataset/data/VQA/done/test2015_im_path.txt','path to the test impath file')

opt = cmd:parse(arg)
print(opt)

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

local input_json
local out_name
if opt.subset then
    input_json = string.format('data/data_prepro_sub_s%d.json',opt.split)
    out_name = string.format('data/data_img_sub_s%d_%s_l%d_d%dx%d.h5',
      opt.split,opt.CNNmodel,opt.layer,opt.num_region,opt.dim)
else
    input_json = string.format('data/data_prepro_s%d.json',opt.split)
    out_name = string.format('data/data_img_s%d_%s_l%d_d%dx%d.h5',
      opt.split,opt.CNNmodel,opt.layer,opt.num_region,opt.dim)
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

-- load all image features
local train_npy=npy4th.loadnpy(opt.input_npy_train)
local val_npy
local test_npy
if opt.split ~= 1 then
  val_npy=npy4th.loadnpy(opt.input_npy_val)
  train_npy=torch.cat(train_npy,val_npy,1)
  val_npy = nil
  collectgarbage()
end
print('Training feature size:')
print(train_npy:size())

local ndim1=opt.num_region
local ndim2=opt.dim
local sz=#train_list
local feat_train=torch.Tensor(sz,ndim1,ndim2)
print(string.format('actual processing %d train image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_train[i]=train_npy[train_indlist[train_list[i]]]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)
local train_h5_file = hdf5.open(out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:close()
feat_train = nil
train_npy = nil
collectgarbage()

if opt.split == 1 then
  test_npy=npy4th.loadnpy(opt.input_npy_val)
else--if opt.split ~= 1 then
  test_npy=npy4th.loadnpy(opt.input_npy_test)
end
print('Testing feature size:')
print(test_npy:size())

local sz=#test_list
local feat_test=torch.Tensor(sz,ndim1,ndim2)
print(string.format('actual processing %d test image features...',sz))
for i=1,sz do
  xlua.progress(i, sz)
  feat_test[i]=test_npy[test_indlist[test_list[i]]]:clone()
  --collectgarbage()
end
xlua.progress(sz, sz)
test_npy = nil
collectgarbage()

train_h5_file = hdf5.open(out_name, 'a')
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()
print('save image feature to: '..out_name)
