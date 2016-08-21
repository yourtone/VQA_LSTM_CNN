require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','','path to the image root')
cmd:option('-cnn_proto', '', 'path to the cnn prototxt')
cmd:option('-cnn_model', '', 'path to the cnn model')

cmd:option('-out_name', 'data_img.h5', 'output name')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:evaluate()
net=net:cuda()
print('#net.modules: ',#net.modules)

function loadim(imname)
    im=image.load(imname)
    im=image.scale(im,448,448)
    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939

    local regions = torch.Tensor(9, 3, 224, 224)
    for i=1,3 do
        for j=1,3 do
            regions[{{(i-1)*3+j},{},{},{}}] = im2[{{},{(i-1)*112+1,(i+1)*112},{(j-1)*112+1,(j+1)*112}}]
        end
    end
    return regions
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end

local ndims=4096
print('DataLoader loading h5 file: ', 'data_train')
local sz=#train_list
local feat_train=torch.Tensor(sz,9,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz do
    xlua.progress(i, sz)
    local regions=loadim(train_list[i]):cuda()
    net:forward(regions)
    feat_train[{{i},{},{}}]=net.modules[43].output:clone():float()
    collectgarbage()
end
xlua.progress(sz, sz)
feat_train = feat_train:transpose(2, 3):resize(sz, ndims, 3, 3)
local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:close()
feat_train = nil
collectgarbage()

print('DataLoader loading h5 file: ', 'data_test')
local sz=#test_list
local feat_test=torch.Tensor(sz,9,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz do
    xlua.progress(i, sz)    
    local regions=loadim(test_list[i]):cuda()
    net:forward(regions)
    feat_test[{{i},{},{}}]=net.modules[43].output:clone():float()
    collectgarbage()
end
xlua.progress(sz, sz)
feat_test = feat_test:transpose(2, 3):resize(sz, ndims, 3, 3)
local train_h5_file = hdf5.open(opt.out_name, 'a')
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()

