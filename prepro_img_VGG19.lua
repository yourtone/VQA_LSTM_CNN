-- Modified by Yuetan Lin (2016/06/18 15:41)
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
cmd:option('-subset', false, 'true: use subset, false: use all dataset')
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')
cmd:option('-num_ans', 1000, 'number of top answers for the final classifications')
cmd:option('-image_root','data/','path to the image root')
cmd:option('-cnn_proto', '/home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('-cnn_model', '/home/deepnet/caffe/models/VGG_19/VGG_ILSVRC_19_layers.caffemodel', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')

cmd:option('-CNNmodel', 'VGG19', 'CNN model')
cmd:option('-layer', 43, 'layer number')
cmd:option('-dim', 4096, 'image feature dimension')

cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:evaluate()
net=net:cuda()
print('#net.modules: ',#net.modules)
for i=1,#net.modules do
    print(i,net.modules[i])
end

function loadim(imname)
    im=image.load(imname)
    im=image.scale(im,224,224)
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
    return im2
end

local image_root = opt.image_root
local input_json
local out_name
if opt.subset then
    input_json = string.format('data/data_prepro_sub_s%d.json',opt.split)
    out_name = string.format('data/data_img_sub_s%d_%s_l%d_d%d.h5',opt.split,opt.CNNmodel,opt.layer,opt.dim)
else
    input_json = string.format('data/data_prepro_s%d.json',opt.split)
    out_name = string.format('data/data_img_s%d_%s_l%d_d%d.h5',opt.split,opt.CNNmodel,opt.layer,opt.dim)
end
local file = io.open(input_json, 'r')
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

local ndims=opt.dim
local batch_size = opt.batch_size
print('DataLoader loading h5 file: ', 'data_train')
local sz=#train_list
local feat_train=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(train_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[opt.layer].output:clone()
    collectgarbage()
end
xlua.progress(sz, sz)
local train_h5_file = hdf5.open(out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:close()
feat_train = nil
collectgarbage()

print('DataLoader loading h5 file: ', 'data_test')
local sz=#test_list
local feat_test=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(test_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[opt.layer].output:clone()
    collectgarbage()
end
xlua.progress(sz, sz)

local train_h5_file = hdf5.open(out_name, 'a')
--train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()
