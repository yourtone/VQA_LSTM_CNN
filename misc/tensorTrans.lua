require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Transform tensor file')
cmd:text()
cmd:text('Options')

cmd:option('-input','data_img_s1_VGG16_l30_d512x14x14_o1k_sub.h5','path to the original image feature')
cmd:option('-output','data_img_s1_VGG16_l30_snake196x512_o1k_sub.h5','path to the new image feature')
cmd:option('-showper',1000,'show per #samples')

opt = cmd:parse(arg)
print(opt)

-- Train feature
print('DataLoader loading h5 file - /images_train: ', opt.input)
local h5_file = hdf5.open(opt.input, 'r')
tensorold = h5_file:read('/images_train'):all() -- Nx512x14x14
h5_file:close()

local N=tensorold:size(1)
local H=tensorold:size(3)
local W=tensorold:size(4)
local D=tensorold:size(2)
tensornew=torch.Tensor(N,H*W,D)
print(string.format('OldTensor:\t%dx%dx%dx%d',N,D,H,W))
print(string.format('NewTensor:\t%dx%dx%d',N,H*W,D))
for n=1,N do
  if n%opt.showper==0 then
    print(n)
    collectgarbage()
  end
  tmp=1
  for h=1,H,2 do
    for w=1,W do
      tensornew[n][tmp]=tensorold[{n,{},h,w}]
      tmp=tmp+1
    end
    for w=1,W do
      tensornew[n][tmp]=tensorold[{n,{},h+1,W-w+1}]
      tmp=tmp+1
    end
  end
end
tensorold=nil
collectgarbage()

h5_file = hdf5.open(opt.output, 'w')
h5_file:write('/images_train', tensornew:float())
h5_file:close()
tensornew=nil
collectgarbage()

-- Test feature
print('DataLoader loading h5 file - /images_test: ', opt.input)
h5_file = hdf5.open(opt.input, 'r')
tensorold = h5_file:read('/images_test'):all() -- Nx512x14x14
h5_file:close()

N=tensorold:size(1)
H=tensorold:size(3)
W=tensorold:size(4)
D=tensorold:size(2)
tensornew=torch.Tensor(N,H*W,D)
print(string.format('OldTensor:\t%dx%dx%dx%d',N,D,H,W))
print(string.format('NewTensor:\t%dx%dx%d',N,H*W,D))
for n=1,N do
  if n%opt.showper==0 then
    print(n)
    collectgarbage()
  end
  tmp=1
  for h=1,H,2 do
    for w=1,W do
      tensornew[n][tmp]=tensorold[{n,{},h,w}]
      tmp=tmp+1
    end
    for w=1,W do
      tensornew[n][tmp]=tensorold[{n,{},h+1,W-w+1}]
      tmp=tmp+1
    end
  end
end
tensorold=nil
collectgarbage()

h5_file = hdf5.open(opt.output, 'a')
h5_file:write('/images_test', tensornew:float())
h5_file:close()

-- Script --
--th misc/tensorTrans.lua -input data_img_s1_VGG16_l30_d512x14x14_o1k_sub.h5 -output data_img_s1_VGG16_l30_snake196x512_o1k_sub.h5


-- Test --
--tensorold=torch.rand(5,2,4,3)
--tensortrans=nn.Transpose({2,3},{3,4}):forward(tensorold)
--tensortrans:size()
--N=tensortrans:size(1)
--H=tensortrans:size(2)
--W=tensortrans:size(3)
--D=tensortrans:size(4)
--tensornew=torch.Tensor(N,H*W,D)
--for n=1,N do
--  tmp=1
--  for h=1,H,2 do
--    for w=1,W do
--      tensornew[n][tmp]=tensortrans[n][h][w]
--      tmp=tmp+1
--    end
--    for w=1,W do
--      tensornew[n][tmp]=tensortrans[n][h+1][W-w+1]
--      tmp=tmp+1
--    end
--  end
--end
