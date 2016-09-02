local input_name
local input_img_name
if opt.CNNmodel == 'VGG19' then
  input_img_name = string.format('s%d_%s_l%d_d%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim)
elseif opt.CNNmodel == 'GoogLeNet' then
  input_img_name = string.format('s%d_%s_d%d',opt.split,opt.CNNmodel,opt.imdim)
elseif opt.CNNmodel == 'VGG16' or opt.CNNmodel == 'VGG19R' or opt.CNNmodel == 'ResNet152' then
  input_img_name = string.format('s%d_%s_l%d_d%dx%dx%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim,opt.num_region_height,opt.num_region_width)
else
  print('CNN model name error')
end
if opt.subset then
  input_name = string.format('data_prepro_sub_s%d',opt.split)
  input_img_name = 'sub_' .. input_img_name
else
  input_name = string.format('data_prepro_s%d',opt.split)
end
local param_name = string.format('lstm_'..input_img_name..'_%s_es%d_rs%d_rl%d_cs%d_bs%d',
  opt.netmodel,opt.input_encoding_size,opt.rnn_size,opt.rnn_layer,opt.common_embedding_size,opt.batch_size)

if opt.img_norm == 1 then
  input_img_h5 = 'data_img_' .. input_img_name .. 'norm.h5'
else
  input_img_h5 = 'data_img_' .. input_img_name .. '.h5'
end
input_img_h5 = 'data/' .. input_img_h5
input_ques_h5 = 'data/' .. input_name .. '.h5'
input_json = 'data/' .. input_name .. '.json'
CP_name = param_name..'_iter%d.t7'
final_model_name = param_name..'.t7'
local choice
if opt.dotrain then
  choice = 'train'
else
  choice = 'test'
end
result_name = param_name..'_'..choice..'_results.json'
