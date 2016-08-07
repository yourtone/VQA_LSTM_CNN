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
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-pooling_window_size', 2, 'size of pooling window right before bilstm')
cmd:option('-batch_size',50,'batch_size for each iterations')
cmd:option('-max_iters', 50000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 512, 'size of the common embedding vector')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

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

model_path = opt.checkpoint_path
batch_size = opt.batch_size
embedding_size_q = opt.input_encoding_size
lstm_size_q = opt.rnn_size
nlstm_layers_q = opt.rnn_layer
nhimage = opt.imdim
common_embedding_size = opt.common_embedding_size
noutput = opt.num_output
dummy_output_size = 1
decay_factor = 0.99997592083 -- 50000
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local input_name
local input_img_name
if opt.CNNmodel == 'VGG19' then
    input_img_name = string.format('s%d_%s_l%d_d%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim)
elseif opt.CNNmodel == 'GoogLeNet' then
    input_img_name = string.format('s%d_%s_d%d',opt.split,opt.CNNmodel,opt.imdim)
elseif opt.CNNmodel == 'VGG16' then
    input_img_name = string.format('s%d_%s_l%d_d%dx%dx%d',opt.split,opt.CNNmodel,opt.layer,opt.imdim,opt.num_region_width,opt.num_region_height)
else
    print('CNN model name error')
end
if opt.subset then
    input_name = string.format('data_prepro_sub_s%d',opt.split)
    input_img_name = 'sub_' .. input_img_name
else
    input_name = string.format('data_prepro_s%d',opt.split)
end
local input_img_h5 = 'data_img_' .. input_img_name .. 'norm.h5'
local input_ques_h5 = input_name .. '.h5'
local input_json = input_name .. '.json'
local CP_name = string.format('lstm_'..input_img_name..'_es%d_rs%d_rl%d_cs%d_bs%d_iter%%d.t7',
    opt.input_encoding_size,opt.rnn_size,opt.rnn_layer,opt.common_embedding_size,opt.batch_size)
local final_model_name = string.format('lstm_'..input_img_name..'_es%d_rs%d_rl%d_cs%d_bs%d.t7',
    opt.input_encoding_size,opt.rnn_size,opt.rnn_layer,opt.common_embedding_size,opt.batch_size)

print('DataLoader loading h5 file: ', input_json)
file = io.open(input_json, 'r')
text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', input_ques_h5)
dataset = {}
h5_file = hdf5.open(input_ques_h5, 'r')
dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
h5_file:close()

print('DataLoader loading h5 file: ', input_img_h5)
h5_file = hdf5.open(input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Normalize the image feature
--if opt.img_norm == 1 then
--  local nm = torch.norm(dataset['fv_im'], 2, 2)
--  nm[nm:eq(0)]=1e-5
--  dataset['fv_im']=torch.cdiv(dataset['fv_im'], torch.repeatTensor(nm,1,nhimage,1,1)):float()
--end
--assert(torch.sum(dataset['fv_im']:ne(dataset['fv_im']))==0)

count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
vocabulary_size_q=count

collectgarbage()

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

--Network definitions
--VQA
--embedding: word-embedding
embedding_net_q=nn.Sequential()
        :add(nn.Linear(vocabulary_size_q,embedding_size_q))
        :add(nn.Dropout(0.5))
        :add(nn.Tanh())

--encoder: RNN body
encoder_net_q=LSTM.lstm_conventional(embedding_size_q,lstm_size_q,dummy_output_size,nlstm_layers_q,0.5)

--multimodal way of combining different spaces
local nhquestion = 2 * lstm_size_q * nlstm_layers_q
local grid_height = opt.num_region_height
local grid_width = opt.num_region_width
local ws = opt.pooling_window_size
multimodal_net=nn.Sequential()
        :add(netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5))
        :add(nn.SpatialMaxPooling(ws, ws, ws, ws))
        :add(nn.Tanh())
        :add(nn.Zigzag())
        :add(nn.SplitTable(2, 2))

-- correlate the multimodel features by bidirection lstm.
local num_selected_region = (grid_height/ws)*(grid_width/ws)
fusion_net = nn.Bilstm(common_embedding_size, lstm_size_q, common_embedding_size/2, 
                       1, num_selected_region, 0.5, opt.gpuid)

-- answer generation
add_new_index = nn.ParallelTable()
for i=1,num_selected_region do
    add_new_index:add(nn.Reshape(1, common_embedding_size))
end
answer_net=nn.Sequential()
        :add(add_new_index)
        :add(nn.JoinTable(1, 2))
        :add(nn.Reshape(1,num_selected_region,common_embedding_size))
        :add(nn.SpatialMaxPooling(1,num_selected_region))
        :add(nn.Squeeze())
        :add(nn.Dropout(0.5))
        :add(nn.Linear(common_embedding_size,noutput))

--criterion
criterion=nn.CrossEntropyCriterion()

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

init_state_f = {
      torch.zeros(batch_size, 2*lstm_size_q*1),
      torch.zeros(batch_size, 2*lstm_size_q*1),
}
dummy_dend_state_f = {
      torch.zeros(batch_size, 2*lstm_size_q*1),
      torch.zeros(batch_size, 2*lstm_size_q*1),
}
batch_per_step_f = torch.LongTensor(num_selected_region):fill(batch_size)

if opt.gpuid >= 0 then
  print('shipped data function to cuda...')
  embedding_net_q = embedding_net_q:cuda()
  encoder_net_q = encoder_net_q:cuda()
  multimodal_net = multimodal_net:cuda()
  answer_net = answer_net:cuda()
  criterion = criterion:cuda()
  dummy_state_q = dummy_state_q:cuda()
  dummy_output_q = dummy_output_q:cuda()
  init_state_f[1] = init_state_f[1]:cuda()
  init_state_f[2] = init_state_f[2]:cuda()
  dummy_dend_state_f[1] = dummy_dend_state_f[1]:cuda()
  dummy_dend_state_f[2] = dummy_dend_state_f[2]:cuda()
end

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters()
embedding_w_q:uniform(-0.08, 0.08);

encoder_w_q,encoder_dw_q=encoder_net_q:getParameters()
encoder_w_q:uniform(-0.08, 0.08);

multimodal_w,multimodal_dw=multimodal_net:getParameters()
multimodal_w:uniform(-0.08, 0.08);

fusion_w, fusion_dw = fusion_net:getParameters()
fusion_w:uniform(-0.08, 0.08);

answer_w,answer_dw=answer_net:getParameters()
answer_w:uniform(-0.08, 0.08);

sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1),
       fusion_w:size(1),answer_w:size(1)}


-- optimization parameter
optimize={}
optimize.maxIter=opt.max_iters
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1

optimize.winit=join_vector({encoder_w_q,embedding_w_q,multimodal_w,fusion_w,answer_w})


------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()
  local qinds=torch.LongTensor(batch_size):fill(0)
  local iminds=torch.LongTensor(batch_size):fill(0)

  local nqs=dataset['question']:size(1)
  -- we use the last val_num data for validation (the data already randomlized when created)
  --
  for i=1,batch_size do
    qinds[i]=torch.random(nqs)
    iminds[i]=dataset['img_list'][qinds[i]]
  end

  local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question']:index(1,qinds),dataset['lengths_q']:index(1,qinds),vocabulary_size_q)
  local fv_im=dataset['fv_im']:index(1,iminds)
  local labels=dataset['answers']:index(1,qinds)

  -- ship to gpu
  if opt.gpuid >= 0 then
    fv_sorted_q[1]=fv_sorted_q[1]:cuda()
    fv_sorted_q[3]=fv_sorted_q[3]:cuda()
    fv_sorted_q[4]=fv_sorted_q[4]:cuda()
    fv_im = fv_im:cuda()
    labels = labels:cuda()
  end
  
  return fv_sorted_q, fv_im, labels, batch_size
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- duplicate the RNN
encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q)

-- Objective function
function JdJ(x)
  local params=split_vector(x,sizes)
  --load x to net parameters--
  if encoder_w_q~=params[1] then
    encoder_w_q:copy(params[1])
    for i=1,buffer_size_q do
      encoder_net_buffer_q[2][i]:copy(params[1])
    end
  end
  if embedding_w_q~=params[2] then
    embedding_w_q:copy(params[2])
  end
  if multimodal_w~=params[3] then
    multimodal_w:copy(params[3])
  end
  fusion_net:updateParameters(params[4])
  if answer_w~=params[5] then
    answer_w:copy(params[5])
  end

  --clear gradients--
  for i=1,buffer_size_q do
    encoder_net_buffer_q[3][i]:zero()
  end
  embedding_dw_q:zero()
  multimodal_dw:zero()
  fusion_net:zeroGradParameters()
  answer_dw:zero()

  --grab a batch--
  local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch()
  local question_max_length=fv_sorted_q[2]:size(1)

  --embedding forward--
  local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2])

  --encoder forward--
  local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2])

  --multimodal forward--
  local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4])

  --fusion forward--
  local fusion_fea=multimodal_net:forward({tv_q,fv_im})
  init_state_f[1]:zero()
  init_state_f[2]:zero()
  local states_f, outputs_f = fusion_net:forward(init_state_f, fusion_fea, batch_per_step_f)

  --answer/criterion forward--
  local scores=answer_net:forward(outputs_f)
  local f=criterion:forward(scores,labels)

  --answer/creterion backward--
  local dscores=criterion:backward(scores,labels)
  local doutputs_f=answer_net:backward(outputs_f,dscores)

  --fusion backward--
  dummy_dend_state_f[1]:zero()
  dummy_dend_state_f[2]:zero()
  local _, dfusion_fea = fusion_net:backward(dummy_dend_state_f, doutputs_f, states_f, fusion_fea, batch_per_step_f)

  --multimodal backward--
  local tmp=multimodal_net:backward({tv_q,fv_im},dfusion_fea)
  local dtv_q=tmp[1]:index(1,fv_sorted_q[3])

  --encoder backward--
  local junk4,dword_embedding_q=rnn_backward(encoder_net_buffer_q,dtv_q,dummy_output_q,states_q,word_embedding_q,fv_sorted_q[2])

  --embedding backward--
  dword_embedding_q=join_vector(dword_embedding_q)
  embedding_net_q:backward(fv_sorted_q[1],dword_embedding_q)

  --summarize f and gradient
  local encoder_adw_q=encoder_dw_q:clone():zero()
  for i=1,question_max_length do
    encoder_adw_q=encoder_adw_q+encoder_net_buffer_q[3][i]
  end

  gradients=join_vector({encoder_adw_q,embedding_dw_q,multimodal_dw,fusion_dw,answer_dw})
  gradients:clamp(-10,10)
  if running_avg == nil then
    running_avg = f
  end
  running_avg=running_avg*0.95+f*0.05
  return f,gradients
end


----------------------------------------------------------------------------------------------
-- Training
----------------------------------------------------------------------------------------------
-- With current setting, the network seems never overfitting, so we just use all the data to train

state={}
for iter = 1, opt.max_iters do
  if iter%opt.save_checkpoint_every == 0 then
    paths.mkdir(model_path..'save')
    torch.save(string.format(model_path..'save/'..CP_name,iter),
      {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w,
       fusion_w=fusion_w,answer_w=answer_w})
  end
  if iter%100 == 0 then
    print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
  end
  optim.rmsprop(JdJ, optimize.winit, optimize, state)
  optimize.learningRate=optimize.learningRate*decay_factor
  if iter%50 == 0 then -- change this to smaller value if out of the memory
    collectgarbage()
  end
end

-- Saving the final model
torch.save(string.format(model_path..final_model_name),
  {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w,
   fusion_w=fusion_w,answer_w=answer_w})
