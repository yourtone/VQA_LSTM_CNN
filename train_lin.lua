-- Modified by Yuetan Lin (2016/06/18 22:06)
require 'nn';
require 'torch';
require 'nngraph';
require 'optim';
require 'misc.netdef';
require 'cutorch';
require 'cunn';
require 'hdf5';
cjson=require('cjson');
LSTM=require 'misc.LSTM';
require 'misc.Zigzag';
require 'misc.Unzigzag';
require 'rnn';

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Configuration file
cmd:option('-conf_file', '', 'configuration file path')

-- Data input settings
cmd:option('-subset', false, 'true: use subset, false: use all dataset')
cmd:option('-split', 1, '1: train on Train and test on Val, 2: train on Tr+V and test on Te, 3: train on Tr+V and test on Te-dev')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-CNNmodel', 'VGG19R', 'CNN model')
cmd:option('-layer', 43, 'layer number')
cmd:option('-imdim', 4096, 'image feature dimension')
cmd:option('-num_region_width', 3, 'number of image regions in the side of width')
cmd:option('-num_region_height', 3, 'number of image regions in the side of heigth')
cmd:option('-netmodel', 'RegMax', 'holistic|RegMax|RegSpa|SalMax|SalSpa|RegMaxQ|RegSpaQ|SalMaxQ|SalSpaQ|QSalMax')
-- holistic: baseline (holistic image feature .* question feature)
-- Reg/Sal: region / saliency Bi-LSTM
-- Pool/Spa: max pooling / spatial Bi-LSTM
-- Q: concatenate question

-- Model parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-weightdecay', 5e-4, 'weight decay')
cmd:option('-optim_method', 'rmsprop', 'adadelta|rmsprop')
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-max_iters', 50000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

opt = cmd:parse(arg)
if opt.conf_file ~= '' then
  local conf = dofile(opt.conf_file)
  local default_opt = cmd:default()
  for k,v in pairs(conf) do
    default_opt[k] = v
  end
  opt = default_opt
end
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
local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=opt.imdim
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000
local optim_method = opt.optim_method
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
require 'misc.autoNaming'

print('DataLoader loading h5 file: ', input_json)
local file = io.open(input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(input_ques_h5, 'r')
dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
h5_file:close()

print('DataLoader loading h5 file: ', input_img_h5)
local h5_file = hdf5.open(input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count
collectgarbage();

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

--MULTIMODAL
--multimodal way of combining different spaces
local nhquestion = 2 * lstm_size_q * nlstm_layers_q
local grid_height = opt.num_region_height
local grid_width = opt.num_region_width
if opt.netmodel == 'holistic' then
  multimodal_net=nn.Sequential()
    :add(netdef.AxB(nhquestion,nhimage,common_embedding_size,0.5))
    :add(nn.Dropout(0.5))
    :add(nn.Linear(common_embedding_size,noutput))
elseif opt.netmodel == 'RegMax' then
--  multimodal_net=nn.Sequential()
--    :add(netdef.AxBB(nhquestion,nhimage,opt.num_region,common_embedding_size,0.5))
--    :add(nn.Dropout(0.5))
--    :add(nn.Linear(common_embedding_size,noutput))
  multimodal_net=nn.Sequential()
    :add(netdef.Qx2DII(nhquestion,nhimage,grid_height,grid_width,common_embedding_size,0.5))
    :add(nn.Tanh())
    :add(nn.SpatialMaxPooling(grid_width,grid_height))
    :add(nn.Squeeze())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(common_embedding_size,noutput))
elseif opt.netmodel == 'RegSpa' then
  multimodal_net=nn.Sequential()
    :add(netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5))
    :add(nn.Tanh())
    :add(nn.Zigzag())
    :add(nn.SplitTable(2, 2))
  -- correlate the multimodel features by bidirection lstm.
  local num_selected_region = grid_height*grid_width
  nn.FastLSTM.usenngraph = true
  cell = nn.FastLSTM(common_embedding_size, common_embedding_size/2)
  fusion_net = nn.BiSequencer(cell)
  multimodal_net:add(fusion_net)
  -- answer generation
  add_new_index = nn.ParallelTable()
  for i=1,num_selected_region do
    add_new_index:add(nn.Reshape(1, common_embedding_size))
  end
  multimodal_net:add(add_new_index)
    :add(nn.JoinTable(1, 2))
    :add(nn.Reshape(1, num_selected_region, common_embedding_size))
    :add(nn.SpatialMaxPooling(1, num_selected_region))
    :add(nn.Squeeze())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(common_embedding_size, noutput))
elseif opt.netmodel == 'SalMax' then
    multimodal_net=nn.Sequential()
      :add(nn.ParallelTable()
             :add(nn.Identity())
             :add(nn.Sequential()
                      :add(nn.ConcatTable()
                               :add(netdef.salient_weight(nhimage))
                               :add(nn.Identity()))
                      :add(netdef.attend(nhimage, grid_height, grid_width))))
      :add(netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5))
      :add(nn.Tanh())
      :add(nn.SpatialMaxPooling(grid_width, grid_height))
      :add(nn.Squeeze())
      :add(nn.Dropout(0.5))
      :add(nn.Linear(common_embedding_size, noutput))
elseif opt.netmodel == 'SalMaxQ' then
    q = nn.Identity()()
    i = nn.Identity()()
    salient_i = netdef.attend(nhimage, grid_height, grid_width)({netdef.salient_weight(nhimage)(i), i})
    mul_fea = netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5)({q, salient_i})
    fusion_fea = nn.Dropout(0.5)(
                   nn.Squeeze()(
                     nn.SpatialMaxPooling(grid_width, grid_height)(
                       nn.Tanh()(mul_fea))))
    concat_fea = nn.JoinTable(1, 1)({fusion_fea, nn.Linear(nhquestion, common_embedding_size)(q)})
    scores = nn.Linear(2*common_embedding_size, noutput)(concat_fea)
    multimodal_net = nn.gModule({q, i}, {scores})
elseif opt.netmodel == 'QSalMax' then
    q = nn.Identity()()
    i = nn.Identity()()
    q_i = nn.JoinTable(1, 3)(
            {nn.Reshape(nhquestion, grid_height, grid_width)(
             nn.Replicate(grid_height*grid_width, 2, 1)(q)), i})
    salient_i = netdef.attend(nhimage, grid_height, grid_width)(
            {netdef.salient_weight(nhimage+nhquestion)(q_i), i})
    mul_fea = netdef.Qx2DII(nhquestion, nhimage, grid_height, 
                            grid_width, common_embedding_size, 0.5)({q, salient_i})
    fusion_fea = nn.Dropout(0.5)(
                 nn.Squeeze()(
                 nn.SpatialMaxPooling(grid_width, grid_height)(
                 nn.Tanh()(mul_fea))))
    scores = nn.Linear(common_embedding_size, noutput)(fusion_fea)
    multimodal_net = nn.gModule({q, i}, {scores})
elseif opt.netmodel == 'SalSpa' then
    q = nn.Identity()()
    i = nn.Identity()()
    salient_i = netdef.attend(nhimage, grid_height, grid_width)(
        {netdef.salient_weight(nhimage)(i), i})
    mul_fea = netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, 
                            common_embedding_size, 0.5)({q, salient_i})
    fusion_fea = nn.Dropout(0.5)(
                 nn.Squeeze()(
                 nn.SpatialMaxPooling(grid_width, grid_height)(
                 nn.Reshape(common_embedding_size, grid_height, grid_width)(
                 nn.Transpose({2,3})(
                 nn.SeqBRNN(common_embedding_size, common_embedding_size, true)(
                 nn.Transpose({2,3})(
                 nn.Reshape(common_embedding_size, grid_height*grid_width)(mul_fea))))))))
    scores = nn.Linear(common_embedding_size, noutput)(fusion_fea)
    multimodal_net = nn.gModule({q, i}, {scores})
else
  print('ERROR: netmodel is not defined: '..opt.netmodel)
end

--criterion
criterion=nn.CrossEntropyCriterion()

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
  print('shipped data function to cuda...')
  embedding_net_q = embedding_net_q:cuda()
  encoder_net_q = encoder_net_q:cuda()
  multimodal_net = multimodal_net:cuda()
  criterion = criterion:cuda()
  dummy_state_q = dummy_state_q:cuda()
  dummy_output_q = dummy_output_q:cuda()
end

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters()
embedding_w_q:uniform(-0.08, 0.08);

encoder_w_q,encoder_dw_q=encoder_net_q:getParameters()
encoder_w_q:uniform(-0.08, 0.08);

multimodal_w,multimodal_dw=multimodal_net:getParameters()
multimodal_w:uniform(-0.08, 0.08);

sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1)}


-- optimization parameter
local optimize={}
optimize.maxIter=opt.max_iters
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1
optimize.weightDecay = opt.weightdecay
optimize.winit=join_vector({encoder_w_q,embedding_w_q,multimodal_w})


------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()
  local qinds=torch.LongTensor(batch_size):fill(0)
  local iminds=torch.LongTensor(batch_size):fill(0)

  local nqs=dataset['question']:size(1)
  -- we use the last val_num data for validation (the data already randomlized when created)

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
local encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q)

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

  --clear gradients--
  for i=1,buffer_size_q do
    encoder_net_buffer_q[3][i]:zero()
  end
  embedding_dw_q:zero()
  multimodal_dw:zero()
  if opt.netmodel == 'RegSpa' then
    fusion_net:forget()
  end

  --grab a batch--
  local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch()
  local question_max_length=fv_sorted_q[2]:size(1)

  --embedding forward--
  local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2])

  --encoder forward--
  local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2])

  --multimodal/criterion forward--
  local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4])
  local scores=multimodal_net:forward({tv_q,fv_im})
  local f=criterion:forward(scores,labels)
  --multimodal/criterion backward--
  local dscores=criterion:backward(scores,labels)

  local tmp=multimodal_net:backward({tv_q,fv_im},dscores)
  local dtv_q=tmp[1]:index(1,fv_sorted_q[3])

  --encoder backward
  local junk4,dword_embedding_q=rnn_backward(encoder_net_buffer_q,dtv_q,dummy_output_q,states_q,word_embedding_q,fv_sorted_q[2])

  --embedding backward--
  dword_embedding_q=join_vector(dword_embedding_q)
  embedding_net_q:backward(fv_sorted_q[1],dword_embedding_q)

  --summarize f and gradient
  local encoder_adw_q=encoder_dw_q:clone():zero()
  for i=1,question_max_length do
    encoder_adw_q=encoder_adw_q+encoder_net_buffer_q[3][i]
  end

  gradients=join_vector({encoder_adw_q,embedding_dw_q,multimodal_dw})
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

local state={}
paths.mkdir(model_path..'save')
for iter = 1, opt.max_iters do
  if iter%opt.save_checkpoint_every == 0 then
    torch.save(string.format(model_path..'save/'..CP_name,iter),
      {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w})
  end
  if iter%100 == 0 then
    print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
  end
  if optim_method == 'rmsprop' then
    optim.rmsprop(JdJ, optimize.winit, optimize, state)
  elseif optim_method == 'adadelta' then
    optim.adadelta(JdJ, optimize.winit, optimize, state)
  end

  optimize.learningRate=optimize.learningRate*decay_factor
  if opt.learning_rate_decay_start>0 and iter>opt.learning_rate_decay_start and iter%opt.learning_rate_decay_every==0 then
    optimize.learningRate = optimize.learningRate*0.5
  end
  if iter%50 == 0 then -- change this to smaller value if out of the memory
    collectgarbage()
  end
end

-- Saving the final model
torch.save(string.format(model_path..final_model_name),
  {encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w})
