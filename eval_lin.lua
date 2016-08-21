-- Modified by Yuetan Lin (2016/06/19 08:38)
require 'nn';
require 'cutorch';
require 'cunn';
require 'nngraph';
require 'optim';
require 'misc.netdef';
require 'hdf5';
LSTM=require 'misc.LSTM';
cjson=require('cjson');
require 'misc.Zigzag';
require 'misc.Unzigzag';
require 'rnn';

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the Visual Question Answering model')
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
cmd:option('-netmodel', 'RegMax', 'holistic|RegMax|RegSpa|SalMax|SalSpa|RegMaxQ|RegSpaQ|SalMaxQ|SalSpaQ|QSalMax|ConvMax')
-- holistic: baseline (holistic image feature .* question feature)
-- Reg/Sal: region / saliency Bi-LSTM
-- Pool/Spa: max pooling / spatial Bi-LSTM
-- Q: concatenate question

cmd:option('-out_path', 'result/', 'path to save output json file')

--check point
cmd:option('-dofinal', false, 'do evaluation on final model')
cmd:option('-doiter', false, 'do evaluation per iteration')
cmd:option('-dotrain', false, 'do evaluation on training set')
cmd:option('-eval_start_iter', 1, 'evaluation start iteration')
cmd:option('-max_iters', 50000, 'max number of iterations to run for')
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')
cmd:option('-zigzag', false, 'use Zigzag to arrange image patch')

cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

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

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=opt.imdim
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
require 'misc.autoNaming'

print('DataLoader loading h5 file: ', input_json)
local file = io.open(input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

if opt.dotrain then
  choice = 'train'
else
  choice = 'test'
end
-- train/test ques
print('DataLoader loading h5 file: ', input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(input_ques_h5, 'r')
dataset['question'] = h5_file:read('/ques_'..choice):all()
dataset['lengths_q'] = h5_file:read('/ques_length_'..choice):all()
dataset['img_list'] = h5_file:read('/img_pos_'..choice):all()
dataset['ques_id'] = h5_file:read('/question_id_'..choice):all()
dataset['MC_ans_'..choice] = h5_file:read('/MC_ans_'..choice):all()
h5_file:close()

-- train/test image
print('DataLoader loading h5 file: ', input_img_h5)
local h5_file = hdf5.open(input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_'..choice):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])
collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

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
    :add(netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5))
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
            :add(netdef.salient_weight(nhimage, grid_height, grid_width, opt.zigzag))
            :add(nn.Identity()))
          :add(netdef.attend(nhimage, grid_height, grid_width, opt.zigzag))))
      :add(netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5))
      :add(nn.Tanh())
      :add(nn.SpatialMaxPooling(grid_width, grid_height))
      :add(nn.Squeeze())
      :add(nn.Dropout(0.5))
      :add(nn.Linear(common_embedding_size, noutput))
elseif opt.netmodel == 'SalMaxQ' then
    q = nn.Identity()()
    i = nn.Identity()()
    salient_i = netdef.attend(nhimage, grid_height, grid_width, opt.zigzag)(
        {netdef.salient_weight(nhimage, grid_height, grid_width, opt.zigzag)(i), i})
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
    q_i = nn.JoinTable(1, 3)({nn.Reshape(nhquestion, grid_height, grid_width)(nn.Replicate(grid_height*grid_width, 2, 1)(q)), i}) 
    salient_i = netdef.attend(nhimage, grid_height, grid_width, opt.zigzag)(
        {netdef.salient_weight(nhimage+nhquestion, grid_height, grid_width, opt.zigzag)(q_i), i})
    mul_fea = netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5)({q, salient_i})
    fusion_fea = nn.Dropout(0.5)(
      nn.Squeeze()(
        nn.SpatialMaxPooling(grid_width, grid_height)(
          nn.Tanh()(mul_fea))))
    scores = nn.Linear(common_embedding_size, noutput)(fusion_fea)
    multimodal_net = nn.gModule({q, i}, {scores})
elseif opt.netmodel == 'SalSpa' then
    q = nn.Identity()()
    i = nn.Identity()()
    salient_i = netdef.attend(nhimage, grid_height, grid_width, opt.zigzag)(
        {netdef.salient_weight(nhimage, grid_height, grid_width, opt.zigzag)(i), i})
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
elseif opt.netmodel == 'ConvMax' then
    q = nn.Identity()()
    i = nn.Identity()()
    salient_weight = nn.Reshape(grid_height*grid_width)(
                     nn.SpatialConvolution(nhimage, 1, 1, 1)(i))
    salient_i = netdef.attend(nhimage, grid_height, grid_width, opt.zigzag)(
        {salient_weight, i})
    mul_fea = netdef.Qx2DII(nhquestion, nhimage, grid_height, grid_width, common_embedding_size, 0.5)({q, salient_i})
    fusion_fea = nn.Dropout(0.5)(
                 nn.Squeeze()(
                 nn.SpatialMaxPooling(grid_width, grid_height)(
                 nn.Tanh()(mul_fea))))
    scores = nn.Linear(common_embedding_size, noutput)(fusion_fea)
    multimodal_net = nn.gModule({q, i}, {scores})
else
  print('ERROR: netmodel is not defined: '..opt.netmodel)
end

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
  print('shipped data function to cuda...')
  embedding_net_q = embedding_net_q:cuda()
  encoder_net_q = encoder_net_q:cuda()
  multimodal_net = multimodal_net:cuda()
  dummy_state_q = dummy_state_q:cuda()
  dummy_output_q = dummy_output_q:cuda()
end

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_eval(s,e)
  local batch_size=e-s+1;
  local qinds=torch.LongTensor(batch_size):fill(0);
  local iminds=torch.LongTensor(batch_size):fill(0);
  for i=1,batch_size do
    qinds[i]=s+i-1;
    iminds[i]=dataset['img_list'][qinds[i]];
  end

  local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question']:index(1,qinds),dataset['lengths_q']:index(1,qinds),vocabulary_size_q);

  local fv_im=dataset['fv_im']:index(1,iminds);
  local qids=dataset['ques_id']:index(1,qinds);

  -- ship to gpu
  if opt.gpuid >= 0 then
    fv_sorted_q[1]=fv_sorted_q[1]:cuda()
    fv_sorted_q[3]=fv_sorted_q[3]:cuda()
    fv_sorted_q[4]=fv_sorted_q[4]:cuda()
    fv_im = fv_im:cuda()
  end

  --print(string.format('batch_sort:%f',timer:time().real));
  return fv_sorted_q,fv_im:cuda(),qids,batch_size;
end

-- duplicate the RNN
local encoder_net_buffer_q
------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
function forward(s,e)
  local timer = torch.Timer();
  --grab a batch--
  local fv_sorted_q,fv_im,qids,batch_size=dataset:next_batch_eval(s,e);
  local question_max_length=fv_sorted_q[2]:size(1);

  --embedding forward--
  local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]);

  --encoder forward--
  local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]);

  --multimodal forward--
  local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]);
  local scores=multimodal_net:forward({tv_q,fv_im});
  if opt.netmodel == 'RegSpa' then
    fusion_net:forget()
  end
  return scores:double(),qids;
end

------------------------------------------------------------------------
-- Write to Json file
------------------------------------------------------------------------
function writeAll(file,data)
  local f = io.open(file, "w")
  f:write(data)
  f:close()
end

function saveJson(fname,t)
  return writeAll(fname,cjson.encode(t))
end

------------------------------------------------------------------------
-- Evaluation per Iteration
------------------------------------------------------------------------
buffer_size_q=dataset['question']:size()[2]
if opt.doiter then
  for iter = opt.eval_start_iter, opt.max_iters do
    if iter%opt.save_checkpoint_every == 0 then
      -- setting to evaluation
      embedding_net_q:evaluate();
      encoder_net_q:evaluate();
      multimodal_net:evaluate();
      if opt.netmodel == 'RegSpa' then
        cell:evaluate();
      end

      embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
      encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();
      multimodal_w,multimodal_dw=multimodal_net:getParameters();

      model_param=torch.load(string.format(opt.checkpoint_path..'save/'..CP_name,iter))
      embedding_w_q:copy(model_param['embedding_w_q']);
      encoder_w_q:copy(model_param['encoder_w_q']);
      multimodal_w:copy(model_param['multimodal_w']);
      print("Iteration: "..iter)

      -- duplicate the RNN
      encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q);

      -- Do Prediction
      nqs=dataset['question']:size(1);
      scores=torch.Tensor(nqs,noutput);
      qids=torch.LongTensor(nqs);
      for i=1,nqs,batch_size do
        xlua.progress(i, nqs)
        r=math.min(i+batch_size-1,nqs);
        scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
      end
      xlua.progress(nqs, nqs)
      tmp,pred=torch.max(scores,2);

      --response={};
      --for i=1,nqs do
      --  table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
      --end
      --saveJson(opt.out_path .. 'OpenEnded_iter' .. iter .. '_' .. result_name,response);
      --print('save results in: '..opt.out_path .. 'OpenEnded_iter' .. iter .. '_' .. result_name)

      mc_response={};
      for i=1,nqs do
        local mc_prob = {}
        local mc_idx = dataset['MC_ans_'..choice][i]
        local tmp_idx = {}
        for j=1, mc_idx:size()[1] do
          if mc_idx[j] ~= 0 then
            table.insert(mc_prob, scores[{i, mc_idx[j]}])
            table.insert(tmp_idx, mc_idx[j])
          end
        end
        local tmp,tmp2=torch.max(torch.Tensor(mc_prob), 1);
        table.insert(mc_response, {question_id=qids[i],answer=json_file['ix_to_ans'][tostring(tmp_idx[tmp2[1]])]})
      end
      saveJson(opt.out_path .. 'MultipleChoice_iter' .. iter .. '_' .. result_name, mc_response);
      print('save results in: '..opt.out_path .. 'MultipleChoice_iter' .. iter .. '_' .. result_name)
    end
  end
end


------------------------------------------------------------------------
-- Evaluation on Final Iteration
------------------------------------------------------------------------
if opt.dofinal then
  -- setting to evaluation
  embedding_net_q:evaluate();
  encoder_net_q:evaluate();
  multimodal_net:evaluate();
  if opt.netmodel == 'RegSpa' then
    cell:evaluate();
  end

  embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
  encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();
  multimodal_w,multimodal_dw=multimodal_net:getParameters();

  -- loading the model
  model_param=torch.load('model/'..final_model_name);
  embedding_w_q:copy(model_param['embedding_w_q']);
  encoder_w_q:copy(model_param['encoder_w_q']);
  multimodal_w:copy(model_param['multimodal_w']);

  -- duplicate the RNN
  encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q);
  -----------------------------------------------------------------------
  -- Do Prediction
  -----------------------------------------------------------------------
  nqs=dataset['question']:size(1);
  scores=torch.Tensor(nqs,noutput);
  qids=torch.LongTensor(nqs);
  for i=1,nqs,batch_size do
    xlua.progress(i, nqs)
    r=math.min(i+batch_size-1,nqs);
    scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
  end
  xlua.progress(nqs, nqs)
  tmp,pred=torch.max(scores,2);

  --response={};
  --for i=1,nqs do
  --  table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
  --end
  --paths.mkdir(opt.out_path)
  --saveJson(opt.out_path .. 'OpenEnded_' .. result_name,response);
  --print('save results in: '..opt.out_path .. 'OpenEnded_' .. result_name)

  mc_response={};
  for i=1,nqs do
    local mc_prob = {}
    local mc_idx = dataset['MC_ans_'..choice][i]
    local tmp_idx = {}
    for j=1, mc_idx:size()[1] do
      if mc_idx[j] ~= 0 then
        table.insert(mc_prob, scores[{i, mc_idx[j]}])
        table.insert(tmp_idx, mc_idx[j])
      end
    end
    local tmp,tmp2=torch.max(torch.Tensor(mc_prob), 1);
    table.insert(mc_response, {question_id=qids[i],answer=json_file['ix_to_ans'][tostring(tmp_idx[tmp2[1]])]})
  end
  saveJson(opt.out_path .. 'MultipleChoice_' .. result_name, mc_response);
  print('save results in: '..opt.out_path .. 'MultipleChoice_' .. result_name)
end
