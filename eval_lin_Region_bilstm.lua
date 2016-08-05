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
cjson=require('cjson');
LSTM=require 'misc.LSTM';
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_img_s1_VGG16_l30_snake196x512_o1k_sub.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_prepro_s1_wct0_o1k_sub.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro_s1_wct0_o1k_sub.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'model/lstm_s1_wct0_VGG16_d196x512_es200_rs512_rl2_cs256_bs50_o1k_sub.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'result/', 'path to save output json file')
cmd:option('-result_name', 'lstm_s1_wct0_VGG16_d196x512_es200_rs512_rl2_cs256_bs50_o1k_sub.json', 'output json file name')

-- Model parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-imdim',512,'image feature dimension')
cmd:option('-num_region',196,'number of image regions')
cmd:option('-batch_size',50,'batch_size for each iterations')
cmd:option('-max_iters', 50000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 256, 'size of the common embedding vector')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')
cmd:option('-CP_name', 'lstm_s1_wct0_VGG16_d196x512_es200_rs512_rl2_cs256_bs50_o1k_sub_iter%d.t7', 'checkpoints file names')
cmd:option('-final_model_name', 'lstm_s1_wct0_VGG16_d196x512_es200_rs512_rl2_cs256_bs50_o1k_sub.t7', 'checkpoints file names')

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

checkpoint_path = opt.checkpoint_path
batch_size = opt.batch_size
embedding_size_q = opt.input_encoding_size
lstm_size_q = opt.rnn_size
nlstm_layers_q = opt.rnn_layer
nhimage = opt.imdim
common_embedding_size = opt.common_embedding_size
noutput = opt.num_output
dummy_output_size = 1

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
print('DataLoader loading h5 file: ', opt.input_json)
file = io.open(opt.input_json, 'r')
text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
dataset = {}
h5_file = hdf5.open(opt.input_ques_h5, 'r')
dataset['question'] = h5_file:read('/ques_test'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
dataset['img_list'] = h5_file:read('/img_pos_test'):all()
dataset['ques_id'] = h5_file:read('/question_id_test'):all()
dataset['MC_ans_test'] = h5_file:read('/MC_ans_test'):all()
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
h5_file = hdf5.open(opt.input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_test'):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Normalize the image feature
if opt.img_norm == 1 then
  local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),3))
  nm[nm:eq(0)]=1e-5
  dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,1,nhimage)):float()
end
assert(torch.sum(dataset['fv_im']:ne(dataset['fv_im']))==0)

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
multimodal_net=nn.Sequential()
        :add(netdef.QxII(2*lstm_size_q*nlstm_layers_q,nhimage,opt.num_region,common_embedding_size,0.5))
        :add(nn.Tanh())
        :add(nn.SplitTable(1, 2))

-- correlate the multimodel features by bidirection lstm.
fusion_net = nn.Bilstm(common_embedding_size, lstm_size_q, common_embedding_size/2, 
                       1, opt.num_region, 0.5, opt.gpuid)

-- answer generation
add_new_index = nn.ParallelTable()
for i=1,opt.num_region do
    add_new_index:add(nn.Reshape(1, common_embedding_size))
end
answer_net=nn.Sequential()
        :add(add_new_index)
        :add(nn.JoinTable(1, 2))
        :add(nn.Reshape(1,opt.num_region,common_embedding_size))
        :add(nn.SpatialMaxPooling(1,opt.num_region))
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
batch_per_step_f = torch.LongTensor(opt.num_region):fill(batch_size)

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

embedding_net_q:evaluate();
encoder_net_q:evaluate();
multimodal_net:evaluate();
fusion_net:evaluate();
answer_net:evaluate();

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters()
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters()
multimodal_w,multimodal_dw=multimodal_net:getParameters()
fusion_w, fusion_dw = fusion_net:getParameters()
answer_w,answer_dw=answer_net:getParameters()

model_param=torch.load(model_path);
embedding_w_q:copy(model_param['embedding_w_q']);
encoder_w_q:copy(model_param['encoder_w_q']);
multimodal_w:copy(model_param['multimodal_w']);
fusion_w:copy(model_param['fusion_w']);
answer_w:copy(model_param['answer_w']);

sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1),
       fusion_w:size(1),answer_w:size(1)}

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch_test(s,e)
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

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- duplicate the RNN
encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q)
function forward(s,e)
  local timer = torch.Timer();
  --grab a batch--
  local fv_sorted_q,fv_im,qids,batch_size=dataset:next_batch_test(s,e);
  local question_max_length=fv_sorted_q[2]:size(1);

  --embedding forward--
  local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]);

  --encoder forward--
  local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]);

  --multimodal/criterion forward--
  local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]);
  local scores=multimodal_net:forward({tv_q,fv_im});
  return scores:double(),qids;
end


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

response={};
for i=1,nqs do
  table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
end

paths.mkdir(opt.out_path)
saveJson(opt.out_path .. 'OpenEnded_' .. opt.result_name,response);
print('save results in: '..opt.out_path .. 'OpenEnded_' .. opt.result_name)

mc_response={};

for i=1,nqs do
  local mc_prob = {}
  local mc_idx = dataset['MC_ans_test'][i]
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

saveJson(opt.out_path .. 'MultipleChoice_' .. opt.result_name, mc_response);
print('save results in: '..opt.out_path .. 'MultipleChoice_' .. opt.result_name)
