require 'nn'
require 'misc.Lstm'

local Bilstm, parent = torch.class('nn.Bilstm', 'nn.Module')

function Bilstm:__init(input_size, rnn_size, output_size, nlayer, nstep, dropout, gpuid)
    self.net = nn.Lstm(input_size, rnn_size, output_size, nlayer, nstep, dropout, gpuid)
    self.r_net = nn.Lstm(input_size, rnn_size, output_size, nlayer, nstep, dropout, gpuid)
    local jtnn = nn.JoinTable(1, 1)
    if gpuid >= 0 then
        jtnn = jtnn:cuda()
    end
    self.jtnn = jtnn
    self.output_size = output_size
end


function Bilstm:parameters()
    local weight, gradWeight = self.net:parameters()
    local r_weight, r_gradWeight = self.r_net:parameters()
    return {weight[1], r_weight[1]}, {gradWeight[1], r_gradWeight[1]}
end


function Bilstm:updateParameters(param)
    local size = param:size(1)
    self.net:updateParameters(param[{{1,size/2}}])
    self.r_net:updateParameters(param[{{size/2+1, size}}])
end


function Bilstm:zeroGradParameters()
    self.net:zeroGradParameters()
    self.r_net:zeroGradParameters()
end


local function reverse(inputs, batch_per_step)
    local r_inputs = {}
    local r_batch_per_step = batch_per_step:clone()
    local L = #inputs
    for i=1,L do
        r_inputs[L-i+1] = inputs[i]
        r_batch_per_step[L-i+1] = batch_per_step[i]
    end
    return r_inputs, r_batch_per_step
end


function Bilstm:forward(init_state, inputs, batch_per_step)
    local r_inputs, r_batch_per_step = reverse(inputs, batch_per_step)
    local f_states, f_outputs = self.net:forward(init_state[1], inputs, batch_per_step)
    local r_states, r_outputs = self.r_net:forward(init_state[2], r_inputs, r_batch_per_step)

    local outputs = {}
    local jtnn = self.jtnn
    local L = #f_outputs
    for i=1,L do
        outputs[i] = jtnn:forward({f_outputs[i], r_outputs[L-i+1]})
    end
    return {f_states, r_states}, outputs
end


function Bilstm:backward(dend_state, doutputs, states, inputs, batch_per_step)
    local f_doutputs = {}
    local r_doutputs = {}
    local size = self.output_size
    if type(doutputs) == "table" then
        local L = #doutputs
        for i=1,L do
            f_doutputs[i] = doutputs[i][{{}, {1,size}}]
            r_doutputs[L-i+1] = doutputs[i][{{}, {size+1, 2*size}}]
        end
    else
        f_doutputs = doutputs[{{}, {1,size}}]
        r_doutputs = doutputs[{{}, {size+1, 2*size}}]
    end
    local r_inputs, r_batch_per_step = reverse(inputs, batch_per_step)
    local f_dstate, f_dinput = self.net:backward(dend_state[1], f_doutputs, states[1], inputs, batch_per_step)
    local r_dstate, r_dinput = self.r_net:backward(dend_state[2], r_doutputs, states[2], r_inputs, r_batch_per_step)
    local dinput = {}
    local L = #f_dinput
    for i=1,L do
        dinput[i] = f_dinput[i] + r_dinput[L-i+1]
    end
    return {f_dstate, r_dstate}, dinput
end
