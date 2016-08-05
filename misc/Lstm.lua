----------------------------------------------------------------------------
-- Copy some functions from `misc.RNNUtils`.
-- `misc.RNNUtils` could not import directly here, because it depends on 
-- the argument `opt` in train.lua .
---------------------------------------------------------------------------

--duping an RNN block into multiple ones
function dupe_rnn(net,times)
    local w,dw=net:getParameters();
    local net_arr={};
    local net_dwarr={};
    local net_warr={};
    for i=1,times do
        local tmp=net:clone();
        local tmp1,tmp2=tmp:getParameters();
        --tmp1:set(w); --new test
        table.insert(net_arr,tmp);
        table.insert(net_warr,tmp1);
        table.insert(net_dwarr,tmp2);
    end
    collectgarbage();
    return {net_arr,net_warr,net_dwarr};
end


--rnn forward, tries to handle most cases
function rnn_forward(net_buffer,init_state,inputs,sizes)
    local N=sizes:size(1);
    local states={init_state[{{1,sizes[1]},{}}]};
    local outputs={};
    for i=1,N do
        local tmp;
        if i==1 or sizes[i]==sizes[i-1] then
            tmp=net_buffer[1][i]:forward({states[i],inputs[i]});
        elseif sizes[i]>sizes[i-1] then
            --right align
            local padding=init_state[{{1,sizes[i]},{}}];
            padding[{{1,sizes[i-1]},{}}]=states[i];
            states[i]=padding;
            tmp=net_buffer[1][i]:forward({padding,inputs[i]});
        elseif sizes[i]<sizes[i-1] then
            --left align
            tmp=net_buffer[1][i]:forward({states[i][{{1,sizes[i]}}],inputs[i]});
        end
        table.insert(states,tmp[1]);
        table.insert(outputs,tmp[2]);
    end
    return states,outputs;
end


--rnn backward
function rnn_backward(net_buffer,dend_state,doutputs,states,inputs,sizes)
    if type(doutputs)=="table" then
        local N=sizes:size(1);
        local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
        local dinput_embedding={};
        for i=N,1,-1 do
            local tmp;
            if i==1 or sizes[i]==sizes[i-1] then
                tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],doutputs[i]});
                dstate[i]=tmp[1];
            elseif sizes[i]>sizes[i-1] then
                --right align
                tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],doutputs[i]});
                dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
            elseif sizes[i]<sizes[i-1] then
                --left align
                --compute a larger dstate that matches i-1
                tmp=net_buffer[1][i]:backward({states[i][{{1,sizes[i]}}],inputs[i]},{dstate[i+1],doutputs[i]});
                local padding=dend_state[{{1,sizes[i-1]},{}}];
                padding[{{1,sizes[i]},{}}]=tmp[1];
                dstate[i]=padding;
            end
            dinput_embedding[i]=tmp[2];
        end
        return dstate,dinput_embedding;
    else
        local N=sizes:size(1);
        local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
        local dinput_embedding={};
        for i=N,1,-1 do
            local tmp;
            if i==1 or sizes[i]==sizes[i-1] then
                tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
                dstate[i]=tmp[1];
            elseif sizes[i]>sizes[i-1] then
                --right align
                tmp=net_buffer[1][i]:backward({states[i],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
                dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
            elseif sizes[i]<sizes[i-1] then
                --left align
                --compute a larger dstate that matches i-1
                tmp=net_buffer[1][i]:backward({states[i][{{1,sizes[i]}}],inputs[i]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
                local padding=dend_state[{{1,sizes[i-1]},{}}];
                padding[{{1,sizes[i]},{}}]=tmp[1];
                dstate[i]=padding;
            end
            dinput_embedding[i]=tmp[2];
        end
        return dstate,dinput_embedding;
    end 
end


-------------------------------------------------------------------------------------
-- A wrapper class for LSTM.
------------------------------------------------------------------------------------
LSTM = require 'misc.LSTM'


local Lstm, parent = torch.class('nn.Lstm', 'nn.Module')

function Lstm:__init(input_size, rnn_size, output_size, nlayer, nstep, dropout, gpuid)
    local master_node = LSTM.lstm_conventional(
            input_size, rnn_size, output_size, nlayer, dropout)
    if gpuid >= 0 then
        master_node = master_node:cuda()
    end
    master_node:getParameters():uniform(-0.08, 0.08)
    local step_nodes = dupe_rnn(master_node, nstep)
    local weight, gradWeight = master_node:getParameters()

    self.weight = weight
    self.gradWeight = gradWeight
    self.step_nodes = step_nodes
    self.nstep = nstep
end


function Lstm:updateParameters(param)
    if self.weight ~= param then
        self.weight:copy(param)
        local nodes = self.step_nodes
        for i=1,self.nstep do
            nodes[2][i]:copy(param)
        end
    end
end


function Lstm:zeroGradParameters()
    self.gradWeight:zero()
    local nodes = self.step_nodes
    for i=1,self.nstep do
        nodes[3][i]:zero()
    end
end


function Lstm:forward(init_state, inputs, batch_per_step)
    -- states, outputs
    return rnn_forward(self.step_nodes, init_state, inputs, batch_per_step)
end


function Lstm:backward(dend_state, doutputs, states, inputs, batch_per_step)
    local dstate, dinput = rnn_backward(self.step_nodes, dend_state, doutputs, 
                                        states, inputs, batch_per_step)
    local gradWeight = self.gradWeight:clone():zero()
    local nodes = self.step_nodes
    for i=1,batch_per_step:size(1) do
        gradWeight = gradWeight + nodes[3][i]
    end
    self.gradWeight:copy(gradWeight)
    return dstate, dinput
end
