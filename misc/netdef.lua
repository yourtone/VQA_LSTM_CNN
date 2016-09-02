require 'nn'
require 'rnn'
require 'nngraph'

netdef={};

function netdef.AxB(nhA,nhB,nhcommon,dropout)
	dropout = dropout or 0
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
	local output=nn.CMulTable()({qc,ic});
	return nn.gModule({q,i},{output});
end

function netdef.A_B(nhA,nhB,nhcommon,dropout)
    dropout = dropout or 0
    local q=nn.Identity()();
    local i=nn.Identity()();
    local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
    local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
    local output=nn.JoinTable(2)({qc,ic});
    return nn.gModule({q,i},{output});
end

-- add by Yuetan Lin
function netdef.AxBB(nhA,nhB,NB,nhcommon,dropout)
    dropout = dropout or 0
    local q=nn.Identity()();
    local i=nn.Identity()();
    local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q))); -- nhA -> nhcommon
    local ic=nn.Tanh()(nn.TemporalConvolution(nhB,nhcommon,1)(nn.Dropout(dropout)(i))); -- (nbatch*)NB*nhB -> (nbatch*)NB*nhcommon
    qc=nn.Reshape(-1,nhcommon,false)(qc) -- for batch or non-batch
    qc=nn.Replicate(NB,2)(qc);-- nhcommon -> NB*nhcommon
    qc=nn.Squeeze(1)(qc)
    local output=nn.Tanh()(nn.CMulTable()({qc,ic}));
    output=nn.Reshape(1,NB,nhcommon)(output)
    output=nn.SpatialMaxPooling(1,NB)(output)
    output=nn.Squeeze()(output)
    return nn.gModule({q,i},{output});
end
function netdef.QxII(nhA,nhB,NB,nhcommon,dropout)
    dropout = dropout or 0
    local q=nn.Identity()();
    local i=nn.Identity()();
    local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q))); -- nhA -> nhcommon
    local ic=nn.Tanh()(nn.TemporalConvolution(nhB,nhcommon,1)(nn.Dropout(dropout)(i))); -- (nbatch*)NB*nhB -> (nbatch*)NB*nhcommon
    qc=nn.Reshape(-1,nhcommon,false)(qc) -- for batch or non-batch
    qc=nn.Replicate(NB,2)(qc);-- nhcommon -> NB*nhcommon
    qc=nn.Squeeze(1)(qc)
    local output=nn.CMulTable()({qc,ic});
    return nn.gModule({q,i},{output});
end


function netdef.Qx2DII(nhA, nhB, hB, wB, nhcommon, dropout)
    dropout = dropout or 0
    local q = nn.Identity()();
    local i = nn.Identity()();
    local qc = nn.Tanh()(nn.Linear(nhA, nhcommon)(nn.Dropout(dropout)(q)));
    local ic = nn.Tanh()(nn.SpatialConvolution(nhB, nhcommon, 1, 1)(nn.Dropout(dropout)(i)));
    qc = nn.Reshape(nhcommon, hB, wB)(nn.Replicate(hB*wB, 2, 1)(qc))
    local output = nn.CMulTable()({qc, ic})
    return nn.gModule({q, i}, {output})
end

local function flat_image_feature(nhimage, gridH, gridW, zigzag)
    local flat_feature = nn.Sequential()
    if zigzag then
        flat_feature:add(nn.Zigzag())
    else
        flat_feature:add(nn.Reshape(nhimage, gridH*gridW))
    end
    flat_feature:add(nn.Transpose({2, 3}))
    return flat_feature
end


function netdef.salient_weight(nhimage, gridH, gridW, zigzag)
    -- input image feature: bs x nhimage x gridH x gridW, output shape is the same as input.
    local get_weight = nn.Sequential()
    get_weight:add(flat_image_feature(nhimage, gridH, gridW, zigzag))
              :add(nn.SeqBRNN(nhimage, 1, true))
              --:add(nn.SeqBRNN(nhimage, 16, true))
	      --:add(nn.Transpose({1, 2}))
	      --:add(nn.Sequencer(nn.Linear(16, 1)))
	      --:add(nn.Transpose({1, 2}))
              :add(nn.Squeeze(2, 2))
              :add(nn.SoftMax())
    return get_weight
end


function netdef.attend(nhimage, gridH, gridW, zigzag) 
    -- input: {weights(NxC), image_feature(NxCxHxW))}
    local attend = nn.Sequential()
    attend:add(nn.ParallelTable()
                  :add(nn.Replicate(nhimage, 2, 1))
                  :add(flat_image_feature(nhimage, gridH, gridW, zigzag)))
          :add(nn.CMulTable())
          :add(nn.Transpose({2, 3}))
    if zigzag then
        attend:add(nn.Unzigzag(gridH, gridW))
    else
        attend:add(nn.Reshape(nhimage, gridH, gridW))
    end

    return attend
end

return netdef;
