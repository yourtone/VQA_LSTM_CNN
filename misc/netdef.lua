require 'nn'
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
    qc = nn.Reshape(nhcommon, hB, wB)(nn.Replicate(hB*wB, 3, 1)(qc))
    local output = nn.CMulTable()({qc, ic})
    return nn.gModule({q, i}, {output})
end

return netdef;
