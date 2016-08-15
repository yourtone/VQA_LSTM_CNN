local Unzigzag, parent = torch.class('nn.Unzigzag', 'nn.Module')

function Unzigzag:__init(height, width)
    parent.__init(self)
    self.H = height
    self.W = width
end


function Unzigzag:_getSize(input)
    return unpack(torch.totable(input:size()))
end

function Unzigzag:updateOutput(input)
    assert(input:dim() == 3, string.format(
            'Unexpected input dimension: %d != 3, format NxCx(HxW)', input:dim()))
    local output = self.output
    local N, C, HW = self:_getSize(input)
    local H, W = self.H, self.W
    assert(HW == H*W)
    if not output then
        output = torch.Tensor(N, C, H, W)
    end
    output:resize(N, C, H, W)

    for i=1,H,2 do
        for j=1,W do
            output[{{},{},{i},{j}}] = input[{{},{},{(i-1)*W+j}}]
        end
    end
    for i=2,H,2 do
        for j=1,W do
            output[{{},{},{i},{W-j+1}}] = input[{{},{},{(i-1)*W+j}}]
        end
    end
    self.output = output
    return self.output
end


function Unzigzag:updateGradInput(input, gradOutput)
    assert(input:dim() == 3, string.format(
            'Unexpected input dimension: %d != 3, format NxCx(HxW)', input:dim()))
    assert(gradOutput:dim() == 4, string.format(
            'Unexpected gradOutput dimension: %d != 4, format NxCxHxW', gradOutput:dim()))
            
    local N, C, HW = self:_getSize(input)
    local H, W = self.H, self.W
    local gradInput = self.gradInput
    if not gradInput then
        gradInput = torch.Tensor(N, C, HW)
    end
    gradInput:resize(N, C, HW)

    for i=1,H,2 do
        for j=1,W do
            gradInput[{{},{},{(i-1)*W+j}}] = gradOutput[{{},{},{i},{j}}]
        end
    end
    for i=2,H,2 do
        for j=1,W do
            gradInput[{{},{},{(i-1)*W+j}}] = gradOutput[{{},{},{i},{W-j+1}}]
        end
    end
    self.gradInput = gradInput
    return self.gradInput
end
    

function Unzigzag:__tostring__()
    return torch.type(self) .. '()'
end
