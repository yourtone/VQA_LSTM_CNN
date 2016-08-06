local Zigzag, parent = torch.class('nn.Zigzag', 'nn.Module')

function Zigzag:__init()
    parent.__init(self)
end


function Zigzag:_getSize(input)
    return unpack(torch.totable(input:size()))
end


function Zigzag:updateOutput(input)
    assert(input:dim() == 4, string.format(
            'Unexpected input dimension: %d != 4, format NxCxHxW', input:dim()))

    local N, C, H, W = self:_getSize(input)
    local output = self.output
    if not output then
        output = torch.Tensor(N, C, H*W)
    end
    output:resize(N, C, H*W)

    for i=1,H,2 do
        for j=1,W do
            output[{{},{},{(i-1)*W+j}}] = input[{{},{},{i},{j}}]
        end
    end
    for i=2,H,2 do
        for j=1,W do
            output[{{},{},{(i-1)*W+j}}] = input[{{},{},{i},{W-j+1}}]
        end
    end
    self.output = output
    return self.output
end


function Zigzag:updateGradInput(input, gradOutput)
    assert(input:dim() == 4, string.format(
            'Unexpected input dimension: %d != 4, format NxCxHxW', input:dim()))
    assert(gradOutput:dim() == 3, string.format(
            'Unexpected gradOutput dimension: %d != 3, format NxCx(H*W)', gradOutput:dim()))

    local N, C, H, W = self:_getSize(input)
    local gradInput = self.gradInput
    if not gradInput then
        gradInput = torch.Tensor(N, C, H, W)
    end
    gradInput:resize(N, C, H, W)

    for i=1,H,2 do
        for j=1,W do
            gradInput[{{},{},{i},{j}}] = gradOutput[{{},{},{(i-1)*W+j}}]
        end
    end
    for i=2,H,2 do
        for j=1,W do
            gradInput[{{},{},{i},{W-j+1}}] = gradOutput[{{},{},{(i-1)*W+j}}]
        end
    end
    self.gradInput = gradInput
    return self.gradInput
end


function Zigzag:__tostring__()
    return torch.type(self) .. '()'
end
