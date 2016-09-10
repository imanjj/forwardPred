local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self:reset()
end

function Linear:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function Linear:updateOutput(input)

   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      updateAddBuffer(self, input)
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   --print('updateGradInput started...')
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
	 -- self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
	 self.gradInput:addmv(0, -1, self.weight:t(), self.output)
	 self.gradInput:addcmul(1, torch.norm(self.weight,2,1):t(), input)
      elseif input:dim() == 2 then
	 --print('self.weight:')
	 --print(self.weight)
	 print('self.bias:')
	 print(self.bias)
	 --print('self.output:')
         --print(self.output)
	 --print('torch.cmul(self.output, torch.ge(self.output, 0):double()):')
	 --print(torch.cmul(self.output, torch.ge(self.output, 0):double()))
         -- self.gradInput:addmm(0, 1, gradOutput, self.weight)
	 --self.gradInput:addmm(0, -1, torch.cmul(self.output, torch.ge(self.output, 0):double()), torch.pow(self.weight,2))
         self.gradInput:addmm(0, -1, torch.tanh(self.output), torch.pow(self.weight,2))
         --print('self.gradInput after first computation:')
         --print(self.gradInput)
	 --print('torch.norm(self.weight,2,1):expandAs(input)')
 	 --print(torch.norm(self.weight,2,1):expandAs(input))
	 --print('input')
	 --print(input)
	 self.gradInput:addcmul(1, torch.norm(self.weight,2,1):expandAs(input), input)
	 --print('self.gradInput after second computation:')
         --print(self.gradInput)
	 
      end

      return self.gradInput
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   --print('accGradParameters started...')
   scale = scale or 1
   --print('gradOutput:')
   --print(gradOutput)
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters

function Linear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
