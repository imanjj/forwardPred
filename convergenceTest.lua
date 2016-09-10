#! /home/pourtaran/torch/install/bin/th
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})




-- XOR
--xt = torch.Tensor({{-1,-1},{1,1},{-1,1},{1,-1}})
--yt = torch.Tensor({'1','1','2','2'})

-- AND
xt = torch.Tensor({{-1,-1},{1,1},{-1,1},{1,-1}})
yt = torch.Tensor({'1','2','1','1'})

-- 2-class problem
noutputs = 2

-- number of hidden units (for MLP only):
ninputs = 2
nhiddens = 2 

-- Simple 1-layer neural network, with NormalforwardPred hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.forwardPred(ninputs,nhiddens))
   --model:add(nn.ReLU())
   model:add(nn.ReLU())
   model:add(nn.forwardPred(nhiddens,noutputs))
   model:add(nn.ReLU())
   model:add(nn.LogSoftMax())

-- The loss works like the MultiMarginCriterion: it takes
-- a vector of classes, and the index of the grountruth class
-- as arguments.


criterion = nn.ClassNLLCriterion()

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end


optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

outputs = torch.Tensor(4,1):fill(0)

function train()
   -- epoch tracker
   epoch = epoch or 1
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         -- reset gradients
         gradParameters:zero()
         -- evaluate function for complete mini batch
         outputs = model:forward(xt)
	 --print('woohoo!')
	 --print(outputs)
         --print(xt)
         local f = criterion:forward(outputs, yt)
         -- estimate df/dW
         local df_do = criterion:backward(outputs, yt)
         model:backward(xt, df_do)
         -- update confusion
         for i = 1,xt:size(1) do
            confusion:add(outputs[i], yt[i])
         end
         -- return f and df/dX
         return f,gradParameters
      end
   optimMethod(feval, parameters, optimState)
   --print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()
   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)
   -- next epoch
   epoch = epoch + 1
end

for i = 1,5 do
   -- train/test
   print('iteration: '.. i)

   train(trainData)  
end
