import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

        
class Architect(object):
  """ architect
  """
  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    # forward and calculate model loss
    loss = self.model._loss(input, target)

    # get all model parameters
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # calculate gradient and pass manual gradients to new model
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # unrolled model
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # forward and calculate unrolled loss
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    # compute gradient
    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()] # dalpha { L_val(w`, alpha)}
    vector = [v.grad.data for v in unrolled_model.parameters()] # dw` { L_val(w`, alpha) }
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    # create new model 
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    # calculate new weight dict(params)
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    # update new param and load updated model
    model_dict.update(params)
    model_new.load_state_dict(model_dict)

    # mask alpha only select two path
    new_alpha = []
    for alpha in self.model.arch_parameters():
      new_step = []
      for step in F.softmax(alpha, dim=-1).data:
        # select two path
        new_step.append(utils.binarize(step, 2))
      new_alpha.append(Variable(torch.stack(new_step), requires_grad=True))
    self.model._alphas_parameters = new_alpha

    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

