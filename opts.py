import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='/home/bumsoo/Data/Planetoid', help='path')
        self.parser.add_argument('--dataset', type=str, default='pubmed', help='[cora | citeseer | pubmed]')
        self.parser.add_argument('--num_hidden', type=int, default=8, help='number of features')
        self.parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        self.parser.add_argument('--init_type', type=str, default='uniform', help='[uniform | xavier]')
        self.parser.add_argument('--model', type=str, default='basic', help='[basic | drop_in | attention | res_attention]')

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        args = vars(self.opt)

        return self.opt

class TrainOptions(BaseOptions):
    # Override
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='[sgd | adam]')
        self.parser.add_argument('--epoch', type=int, default=800, help='number of training epochs')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=5000, help='multiply by a gamma every set iter')
        self.parser.add_argument('--nb_heads', type=int, default=8, help='number of head attentions')
        self.parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
        self.isTrain = True

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False
