from torch import nn
import torch
from torch.autograd import Function
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_class, n_features):
        super().__init__()
        """
        n_features: [0]: 5
                    [1]: 1024
        tmeporal_type: TRN or pooling
                          
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.num_class = num_class
        self.n_feat = n_features

        self.l1 = nn.Linear(self.n_feat[1], 4)
        self.l1.weight.data.fill_(1)
        self.l1.bias.data.fill_(0)
        print("")



    def forward(self, x):
        return self.l1(x)


def main():
    v1 = torch.ones([1,6])
    v2 = torch.tensor([1.0,2.0,3.0,4.0,5.0])
    v2 = v2.reshape(-1,1)
    vec = torch.matmul(v2,v1)
    print("Vec is: ",vec)
    c = Classifier(5,[5,6])
    out = c.forward(vec)
    print("Output is of shape: ", np.shape(out),"and is: \n",out)


if __name__ == '__main__':
    main()
