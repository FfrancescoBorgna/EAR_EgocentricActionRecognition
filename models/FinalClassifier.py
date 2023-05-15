from torch import nn
from torch.autograd import Function
import torch

class Classifier(nn.Module):
    def __init__(self, num_class, n_features,temporal_type,ablation_mask):
        super().__init__()
        """
        n_features: [0]: 5
                    [1]: 1024
        tmeporal_type: TRN or pooling
        ablation_mask: Dict("gsd":
                             gtd
                             grd
                             )
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.num_class = num_class
        self.n_feat = n_features
        self.temporal_type = temporal_type
        self.ablation_mask = ablation_mask
        
        #GSF
        n_gsf_out = 512
        self.n_gsf_out = n_gsf_out
        self.gsf = nn.Sequential()
        self.gsf.add_module('gsf_fc1', nn.Linear(self.n_feat[1], n_gsf_out))
        self.gsf.add_module('gsf_bn1', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu1', nn.ReLU(True))
        self.gsf.add_module('gsf_drop1', nn.Dropout())
        self.gsf.add_module('gsf_fc2', nn.Linear(n_gsf_out, n_gsf_out))
        self.gsf.add_module('gsf_bn2', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu2', nn.ReLU(True))

        #Spatial Domain Discriminator
        if(ablation_mask["gsd"]):
            self.gsd = nn.Sequential()
            self.gsd.add_module('gsd_fc1', nn.Linear(n_gsf_out*n_features[0], 256))
            self.gsd.add_module('gsd_bn1', nn.BatchNorm1d(256))
            self.gsd.add_module('gsd_relu1', nn.ReLU(True))
            self.gsd.add_module('gsd_drop1', nn.Dropout())
            self.gsd.add_module('gsd_fc2', nn.Linear(256, 256))
            self.gsd.add_module('gsd_bn2', nn.BatchNorm1d(256))
            self.gsd.add_module('gsd_relu2', nn.ReLU(True))      
            self.gsd.add_module('gsd_fc3', nn.Linear(256, 2))
            self.gsd.add_module('gsd_softmax', nn.LogSoftmax(dim=1))
        
        #Temporal Pooling
        if(temporal_type == "TRN"):
            raise NotImplementedError
        
        #Temporal Domain discriminator
        if(ablation_mask["gtd"]):
            self.gtd = nn.Sequential()
            self.gtd.add_module('gtd_fc1', nn.Linear(n_gsf_out, 512))
            self.gtd.add_module('gtd_bn1', nn.BatchNorm1d(512))
            self.gtd.add_module('gtd_relu1', nn.ReLU(True))
            self.gtd.add_module('gtd_drop1', nn.Dropout())
            self.gtd.add_module('gtd_fc2', nn.Linear(512, 256))
            self.gtd.add_module('gtd_bn2', nn.BatchNorm1d(256))
            self.gtd.add_module('gtd_relu2', nn.ReLU(True))      
            self.gtd.add_module('gtd_fc3', nn.Linear(256, 2))
            self.gtd.add_module('gtd_softmax', nn.LogSoftmax(dim=1))
        
        #Gy
        self.gy = nn.Sequential()
        self.gy.add_module('c_fc1', nn.Linear(n_gsf_out, num_class))
        self.gy.add_module('c_softmax', nn.LogSoftmax(dim=1))


    def forward(self, x,alpha = 1):
        
        spatial_domain_out = None
        temporal_domain_out = None
        class_out = None

        x = self.gsf(x)
        #spatial domain out
        if(self.ablation_mask["gsd"]):
            reverse_features = ReverseLayerF.apply(x,alpha)
            spatial_domain_out = self.gsd(reverse_features.view(-1,5*self.n_gsf_out))
        #temporal aggregation 
        if(self.temporal_type == "TRN"):
            raise NotImplementedError
        else:
            temporal_aggregation = torch.mean(x,1)
        #temporal domain
        if(self.ablation_mask["gtd"]):
            temporal_domain_out =  self.gtd(ReverseLayerF.apply(temporal_aggregation,alpha))
        
        class_out = self.gy(temporal_aggregation)

        return spatial_domain_out,temporal_domain_out, class_out
      
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
