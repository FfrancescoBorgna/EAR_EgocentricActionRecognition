from torch import nn
from torch.autograd import Function
import torch
from modules.TRNmodule import RelationModuleMultiScale
from scipy.stats import entropy 

class Classifier(nn.Module):
    def __init__(self, num_class, n_features,temporal_type,ablation_mask,device):
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
        self.batch_size = 32 #TODO *************
        self.device = device
        #GSF
        n_gsf_out = 512
        self.n_gsf_out = n_gsf_out
        self.gsf = nn.Sequential()
        self.gsf.add_module('gsf_fc1', nn.Linear(self.n_feat[1], n_gsf_out))
        self.gsf.add_module('gsf_bn1', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu1', nn.LeakyReLU(0.1))
        self.gsf.add_module('gsf_drop1', nn.Dropout())
        self.gsf.add_module('gsf_fc2', nn.Linear(n_gsf_out, n_gsf_out))
        self.gsf.add_module('gsf_bn2', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu2', nn.LeakyReLU(0.1))

        #Spatial Domain Discriminator
        if(ablation_mask["gsd"]):
            n_gsd = 256;
            self.gsd = nn.Sequential()
            self.gsd.add_module('gsd_fc1', nn.Linear(n_gsf_out*n_features[0], n_gsd))
            self.gsd.add_module('gsd_bn1', nn.BatchNorm1d(n_gsd))
            self.gsd.add_module('gsd_relu1', nn.LeakyReLU(0.1))
            self.gsd.add_module('gsd_drop1', nn.Dropout())
            self.gsd.add_module('gsd_fc2', nn.Linear(n_gsd, n_gsd//2))
            self.gsd.add_module('gsd_bn2', nn.BatchNorm1d( n_gsd//2))
            self.gsd.add_module('gsd_relu2', nn.LeakyReLU(0.1))      
            self.gsd.add_module('gsd_fc3', nn.Linear( n_gsd//2, 2))
            #self.gsd.add_module('gsd_softmax', nn.Softmax(dim=1))
        
        #Temporal Pooling
        if(temporal_type == "TRN"):
            self.trn = nn.Sequential()
            self.trn.add_module('trn', RelationModuleMultiScale(img_feature_dim=n_gsf_out, num_bottleneck=n_gsf_out, num_frames=n_features[0]))
            n_grd_out = 256
            if(ablation_mask["grd"]):
                self.grd_all = nn.ModuleList()
                for i in range(self.n_feat[0]-1):
                    grd = nn.Sequential(
                        nn.Linear(n_gsf_out,n_grd_out),
                        nn.BatchNorm1d(n_grd_out),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(n_grd_out, n_grd_out//2),
                        nn.BatchNorm1d(n_grd_out//2),
                        nn.ReLU(True) ,
                        nn.Linear(n_grd_out//2, 2),
                        #nn.Softmax(dim=1)
                        )
                    self.grd_all += [grd]
        
        self.AvgPool = nn.AdaptiveAvgPool2d((1,n_gsf_out))
        #Temporal Domain discriminator
        if(ablation_mask["gtd"]):
            n_gtd = 512
            self.gtd = nn.Sequential()
            self.gtd.add_module('gtd_fc1',     nn.Linear(n_gsf_out, n_gtd))
            self.gtd.add_module('gtd_bn1',     nn.BatchNorm1d(n_gtd))
            self.gtd.add_module('gtd_relu1',   nn.LeakyReLU(0.1))
            self.gtd.add_module('gtd_drop1',   nn.Dropout())
            self.gtd.add_module('gtd_fc2',     nn.Linear(n_gtd, n_gtd//2))
            self.gtd.add_module('gtd_bn2',     nn.BatchNorm1d(n_gtd//2))
            self.gtd.add_module('gtd_relu2',   nn.LeakyReLU(0.1))      
            self.gtd.add_module('gtd_fc3',     nn.Linear(n_gtd//2, 2))
            #self.gtd.add_module('gtd_softmax', nn.Softmax(dim=1))
        
        #Gy
        self.gy = nn.Sequential()
        self.gy.add_module('c_fc1', nn.Linear(n_gsf_out,n_gsf_out//2))
        self.gy.add_module('gy_bn1', nn.BatchNorm1d(n_gsf_out//2))
        self.gy.add_module('gy_relu1', nn.LeakyReLU(0.1))
        self.gy.add_module('gy_fc2', nn.Linear(n_gsf_out//2, num_class))
        #self.gy.add_module('c_softmax', nn.Softmax(dim=1))


    def forward(self, x,alpha = 1):
        
        spatial_domain_out = None
        temporal_domain_out = None
        grd_outs = None
        class_out = None

        x = self.gsf(x)
        #spatial domain out
        if(self.ablation_mask["gsd"]):
            reverse_features = ReverseLayerF.apply(x,alpha)
            spatial_domain_out = self.gsd(reverse_features.view(-1,5*self.n_gsf_out))
        #temporal aggregation 
        if(self.temporal_type == "TRN"):
            TRN_out = self.trn(x)
            if(self.ablation_mask["grd"]):
                grd_outs = torch.zeros([x.shape[0],self.n_feat[0]-1,2]).to(self.device)

                for i in range(0,self.n_feat[0]-1):
                    grd_outs[:,i,:] = self.grd_all[i](ReverseLayerF.apply(TRN_out[:,i,:],alpha))
                
                if(self.ablation_mask["domainA"]):
                #Calcolo Entropia e Attention Weights

                    softmax = nn.Softmax(dim=2)
                    logsoftmax = nn.LogSoftmax(dim=2)
                    #entropy = torch.sum(-(grd_outs) * torch.log(grd_outs), 2).nan_to_num()
                    entropy = torch.sum(-softmax(grd_outs) * logsoftmax(grd_outs), 2)
                    weights = 1-entropy
                    weights = weights.unsqueeze(2).repeat(1,1,TRN_out.shape[2]) #[32,4] -> [32,4,#feat] on dim=2 repeate the element of the second dim


                    temporal_aggregation = torch.sum((weights+1)*TRN_out,dim=1)
                else:
                    temporal_aggregation = self.AvgPool(TRN_out).reshape(TRN_out.shape[0],TRN_out.shape[2])
            else:
                #temporal_aggregation = self.AvgPool(TRN_out).reshape(TRN_out.shape[0],TRN_out.shape[2])
                temporal_aggregation = torch.sum(TRN_out,dim=1)
        else:
            temporal_aggregation =self.AvgPool(x).reshape(x.shape[0],x.shape[2])
        #temporal domain
        if(self.ablation_mask["gtd"]):
            temporal_domain_out =  self.gtd(ReverseLayerF.apply(temporal_aggregation,alpha))
        
        class_out = self.gy(temporal_aggregation)

        
        return spatial_domain_out,temporal_domain_out, class_out,grd_outs
      
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
