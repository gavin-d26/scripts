import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra 




"""
learning_rate:
beta1:
beta2:
batch_size:
weight_decay:
learning_rate_decay:
patience: 6

"""

class Accuracy1():
    def __call__(self, preds, targets):
        accurate_preds = (preds > threshold).float()
        accurate_preds = (accurate_preds == targets).float().mean()
        return accurate_preds

class Accuracy2():
    def __call__(self, preds, targets, threshold = 0.5):
        accurate_preds = (preds > threshold).float()
        accurate_preds = (accurate_preds == targets).float().mean(0)
        return accurate_preds
    

class mymodel(hydra.HydraModule):
    def __init__(self, config, num_classes = 15):
        super(mymodel, self).__init__()
        self.config = config
        self.num_classes = num_classes
        backbone = torchvision.models.resnet34(pretrained = True)
        # for name, layer in backbone.named_children():
        #     if (('1' in name) or ('2' in name))and ('bn' not in name):
        #         for p in layer.parameters():
        #             p.requires_grad = False
            
        # for name,mod in backbone.named_modules(): 
        #     if ('bn' in name):
        #         for p in mod.parameters():
        #             p.requires_grad = True    
                    
        backbone.fc = nn.Linear(512, self.num_classes, bias=True)
        self.backbone = backbone
        
    def forward(self, input):
        return self.backbone(input)

    
    def configure_loss_fn(self):
        return [torch.nn.BCEWithLogitsLoss()]

    def configure_optimizers_and_schedulers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr = self.config['learning_rate'], 
                                    betas = (self.config['beta1'], self.config['beta2']),
                                    weight_decay = self.config['weight_decay'])
        lr1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor = self.config['learning_rate_decay'], patience = self.config['patience'])
        return [optimizer1], [{'lr_scheduler': lr1, 'update': 'eval_epoch'}]

    def configure_metrics(self):
        a = Accuracy1()
        b = Accuracy2()
        d = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2 ,'Infiltration':3,
            'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,'Consolidation':8,
            'Edema':9,'Emphysema':10,'Fibrosis':11,'Pleural_Thickening':12, 
            'Hernia':13, 'No Finding':14}
        return {'accuracy':a, 'class_accuracy':b}, d



    def train_step(self, data, targets):
        opt = self.get_optimizers()
        loss_fn = self.get_loss_fn()
        opt.zero_grad(set_to_none=True)
        preds = self(data)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        return loss, preds


    def eval_step(self, data, targets):
        loss_fn = self.get_loss_fn()
        preds = self(data)
        loss = loss_fn(preds, targets)
        return loss, preds     