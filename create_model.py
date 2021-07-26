import torch
import torch.nn as nn
import torch.nn.functional as F
import hydramodule 
import torchvision




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
    def __call__(self, preds, targets, threshold = 0.5):
        accurate_preds = (preds > threshold).float()
        accurate_preds = (accurate_preds == targets).float().mean()
        return accurate_preds

class Accuracy2():
    def __call__(self, preds, targets, threshold = 0.5):
        accurate_preds = (preds > threshold).float()
        accurate_preds = (accurate_preds == targets).float().mean(0)
        return accurate_preds
    

class Resnet34(hydramodule.HydraModule):
    def __init__(self, config, unrecorded_defaults, num_classes = 15):
        super(Resnet34, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.unrecorded_defaults = unrecorded_defaults
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
        optimizer1 = torch.optim.Adam(self.parameters(), lr = self.config.lr, 
                                    betas = (self.config.beta1, self.config.beta2),
                                    weight_decay = self.config.weight_decay)
        
        lrs_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor = self.unrecorded_defaults['lr_decay'],
                                                         patience = self.unrecorded_defaults['Patience'])
        lr1_dict = {'name':'steplr1', 'lr_scheduler': lrs_1, 'update': 'eval_epoch', 'track_metric':'eval_loss'}
        
        return [optimizer1], [lr1_dict]

    def configure_metrics(self):
        a = Accuracy1()
        b = Accuracy2()
        d = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2 ,'Infiltration':3,
            'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,'Consolidation':8,
            'Edema':9,'Emphysema':10,'Fibrosis':11,'Pleural_Thickening':12, 
            'Hernia':13, 'No Finding':14}
        return {'accuracy':a, 'class_accuracy':b}, d



    def train_step(self, batch):
        data, targets = batch
        opt = self.get_optimizers()
        #print('optim_lr: ',opt.state_dict())
        loss_fn = self.get_loss_fn()
        opt.zero_grad()
        preds = self(data)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        return loss, preds, targets


    def eval_step(self, batch):
        data, targets = batch
        loss_fn = self.get_loss_fn()
        preds = self(data)
        loss = loss_fn(preds, targets)
        return loss, preds, targets     