import torch
import torch.nn as nn
import torch.nn.functional as F
import hydramodule 
import torchvision

torch.manual_seed(0)


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
        d = {0:'Atelectasis', 1:'Cardiomegaly', 2:'Effusion' , 3:'Infiltration',
            4:'Mass', 5:'Nodule', 6:'Pneumonia', 7:'Pneumothorax', 8:'Consolidation',
            9:'Edema', 10:'Emphysema', 11:'Fibrosis', 12:'Pleural_Thickening', 
            13:'Hernia', 14:'No Finding'}
        return {'accuracy':a, 'class_accuracy':b}, d



    def train_step(self, batch):
        data, targets = batch
        opt = self.get_optimizers()
        #print('optim_lr: ',opt.state_dict())
        loss_fn = self.get_loss_fn()
        opt.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            preds = self(data)
            loss = loss_fn(preds, targets)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()
        return loss, preds, targets


    def eval_step(self, batch):
        data, targets = batch
        loss_fn = self.get_loss_fn()
        preds = self(data)
        loss = loss_fn(preds, targets)
        return loss, preds, targets     