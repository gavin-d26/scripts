
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class HydraModule(torch.nn.Module):
    def __init__(self):
        super(HydraModule, self).__init__()    
        
    
    def train_step(self, *args, **kwargs):
        pass
    
    def eval_step(self, *args, **kwargs):
        pass
    
    def on_train_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_step_end(self, *args, **kwargs):
        pass
    
    def on_eval_epoch_end(self, *args, **kwargs):
        pass
    
    def on_eval_step_end(self, *args, **kwargs):
        pass
    
    def get_optimizers(self): 
        """
        Returns optimizers 
        
        """
        if len(self.optimizers) == 1:
            return self.optimizers[0]
        else:    
            return self.optimizers
    
    def get_loss_fn(self):
        if len(self.loss_fn) == 1:
            return self.loss_fn[0]
        else:    
            return self.loss_fn
    
    
    def get_lr_schedulers(self):
        return self.lr_schedulers
    
    def configure_optimizers_and_schedulers(self, *args, **kwargs):
        pass   ####  return  [opt1,opt2,opt3], [{'lr_scheduler': lr_obj, 'update': 'step' or 'epoch}, {...}, {...}]
    
    def configure_loss_fn(self, *args, **kwargs):
        pass    #### return list of loss functions ie [loss1, loss2, loss3]
    
    def configure_metrics(self, *args, **kwargs):
        pass   #return metrics_dict, {0:'classA', 1:'classB', 2:'classC',...}
    
    
    def get_multi_class_dict(self, metric_value):
        assert len(metric_value) == len(self.class_num_to_name)
        class_accuracy_dict = {}
        for i in range(len(metric_value)):
            class_accuracy_dict[self.class_num_to_name[i]] = metric_value[i]
        return class_accuracy_dict    
            
    def write_logs(self, epoch, wandb_p = None, writer = None):
        if writer is not None:
            writer.add_scalar('Loss/train', self.train_loss, epoch)
            writer.add_scalar('Loss/eval', self.eval_loss, epoch)
        if wandb_p is not None:    
            log_dict = {"Loss_Train":self.train_loss, "Loss_Eval": self.eval_loss}
        if hasattr(self, 'metrics'):
            if self.requires_train_metrics:
                for key, value in self.train_metrics.items():
                    if value.dim() !=1:
                        assert ValueError('dim of metric ' + key + ' not equal to 1')

                    elif len(value) > 1 and self.class_num_to_name is not None:
                        assert (len(value) == len(self.class_num_to_name))
                        multi_class_dict = self.get_multi_class_dict(value)
                        if wandb_p is not None:
                            for name, value in multi_class_dict.items():
                                log_dict[key + '_Train_' + name] = value
                        if writer is not None:        
                            writer.add_scalars('Metric/train/' + key, multi_class_dict, epoch)
                    
                    elif len(value) > 1 and self.class_num_to_name is None:
                        raise ValueError('More than one class detected, but no class_name_dict given')
                    
                    elif len(value) == 1:
                        if writer is not None:
                            writer.add_scalar('Metric/train/' + key, value, epoch) 
                        if wandb_p is not None:
                            log_dict[key +'_Train'] = value

                    else:
                        raise RuntimeError('accuracy Size confusion detected, this else block must not be reached under any circumstance')
            
            for key, value in self.eval_metrics.items():
                if value.dim() !=1:
                        assert ValueError('dim of metric ' + key + ' not equal to 1')

                elif len(value) > 1 and self.class_num_to_name is not None:
                    assert (len(value) == len(self.class_num_to_name))
                    multi_class_dict = self.get_multi_class_dict(value)
                    if wandb_p is not None:
                        for name, value in multi_class_dict.items():
                            log_dict[key + '_Eval_' + name] = value
                    if writer is not None:        
                        writer.add_scalars('Metric/eval/' + key, multi_class_dict, epoch)
                
                elif len(value) > 1 and self.class_num_to_name is None:
                        raise ValueError('More than one class detected, but no class_name_dict given')

                elif len(value) == 1:
                    if writer is not None:
                        writer.add_scalar('Metric/eval/' + key, value, epoch)
                    if wandb_p is not None:
                        log_dict[key +'_Eval'] = value

                else:
                    raise RuntimeError('accuracy Size confusion detected, this else block must not be reached under any circumstance')    
            
            if wandb_p is not None:
                wandb_p.log(log_dict)    

    def compute_metrics(self, preds, targets):     ######    COMPUTES AND RETURNS METRICS DICTIONARY OF CURRENT STEP  ######
        computed_metrics = {}
        for key, fn in self.metrics.items():
            computed_metrics[key] = fn(preds, targets).float()
            if (computed_metrics[key].dim() == 0) or ((computed_metrics[key].dim() == 1) and (len(computed_metrics[key])>1)):
                computed_metrics[key] = computed_metrics[key].unsqueeze(0)
        return computed_metrics
        
    def update_step_metrics(self, preds, targets):
        assert self.epoch_train_active != self.epoch_eval_active
        
        if self.epoch_train_active and self.requires_train_metrics:
            new_metrics = self.compute_metrics(preds, targets)
            if not hasattr(self, 'train_metrics'):
                self.train_metrics = new_metrics 
                
            else:
                for key, value in new_metrics.items():
                    self.train_metrics[key] = torch.cat((self.train_metrics[key], value),0) 
        
        elif self.epoch_eval_active: 
            new_metrics = self.compute_metrics(preds, targets)           
            if not hasattr(self, 'eval_metrics'):
                self.eval_metrics = new_metrics 
                
            else:
                for key, value in new_metrics.items():
                    self.eval_metrics[key] = torch.cat((self.eval_metrics[key], value),0)    
                
        else:
            pass      

    def update_step_loss(self, loss):
        assert self.epoch_train_active != self.epoch_eval_active
        if self.epoch_train_active:
            if not hasattr(self, 'train_loss'):
                self.train_loss = loss.unsqueeze(0) 
            else:
                self.train_loss = torch.cat((self.train_loss, loss.unsqueeze(0)), 0)     
        elif self.epoch_eval_active:
            if not hasattr(self, 'eval_loss'):
                self.eval_loss = loss.unsqueeze(0) 
            else:      
                self.eval_loss = torch.cat((self.eval_loss, loss.unsqueeze(0)), 0)    
        else:
            pass      
                        
    def compute_epoch_metrics_mean(self): 
        assert self.epoch_train_active != self.epoch_eval_active
        if self.epoch_train_active and self.requires_train_metrics:
            for key, value in self.train_metrics.items():
                if value.dim()>1:
                    self.train_metrics[key] = value.mean(0)
                elif value.dim()==1:
                    self.train_metrics[key] = value.mean(0, keepdim = True)

                else:
                    raise ValueError(key +' metric mean is not valid ie mean over value.dim() == 0')    


        
        elif self.epoch_eval_active:
            for key, value in self.eval_metrics.items():
                if value.dim()>1:
                    self.eval_metrics[key] = value.mean(0)

                elif value.dim()==1:
                    self.eval_metrics[key] = value.mean(0, keepdim = True)

                else:
                    raise ValueError(key +' metric mean is not valid ie mean over value.dim() == 0')       
        else:
            pass           
    
    
    def per_epoch_loss_and_metrics_collect(self):
        delattr(self, 'train_loss') 
        delattr(self, 'eval_loss') 
        if hasattr(self, 'metrics'):
            if self.requires_train_metrics:
                delattr(self, 'train_metrics') 
            delattr(self, 'eval_metrics')
            
    def model_checkpoint_save(self,model_checkpoint_path):
        """[saves model parameters, optimizer parameters, lr_scheduler parameters]

        Args:
            model_chekpoint_path ([str]): [path to file location]
        """
        save_file = {'model_state_dict':self.state_dict()} 
        save_file['best_checkpoint_metric'] = self.best_checkpoint_metric
        if hasattr(self, 'optimizers'): 
            for i, opt in enumerate(self.optimizers):  
                save_file['optimizer'+str(i+1)+'_state_dict'] = opt.state_dict()
        if hasattr(self, 'lr_schedulers'):
            for i, lrs_dict in enumerate(self.lr_schedulers):
                # save_file['lr_schedular'+ str(i+1)+'_state_dict'] = lrs.state_dict()
                save_file['lr_schedular'+ str(i+1)] = {'lr_schedular_state_dict':lrs_dict['lr_scheduler'].state_dict(),
                        'update': lrs_dict['update']}
        torch.save(save_file, model_checkpoint_path)      
        
    def load_all_weights(self, load_path, only_model = True):
        """[summary]

        Args:
            load_path ([str]): [path to serialized model containing 
            model, optimizer, lr_scheduler parameters]
            only_model (bool, optional): [whether to load only model parameters or 
            to include optimizer, lr_scheduler parameters]. Defaults to True.
        """
        save_file = torch.load(load_path, map_location = 'cpu')
        if only_model is True:
            self.load_state_dict(save_file['model_state_dict'])
            print('loaded model')

        else:
            self.compile_utils()
            self.get_lr_schedulers_epoch_and_step_lists()
            print('compile done')
            self.load_state_dict(save_file['model_state_dict'])
            print('loaded model')
            if hasattr(self, 'optimizers'):
                
                for i, opt in enumerate(self.optimizers):
                    opt.load_state_dict(save_file['optimizer'+str(i+1)+'_state_dict'])
                print('loaded optim')

            if hasattr(self, 'lr_schedulers'):
                
                for i, lrs_dict in enumerate(self.lr_schedulers):
                    # lrs.load_state_dict(save_file['lr_schedular'+ str(i+1)+'_state_dict'])
                    lr_state_dict_update_dict = save_file['lr_schedular'+ str(i+1)]
                    lrs_dict['lr_scheduler'].load_state_dict(lr_state_dict_update_dict['lr_schedular_state_dict'])
                    lrs_dict['update'] = lr_state_dict_update_dict['update']
                print('loaded lr_scheduler')    
                
    def scheduler_epoch_step(self, *args):
        if self.lr_schedulers is not None:
            if (self.train_steps_complete is True) and (self.epoch_train_active is True):
                if (len(self.lr_schedulers_epoch) !=0):
                    for lrs in self.lr_schedulers_epoch:
                        lrs.step()
                else:
                    pass

            elif (self.train_steps_complete is False) and (self.epoch_train_active is True):
                if len(self.lr_schedulers_perstep) != 0:
                    for lrs in self.lr_schedulers_perstep:
                        lrs.step() 
            
                else:
                    pass   
            elif (self.eval_steps_complete is True) and (self.epoch_eval_active is True):
                if (len(self.lr_schedulers_eval_epoch) !=0):
                    for lrs in self.lr_schedulers_eval_epoch:
                        lrs.step()
                        
                else:
                    pass

            elif (self.eval_steps_complete is False) and (self.epoch_eval_active is True):
                if len(self.lr_schedulers_eval_perstep) != 0:
                    for lrs in self.lr_schedulers_eval_perstep:
                        lrs.step() 
                        
                else:
                    pass   
            else:
                raise RuntimeError('INVALID LR_SCHEDULER STEP') 

        else:
            pass               

    def compile_utils(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(device = self.device)  ### move model to gpu

        self.optimizers, self.lr_schedulers = self.configure_optimizers_and_schedulers()
        self.loss_fn = self.configure_loss_fn()
        self.metrics, self.class_num_to_name = self.configure_metrics()
        assert type(self.optimizers) == list
        assert type(self.loss_fn) == list
        assert type(self.metrics) == dict

        if self.class_num_to_name is not None:
            assert type(self.class_num_to_name) == dict


    def get_lr_schedulers_epoch_and_step_lists(self):
        if self.lr_schedulers is not None:
            assert type(self.lr_schedulers) == list
            self.lr_schedulers_epoch = []
            self.lr_schedulers_perstep = []
            self.lr_schedulers_eval_perstep = []
            self.lr_schedulers_eval_epoch = []
            for lrs_dict in self.lr_schedulers:
                if lrs_dict['update'] == 'step':
                    self.lr_schedulers_perstep.append(lrs_dict['lr_scheduler'])
                    
                elif lrs_dict['update'] == 'epoch':
                    self.lr_schedulers_epoch.append(lrs_dict['lr_scheduler'])  
                
                elif lrs_dict['update'] == 'eval_step':
                    self.lr_schedulers_eval_perstep.append(lrs_dict['lr_scheduler'])
                    
                elif lrs_dict['update'] == 'eval_epoch':
                    self.lr_schedulers_eval_epoch.append(lrs_dict['lr_scheduler'])  
                    
                else:
                    raise RuntimeError('only "epoch" and "step" updates valid ')
        
    def load_data_targets(self, batch, **kwargs):
        if type(batch) is list:
            batch = list(map(lambda x: x.to(**kwargs), batch))
        else:
            batch = batch.to(**kwargs)    
        return batch
        
    def train_step_wrapper(self, batch):
        batch = self.load_data_targets(batch)
        loss, preds = self.train_step(batch)
        self.scheduler_epoch_step()    #################
        loss, preds = loss.detach(), preds.detach()


    def fit(self, train_dataset, test_dataset, epochs = 1, batch_size = 32, callbacks = None,
            num_workers = 4, logs_path =None, model_checkpoint_path = None, requires_train_metrics = False, checkpoint_metric = None,
            wandb_p = None):
        
        self.compile_utils()
        self.get_lr_schedulers_epoch_and_step_lists()
        pin_memory = True if ((torch.cuda.is_available()) and (num_workers > 0)) else False
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True,pin_memory = pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, pin_memory = pin_memory)
        
        cpu_to_gpu_tensor_processing = {'device': self.device}
        
        self.requires_train_metrics = requires_train_metrics
        
        if logs_path is not None:
            writer = SummaryWriter(logs_path)
        else:
            writer = None    
            
        for epoch in range(epochs):
            self.train()
            self.epoch_eval_active = False
            self.epoch_train_active = True
            self.train_steps_complete = False
            for batch in train_loader:
                batch = self.load_data_targets(batch)
                loss, preds = self.train_step(batch)
                loss, preds = loss.detach(), preds.detach()
                self.update_step_loss(loss)    
                if hasattr(self, 'metrics'):
                    self.update_step_metrics(preds, targets)  
                self.on_train_step_end(preds, targets, loss)
                self.scheduler_epoch_step()    #################

            self.train_steps_complete = True

            self.scheduler_epoch_step()   ######################

            self.train_loss = self.train_loss.mean()    
            if hasattr(self, 'metrics'):    
                self.compute_epoch_metrics_mean()
                
            self.on_train_epoch_end()                        # TRAIN STEP END FUNCTION CALL
            self.epoch_train_active = False
        
            self.eval()
            self.epoch_eval_active = True
            self.eval_steps_complete = False
            with torch.no_grad():
                for batch in test_loader:
                    batch = self.load_data_targets(batch)
                    loss, preds = self.eval_step(batch)
                    self.scheduler_epoch_step()    #################
                    loss, preds = loss.detach(), preds.detach()
                    
                    self.update_step_loss(loss)
                    if hasattr(self, 'metrics'):
                        self.update_step_metrics(preds, targets) 
            
                    self.on_eval_step_end(preds, targets, loss)                # EVAL STEP END FUNCTION CALL
                
                self.eval_steps_complete = True  
                self.scheduler_epoch_step() ################
                self.eval_loss = self.eval_loss.mean()    
                if hasattr(self, 'metrics'):
                    self.compute_epoch_metrics_mean()
                
                self.on_eval_epoch_end(callbacks)
            self.epoch_eval_active = False
            
            if (logs_path is not None) or (wandb_p is not None):    
                self.write_logs(epoch, wandb_p = wandb_p, writer = writer)
                
            if (model_checkpoint_path is not None) and (checkpoint_metric is not None):
                
                if (self.eval_metrics[checkpoint_metric['name']] > self.best_checkpoint_metric) and (checkpoint_metric['type'] == 'maximize'):
                    self.model_checkpoint_save(model_checkpoint_path)
                    wandb_p.run.summary["best_"+checkpoint_metric['name']] = self.eval_metrics[checkpoint_metric['name']]
                    self.best_checkpoint_metric = self.eval_metrics[checkpoint_metric['name']]
                    
                elif (self.eval_metrics[checkpoint_metric['name']] < self.best_checkpoint_metric) and (checkpoint_metric['type'] == 'minimize'):    
                    self.model_checkpoint_save(model_checkpoint_path)
                    wandb_p.run.summary["best_"+checkpoint_metric['name']] = self.eval_metrics[checkpoint_metric['name']]
                    self.best_checkpoint_metric = self.eval_metrics[checkpoint_metric['name']]
                    
                else:
                    pass
                
            print(f'epoch: {epoch} train_loss: {self.train_loss} eval_loss: {self.eval_loss}')   
            self.per_epoch_loss_and_metrics_collect()  
        if writer is not None:
            writer.close()
                
                
            
############## lr scheduler epoch or batch update step function ##########################   done
    ##############check metrics writer interface ######################  done