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
        pass   ####  return  [opt1,opt2,opt3], [{'name': 'lrs1',
    #                                            lr_scheduler': lr_obj, 
    #                                           'update': 'step' or 'epoch' or 'eval_step' or 'eval_epoch',
    #                                            'track_metric': 'metric_name' or 'eval_loss' or 'train_loss'}, {...}, {...}]
    
    # Note1: if 'update':'step' or 'eval_step', 'track_metric' is ignored
    # Note2: if 'track_metric':'eval_loss', then => 'update' : 'eval_epoch'
    #        similarly 'track_metric':'train_loss', then => 'update' : 'train_epoch'
    
    
    def configure_loss_fn(self, *args, **kwargs):
        pass    #### return list of loss functions ie [loss1, loss2, loss3]
    
    def configure_metrics(self, *args, **kwargs):
        pass   #return {'accuracy':accuracy1_class_instance, 'class_accuracy':accuracy2_class_instance, 'class_accuracy}, {0:'classA', 1:'classB', 2:'classC',...}
    
    
    def get_multi_class_dict(self, metric_value):
        assert metric_value.shape[0] == len(self.class_num_to_name)
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
                    if value.dim()!=1:
                        assert ValueError('dim of metric ' + key + ' not equal to 1')

                    elif value.shape[0] > 1 and self.class_num_to_name is not None:
                        assert (value.shape[0] == len(self.class_num_to_name))
                        multi_class_dict = self.get_multi_class_dict(value)
                        if wandb_p is not None:
                            for name, value in multi_class_dict.items():
                                log_dict[key + '_Train_' + name] = value
                        if writer is not None:        
                            writer.add_scalars('Metric/train/' + key, multi_class_dict, epoch)
                    
                    elif value.shape[0] > 1 and self.class_num_to_name is None:
                        raise ValueError('More than one class detected, but no class_name_dict given')
                    
                    elif value.shape[0] == 1:
                        if writer is not None:
                            writer.add_scalar('Metric/train/' + key, value, epoch) 
                        if wandb_p is not None:
                            log_dict[key +'_Train'] = value

                    else:
                        raise RuntimeError('accuracy Size confusion detected, this else block must not be reached under any circumstance')
            
            for key, value in self.eval_metrics.items():
                if value.dim() !=1:
                        assert ValueError('dim of metric ' + key + ' not equal to 1')

                elif value.shape[0] > 1 and self.class_num_to_name is not None:
                    assert (value.shape[0] == len(self.class_num_to_name))
                    multi_class_dict = self.get_multi_class_dict(value)
                    if wandb_p is not None:
                        for name, value in multi_class_dict.items():
                            log_dict[key + '_Eval_' + name] = value
                    if writer is not None:        
                        writer.add_scalars('Metric/eval/' + key, multi_class_dict, epoch)
                
                elif value.shape[0] > 1 and self.class_num_to_name is None:
                        raise ValueError('More than one class detected, but no class_name_dict given')

                elif value.shape[0] == 1:
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
            if (computed_metrics[key].dim() == 0) or ((computed_metrics[key].dim() == 1) and (computed_metrics[key].shape[0]>1)):
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
            

    def model_save(self, path):                     # add Grad scalar save support
        #save model weights
        model_state_dict = self.state_dict()
        
        #save optimizers
        opt_list = []
        for opt in self.optimizers:
            opt_list.append(opt.state_dict())
            
        #save lrs
        lr_scheduler_dict = {}
        for lrs_dict in self.lr_schedulers:
            lr_scheduler_dict[lrs_dict['name']] = lrs_dict['lr_scheduler'].state_dict()
        
        #final save
        save_file = {}
        save_file['best_checkpoint_metric'] = self.best_checkpoint_metric 
        save_file['model_state_dict'] = model_state_dict
        save_file['lr_scheduler_dict'] = lr_scheduler_dict
        save_file['opt_list'] = opt_list 
          
        torch.save(save_file, path)
            
            
    def load_model(self, path):                         # add Grad scalar load support
        if (hasattr(self, 'optimizers') is False) or  (hasattr(self, 'lr_schedulers') is False):
            self.optimizers, self.lr_schedulers = self.configure_optimizers_and_schedulers()
        loaded_file = torch.load(path) 
        
        #load model weights
        self.load_state_dict(loaded_file['model_state_dict'])
        
        #load optimizer state_dict
        opt_list = loaded_file['opt_list']
        for i in range(len(opt_list)):
            self.optimizers[i].load_state_dict(opt_list[i])
        
        #load lrs
            
        lr_scheduler_dict = loaded_file['lr_scheduler_dict']
        
        for lrs_dict in self.lr_schedulers:
            lrs_dict['lr_scheduler'].load_state_dict(lr_scheduler_dict[lrs_dict['name']])
            

    def scheduler_epoch_step(self):
        if self.lr_schedulers is not None:
            if (self.train_steps_complete is True) and (self.epoch_train_active is True):
                if (len(self.lr_schedulers_epoch) !=0):
                    for _, lrs_dict in self.lr_schedulers_epoch.items():
                        if 'track_metric' in lrs_dict.keys():
                            if lrs_dict['track_metric']=='train_loss':
                                lrs_dict['lr_scheduler'].step(self.train_loss)
                                #print('LRS_TRAIN_EPOCH_EXECUTED_ON_TRAIN_LOSS')
                            else:
                                lrs_dict['lr_scheduler'].step(self.train_metrics[lrs_dict['track_metric']])
                                #print('LRS_TRAIN_METRIC_TRACKED')
                        else:
                            lrs_dict['lr_scheduler'].step()
                            #print('LRS_TRAIN_EPOCH_EXECUTED')
            
                else:
                    pass

            elif (self.train_steps_complete is False) and (self.epoch_train_active is True):
                if len(self.lr_schedulers_perstep) != 0:
                    for _, lrs_dict in self.lr_schedulers_perstep.items():
                        lrs_dict['lr_scheduler'].step()
                        #print('LRS_TRAIN_STEP_EXECUTED')
                else:
                    pass   
                
            elif (self.eval_steps_complete is True) and (self.epoch_eval_active is True):
                if (len(self.lr_schedulers_eval_epoch) !=0):
                    for _, lrs_dict in self.lr_schedulers_eval_epoch.items():
                        if 'track_metric' in lrs_dict.keys():
                            if lrs_dict['track_metric']=='eval_loss':
                                lrs_dict['lr_scheduler'].step(self.eval_loss)
                                #print('LRS_EVAL_EPOCH_EXECUTED_ON_EVAL_LOSS')
                            else:
                                lrs_dict['lr_scheduler'].step(self.eval_metrics[lrs_dict['track_metric']]) 
                                #print('LRS_EVAL_METRIC_TRACKED')   
                        else:
                            lrs_dict['lr_scheduler'].step()
                            #print('LRS_EVAL_EPOCH_EXECUTED')
                        
                else:
                    pass

            elif (self.eval_steps_complete is False) and (self.epoch_eval_active is True):
                if len(self.lr_schedulers_eval_perstep) != 0:
                    for _, lrs_dict in self.lr_schedulers_eval_perstep.items():
                        lrs_dict['lr_scheduler'].step()
                        #print('LRS_EVAL_STEP_EXECUTED')
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
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()    


    def get_lr_schedulers_epoch_and_step_dict(self):
        if self.lr_schedulers is not None:
            assert type(self.lr_schedulers) == list
            self.lr_schedulers_epoch = {}
            self.lr_schedulers_perstep = {}
            self.lr_schedulers_eval_perstep = {}
            self.lr_schedulers_eval_epoch = {}
            for lrs_dict in self.lr_schedulers:
                if lrs_dict['update'] == 'step':
                    self.lr_schedulers_perstep[lrs_dict['name']] = lrs_dict
                    
                elif lrs_dict['update'] == 'epoch':
                    self.lr_schedulers_epoch[lrs_dict['name']] = lrs_dict  
                
                elif lrs_dict['update'] == 'eval_step':
                    self.lr_schedulers_eval_perstep[lrs_dict['name']] = lrs_dict
                    
                elif lrs_dict['update'] == 'eval_epoch':
                    self.lr_schedulers_eval_epoch[lrs_dict['name']] = lrs_dict  
                    
                else:
                    raise RuntimeError('only "epoch" and "step" updates valid ')
        
    def load_data_targets(self, batch, **kwargs):
        if type(batch) is list:
            batch = list(map(lambda x: x.to(**kwargs), batch))
        else:
            batch = batch.to(**kwargs)    
        return batch
        
    def train_step_wrapper(self, batch, cpu_to_gpu_tensor_processing):
        batch = self.load_data_targets(batch, **cpu_to_gpu_tensor_processing)         
        loss, preds, targets = self.train_step(batch)
        loss, preds = loss.detach(), preds.detach()   
        self.update_step_loss(loss)    
        if hasattr(self, 'metrics') and (targets is not None):
            self.update_step_metrics(preds, targets)  
        self.on_train_step_end(preds, targets, loss)
        self.scheduler_epoch_step() 
        
    def train_on_steps_completion_wrapper(self):   
        self.train_loss = self.train_loss.mean()    
        if hasattr(self, 'metrics') and hasattr(self, 'train_metrics'):   
            self.compute_epoch_metrics_mean()
        self.scheduler_epoch_step()   ######################
        self.on_train_epoch_end()        
    
    
    def eval_step_wrapper(self, batch, cpu_to_gpu_tensor_processing):  
        batch = self.load_data_targets(batch, **cpu_to_gpu_tensor_processing)
        #print(batch[0].shape, batch[1].shape)   ##########################################################
        loss, preds, targets = self.eval_step(batch)
        loss, preds = loss.detach(), preds.detach()
        self.update_step_loss(loss)
        if hasattr(self, 'metrics') and (targets is not None):
            self.update_step_metrics(preds, targets) 
        self.on_eval_step_end(preds, targets, loss)                # EVAL STEP END FUNCTION CALL
        self.scheduler_epoch_step()    #################   
        
    def eval_on_steps_completion_wrapper(self):
        self.eval_loss = self.eval_loss.mean()    
        if hasattr(self, 'metrics') and hasattr(self, 'eval_metrics'):
            self.compute_epoch_metrics_mean()
        self.scheduler_epoch_step() ################    
        self.on_eval_epoch_end(self.callbacks)    
        


    def fit(self, 
            train_dataset, 
            test_dataset, 
            epochs = 1, 
            batch_size = 32, 
            callbacks = None,
            num_workers = 4, 
            logs_path =None, 
            model_checkpoint_path = None, 
            requires_train_metrics = False, 
            checkpoint_metric = None,
            wandb_p = None,
            mixed_precision = False):
        
        self.mixed_precision = mixed_precision
        self.compile_utils()
        self.get_lr_schedulers_epoch_and_step_dict()
        pin_memory = True if ((torch.cuda.is_available()) and (num_workers > 0)) else False
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True,pin_memory = pin_memory)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, pin_memory = pin_memory)
        
        self.best_checkpoint_metric = 0.0 if checkpoint_metric['type']=='maximize' else 1000.0 
        cpu_to_gpu_tensor_processing = {'device': self.device}
        
        self.requires_train_metrics = requires_train_metrics
        
        if logs_path is not None:
            writer = SummaryWriter(logs_path)
        else:
            writer = None    
        self.callbacks = callbacks
            
        for epoch in range(epochs):
            print('Epoch: ', epoch)  #######################################################################
            self.train()
            self.epoch_eval_active = False
            self.epoch_train_active = True
            self.train_steps_complete = False
            for batch in tqdm(train_loader, desc='Train'):
                self.train_step_wrapper(batch, cpu_to_gpu_tensor_processing)
                
            self.train_steps_complete = True
            
            self.train_on_steps_completion_wrapper()
            
            self.epoch_train_active = False
        
            self.eval()
            self.epoch_eval_active = True
            self.eval_steps_complete = False
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Eval '):
                    self.eval_step_wrapper(batch, cpu_to_gpu_tensor_processing)    #################
                
                self.eval_steps_complete = True  
                
                self.eval_on_steps_completion_wrapper()
                
            self.epoch_eval_active = False
            
            if (logs_path is not None) or (wandb_p is not None):    
                self.write_logs(epoch, wandb_p = wandb_p, writer = writer)
                
            if (model_checkpoint_path is not None) and (checkpoint_metric is not None):
                
                if (self.eval_metrics[checkpoint_metric['name']] > self.best_checkpoint_metric) and (checkpoint_metric['type'] == 'maximize'):
                    self.model_save(model_checkpoint_path)
                    if wandb_p is not None:
                        wandb_p.run.summary["best_"+checkpoint_metric['name']] = self.eval_metrics[checkpoint_metric['name']]
                    self.best_checkpoint_metric = self.eval_metrics[checkpoint_metric['name']]
                    
                elif (self.eval_metrics[checkpoint_metric['name']] < self.best_checkpoint_metric) and (checkpoint_metric['type'] == 'minimize'):    
                    self.model_save(model_checkpoint_path)
                    if wandb_p is not None:
                        wandb_p.run.summary["best_"+checkpoint_metric['name']] = self.eval_metrics[checkpoint_metric['name']]
                    self.best_checkpoint_metric = self.eval_metrics[checkpoint_metric['name']]
                    
                else:
                    pass
                
            #print(f'epoch: {epoch} train_loss: {self.train_loss} eval_loss: {self.eval_loss}')   
            self.per_epoch_loss_and_metrics_collect()  
        if writer is not None:
            writer.close()
                
                
            
############## lr scheduler epoch or batch update step function ##########################   done
    ##############check metrics writer interface ######################  done