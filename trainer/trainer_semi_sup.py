import numpy as np
import torch
from base_trainer import BaseTrainer
from utils import MetricTracker
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, feature_net, classifier, featurenet_optimizer, classifier_optimizer, 
                 criterion, metric_ftns, config, fold_id, supervised_loader, unsupervised_loader,
                 valid_loader=None, test_loader=None):
        super().__init__(feature_net, classifier, featurenet_optimizer, classifier_optimizer, 
                         criterion, metric_ftns, config, fold_id)
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.n_trains_featurenet = len(supervised_loader)+len(unsupervised_loader)

        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.test_loader = test_loader
        self.do_test = self.test_loader is not None
        self.lr_scheduler_f = featurenet_optimizer
        self.lr_scheduler_c = classifier_optimizer
        self.log_step = int(supervised_loader.batch_size) * 0.5  # reduce this if you want more logs
        

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
       

##################### Train Feature Net            

    def _train_feature_net(self, epoch):
        self.feature_net.train()
        self.train_metrics.reset()
        
        self.feature_net.beta_d = min([self.config['hyper_params']['beta_d'], self.config['hyper_params']['beta_d'] * (epoch * 1.) / self.config['hyper_params']['warmup']])
        self.feature_net.beta_y = min([self.config['hyper_params']['beta_y'], self.config['hyper_params']['beta_y'] * (epoch * 1.) / self.config['hyper_params']['warmup']])

        outs, trgs = np.array([]), np.array([])
        if self.predict_d:
            outs_d, trgs_d = np.array([]), np.array([])
        
        sup_iter = iter(self.supervised_loader)
        unsup_iter = iter(self.unsupervised_loader)
        
        unsupervised_exist = True
        for iter_ in range(self.n_trains_featurenet):
            if unsupervised_exist: 
                try:
                    (x, y, d) = next(unsup_iter)
                    x, y, d = x.to(self.device), None, d.to(self.device)
                    phase = 'UnSupervised learning'
                except:
                    (x, y, d) = next(sup_iter)
                    x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)
                    phase = 'Supervised learning'
                    unsupervised_exist = False
            else:
                (x, y, d) = next(sup_iter)
                x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)
                phase = 'Supervised learning'


            self.featurenet_optimizer.zero_grad()
            all_loss = self.feature_net.get_losses(x=x, y=y, d=d)
            all_loss.backward()
            self.featurenet_optimizer.step()

            if phase == 'Supervised learning':
                output = self.feature_net.predict(x)
                loss = self.criterion(output, y, self.class_weights, self.device)
                self.train_metrics.update('loss', loss.item())
                preds_ = output.data.max(1, keepdim=True)[1].cpu().numpy() 
                outs = np.append(outs, preds_)
                trgs = np.append(trgs, y.data.cpu().numpy())
                acc_class = accuracy_score(y.data.cpu().numpy().reshape(-1,1), preds_.reshape(-1,1))    
  
             
            if iter_ % 25 == 0:
                self.logger.debug('Train Epoch: {} {}  {} - Loss: {:.6f} ContLoss: {:.6f}'.format(
                    epoch,
                    self._progress(iter_),
                    phase,
                    all_loss.item(),
                    Conts_loss.item()
                ))

                    
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        
        log = self.train_metrics.result()


        if self.do_validation:
            val_log = self._valid_feature_net()
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    def _valid_feature_net(self):

        self.feature_net.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                output = self.feature_net.predict(x)
                loss = self.criterion(output, y, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, y.data.cpu().numpy())
                
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))

        return self.valid_metrics.result()
    
    def _test_feature_net(self):

        PATH = str(self.checkpoint_dir / 'featurenet_best.pth')
        self.feature_net.load_state_dict(torch.load(PATH)['state_dict'])
        self.feature_net.eval()
        
        val_log = self._valid_feature_net()
        
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)

                output = self.feature_net.predict(x)
                loss = self.criterion(output, y, self.class_weights, self.device)

                self.test_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, y.data.cpu().numpy())
            
        
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        test_log = self.test_metrics.result()
        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Representation learning is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
            
            
##################### Train Classification Net            

    def _train_classifier(self, epoch):

        
        self.classifier.train()
        self.train_metrics.reset()

        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (x, y, _) in enumerate(self.supervised_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.classifier_optimizer.zero_grad()
            
            features = self.feature_net.get_features(x)
            loss = self.classifier.get_loss(features, y)
            output = self.classifier.predict(features)
            
            loss.backward()
            self.classifier_optimizer.step()
            
                
            self.train_metrics.update('loss', loss.item())
                
            outs = np.append(outs, np.array(output))
            trgs = np.append(trgs, y.data.cpu().numpy())
            
            accuracy = accuracy_score(y.data.cpu().numpy().reshape(-1,1), np.array(output).reshape(-1,1))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    accuracy
                    
                ))
                
                            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
            
        log = self.train_metrics.result()
        
        if self.do_validation:
            val_log = self._valid_classifier()
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    def _valid_classifier(self):
        
        self.feature_net.eval()
        self.classifier.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                features = self.feature_net.get_features(x)
                
                loss = self.classifier.get_loss(features, y)
                output = self.classifier.predict(features)

                self.valid_metrics.update('loss', loss.item())
                    
                preds_ = np.array(output) 
                outs = np.append(outs, preds_)
                trgs = np.append(trgs, y.data.cpu().numpy())
                
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))

        return self.valid_metrics.result()
    
    def _test_classifier(self):

        PATH_f = str(self.checkpoint_dir / 'featurenet_best.pth')
        self.feature_net.load_state_dict(torch.load(PATH_f)['state_dict'])
        self.feature_net.eval()
            
        PATH_c = str(self.checkpoint_dir / 'classifier_best.pth')
        self.classifier.load_state_dict(torch.load(PATH_c)['state_dict'])
        self.classifier.eval()
        
        val_log = self._valid_classifier()
        
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                features = self.feature_net.get_features(x)

                loss = self.classifier.get_loss(features, y)
                output = self.classifier.predict(features)

                self.test_metrics.update('loss', loss.item())
                    
                preds_ = np.array(output) 
                outs = np.append(outs, preds_)
                trgs = np.append(trgs, y.data.cpu().numpy())
            
        outs_name = "test_outs_" + str(self.fold_id)
        trgs_name = "test_trgs_" + str(self.fold_id)
        np.save(self.checkpoint_dir / outs_name, outs)
        np.save(self.checkpoint_dir / trgs_name, trgs)   
        
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        test_log = self.test_metrics.result()
        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Test is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))            
            
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.n_trains_featurenet
        return base.format(current, total, 100.0 * current / total)
