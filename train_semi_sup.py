import argparse
import collections
import numpy as np

from data_loader.data_loader_semi_sup import *
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer.trainer_semi_sup import Trainer
from utils.util import *
from model.dream_semi_sup import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def main(config, fold_id, data_config):    
    batch_size = config["data_loader"]["args"]["batch_size"]
    params = config['hyper_params']
    
    train_sup = SleepDataLoader(config, folds_data[fold_id]['train_sup'], phase='train_sup')
    supervised_loader = DataLoader(dataset=train_sup, shuffle=True, batch_size = batch_size)
    
    train_unsup = SleepDataLoader(config, folds_data[fold_id]['train_unsup'], domain_dict=train_sup.domain_dict, phase='train_unsup')
    unsupervised_loader = DataLoader(dataset=train_unsup, shuffle=True, batch_size = batch_size, drop_last=True)
    
    valid_dataset = SleepDataLoader(config, folds_data[fold_id]['valid'], phase='valid')
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size = batch_size) 
    test_dataset = SleepDataLoader(config, folds_data[fold_id]['test'], phase='test')
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size = batch_size) 
        
    n_domains = train_unsup.n_domains
    
    # build model architecture
    

    feature_net = VAE(params['zd_dim'] , params['zy_dim'], n_domains, config, data_config['d_type']) 
    classifier = Transformer(input_size=params['zy_dim'], config=config) 

    logger.info(feature_net)
    logger.info(classifier)
    logger.info("-"*100)


    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    featurenet_parameters = filter(lambda p: p.requires_grad, feature_net.parameters())
    classifier_parameters = filter(lambda p: p.requires_grad, classifier.parameters())

    featurenet_optimizer = config.init_obj('optimizer', torch.optim, featurenet_parameters)
    classifier_optimizer = config.init_obj('optimizer', torch.optim, classifier_parameters)
    
    
    trainer = Trainer(feature_net, classifier, 
                      featurenet_optimizer, classifier_optimizer,
                      criterion, metrics,
                      config=config,
                      fold_id=fold_id,
                      supervised_loader=supervised_loader,
                      unsupervised_loader=unsupervised_loader, 
                      valid_loader=valid_loader,
                      test_loader=test_loader
                     )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')



    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id)
    
    data_config = dict()
    folds_data = load_folds_semi_sup(config["data_loader"]["args"]["num_folds"], fold_id)
    data_config['d_type'] = 'edf'
    data_config['sampling_rate'] = 100
    

    main(config, fold_id, data_config)
