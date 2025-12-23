# -*- coding: UTF-8 -*-
"""
Optuna Hyperparameter Search for CoGCL in ReChorus Framework

Usage:
    python run_optuna.py \
        --model_name CoGCL \
        --dataset Grocery_and_Gourmet_Food \
        --n_trials 999 \
        --epoch 100 \
        --gpu 0

Features:
    - Continuous hyperparameter ranges with log/linear scales
    - Pruning support for early termination of bad trials
    - SQLite storage for resumable studies
    - Results export to CSV
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import gc
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
# CoGCL 动态导入：根据 code_mix_alpha 参数选择模型版本
# 在 main() 函数中动态导入
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search for CoGCL')
    
    # Basic settings
    parser.add_argument('--model_name', type=str, default='CoGCL', help='Model name')
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food', help='Dataset name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='', help='Optuna study name (for resumable studies)')
    parser.add_argument('--storage', type=str, default='', help='SQLite storage path for Optuna')
    parser.add_argument('--pruning', type=int, default=1, help='Enable pruning (1=enabled, 0=disabled)')
    
    # Training settings (can be overridden)
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stop patience')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Eval batch size')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers')
    
    # Data settings
    parser.add_argument('--path', type=str, default='../data/', help='Data path')
    parser.add_argument('--sep', type=str, default='\t', help='Separator')
    
    # Evaluation settings
    parser.add_argument('--main_metric', type=str, default='NDCG@10', help='Main metric for optimization')
    parser.add_argument('--topk', type=str, default='5,10,20,50', help='Top-K values')
    parser.add_argument('--metric', type=str, default='NDCG,HR', help='Metrics')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='../results/', help='Output directory')
    
    # QER (Quantization-Enhanced Representation) settings
    parser.add_argument('--code_mix_alpha', type=float, default=0.0,
                        help='QER fusion coefficient. If !=0, use CoGCL_new.py; if ==0, use CoGCL.py')
    
    return parser.parse_args()


def suggest_hyperparameters(trial: optuna.Trial) -> Dict:
    """
    Define the hyperparameter search space using Optuna's suggest methods.
    Uses log scale for loss weights to cover multiple orders of magnitude.
    """
    params = {
        # Core loss weights (log scale, covers 4 orders of magnitude)
        'vq_loss_weight': trial.suggest_float('vq_loss_weight', 0.001, 10.0, log=True),
        'cl_weight': trial.suggest_float('cl_weight', 0.001, 10.0, log=True),
        'sim_cl_weight': trial.suggest_float('sim_cl_weight', 0.0, 1.0),
        
        # Graph augmentation parameters (linear scale)
        'graph_replace_p': trial.suggest_float('graph_replace_p', 0.0, 0.5),
        'graph_add_p': trial.suggest_float('graph_add_p', 0.0, 0.5),
        
        # Dropout and regularization
        'drop_p': trial.suggest_float('drop_p', 0.0, 0.5),
        
        # GNN structure
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        
        # Training parameters (log scale)
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'l2': trial.suggest_float('l2', 1e-7, 1e-4, log=True),
        
        # Contrastive learning temperature
        'cl_tau': trial.suggest_float('cl_tau', 0.05, 0.5),
    }
    return params


class OptunaRunner:
    """
    Modified runner that supports Optuna pruning via intermediate value reporting.
    """
    
    def __init__(self, args, trial: optuna.Trial = None):
        self.epoch = args.epoch
        self.early_stop = args.early_stop
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers
        self.pin_memory = 0
        self.trial = trial
        
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = args.main_metric
        self.main_topk = int(self.main_metric.split("@")[1]) if "@" in self.main_metric else 10
        
    def _build_optimizer(self, model, lr, l2):
        optimizer = torch.optim.Adam(model.customize_parameters(), lr=lr, weight_decay=l2)
        return optimizer
    
    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        evaluations = dict()
        gt_rank = (predictions >= predictions[:, 0].reshape(-1, 1)).sum(axis=-1)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
        return evaluations
    
    def train(self, data_dict: Dict, lr: float, l2: float) -> float:
        """
        Train the model and return the best dev metric.
        Supports Optuna pruning by reporting intermediate values.
        """
        model = data_dict['train'].model
        optimizer = self._build_optimizer(model, lr, l2)
        model.optimizer = optimizer
        
        main_metric_results = []
        best_metric = 0.0
        
        for epoch in range(self.epoch):
            # Training
            gc.collect()
            torch.cuda.empty_cache()
            loss = self._fit(data_dict['train'], epoch + 1)
            
            if np.isnan(loss):
                logging.warning(f"Loss is NaN at epoch {epoch + 1}. Stopping.")
                return 0.0
            
            # Evaluation
            dev_result = self._evaluate(data_dict['dev'], [self.main_topk], self.metrics)
            current_metric = dev_result[self.main_metric]
            main_metric_results.append(current_metric)
            
            if current_metric > best_metric:
                best_metric = current_metric
            
            # Report to Optuna for pruning
            if self.trial is not None:
                self.trial.report(current_metric, epoch)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    logging.info(f"Trial pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()
            
            # Early stopping check
            if self.early_stop > 0 and len(main_metric_results) > self.early_stop:
                if self._should_stop(main_metric_results):
                    logging.info(f"Early stop at epoch {epoch + 1}")
                    break
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch + 1}: loss={loss:.4f}, {self.main_metric}={current_metric:.4f}")
        
        return best_metric
    
    def _fit(self, dataset, epoch: int) -> float:
        model = dataset.model
        dataset.actions_before_epoch()
        model.train()
        
        loss_lst = []
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, collate_fn=dataset.collate_batch,
                        pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, desc=f'Epoch {epoch}', ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            
            # Shuffle items
            item_ids = batch['item_id']
            indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)
            batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]
            
            model.optimizer.zero_grad()
            out_dict = model(batch)
            
            # Restore prediction order
            prediction = out_dict['prediction']
            if len(prediction.shape) == 2:
                restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
                restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction
                out_dict['prediction'] = restored_prediction
            
            loss = model.loss(out_dict, batch)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        
        return np.mean(loss_lst).item()
    
    def _evaluate(self, dataset, topks: list, metrics: list) -> Dict[str, float]:
        dataset.model.eval()
        predictions = []
        
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False,
                        num_workers=self.num_workers, collate_fn=dataset.collate_batch,
                        pin_memory=self.pin_memory)
        
        with torch.no_grad():
            for batch in dl:
                batch = utils.batch_to_gpu(batch, dataset.model.device)
                prediction = dataset.model(batch)['prediction']
                predictions.extend(prediction.cpu().data.numpy())
        
        predictions = np.array(predictions)
        return self.evaluate_method(predictions, topks, metrics)
    
    def _should_stop(self, criterion: list) -> bool:
        # Use the same logic as BaseRunner.eval_termination()
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False


def create_objective(base_args, corpus, model_class):
    """
    Create the Optuna objective function.
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Get hyperparameters from Optuna
        hp = suggest_hyperparameters(trial)
        
        # Create args namespace with hyperparameters
        args = argparse.Namespace(**vars(base_args))
        
        # Apply hyperparameters
        args.lr = hp['lr']
        args.l2 = hp['l2']
        args.vq_loss_weight = hp['vq_loss_weight']
        args.cl_weight = hp['cl_weight']
        args.sim_cl_weight = hp['sim_cl_weight']
        args.graph_replace_p = hp['graph_replace_p']
        args.graph_add_p = hp['graph_add_p']
        args.drop_p = hp['drop_p']
        args.n_layers = hp['n_layers']
        args.cl_tau = hp['cl_tau']
        
        # Fixed hyperparameters
        args.embedding_size = 64
        args.user_code_num = 4
        args.item_code_num = 4
        args.user_code_size = 256
        args.item_code_size = 256
        args.code_dist = 'cos'
        args.code_dist_tau = 0.2
        args.code_batch_size = 2048
        args.vq_type = 'rq'
        args.vq_ema = 0
        args.drop_fwd = 1
        args.data_aug_delay = 0
        args.num_neg = 1
        args.dropout = 0
        args.test_all = 0
        args.buffer = 1
        args.model_path = ''
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Trial {trial.number}: {hp}")
        logging.info(f"{'='*50}")
        
        try:
            # Create model
            model = model_class(args, corpus).to(args.device)
            
            # Create datasets
            data_dict = {}
            for phase in ['train', 'dev', 'test']:
                data_dict[phase] = model_class.Dataset(model, corpus, phase)
                data_dict[phase].prepare()
            
            # Train with pruning support
            runner = OptunaRunner(args, trial=trial)
            best_metric = runner.train(data_dict, hp['lr'], hp['l2'])
            
            logging.info(f"Trial {trial.number} finished: {args.main_metric}={best_metric:.4f}")
            
            # Clear memory
            del model, data_dict
            gc.collect()
            torch.cuda.empty_cache()
            
            return best_metric
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    return objective


def main():
    args = parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.output_dir, f'optuna_{args.model_name}_{args.dataset}_{timestamp}.log')
    utils.check_dir(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Optuna Hyperparameter Search for {args.model_name} on {args.dataset}")
    logging.info(f"Args: {args}")
    
    # Set up GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info(f"Device: {args.device}")
    
    # Set random seed
    utils.init_seed(args.random_seed)
    
    # 动态导入 CoGCL 模型：根据 code_mix_alpha 参数选择版本
    if args.model_name == 'CoGCL':
        if args.code_mix_alpha != 0:
            from models.CoGCL_new import CoGCL
            logging.info(f"Using CoGCL_new.py (QER enabled, code_mix_alpha={args.code_mix_alpha})")
        else:
            from models.CoGCL import CoGCL
            logging.info("Using CoGCL.py (Original, code_mix_alpha=0)")
    
    # Load model class
    model_class = eval(args.model_name)
    reader_class = eval(f'{model_class.reader}.{model_class.reader}')
    
    # Set reader args
    args.regenerate = 0
    
    # Load corpus
    corpus_path = os.path.join(args.path, args.dataset, f'{model_class.reader}.pkl')
    if os.path.exists(corpus_path):
        logging.info(f'Load corpus from {corpus_path}')
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        logging.info(f'Building corpus...')
        corpus = reader_class(args)
        logging.info(f'Save corpus to {corpus_path}')
        pickle.dump(corpus, open(corpus_path, 'wb'))
    
    # Set up Optuna study with SQLite storage for optuna-dashboard support
    study_name = args.study_name if args.study_name else f'{args.model_name}_{args.dataset}'
    
    # Default to SQLite storage for dashboard visualization
    if args.storage:
        db_path = args.storage
    else:
        db_path = os.path.join(args.output_dir, f'optuna_{args.model_name}_{args.dataset}.db')
    
    utils.check_dir(db_path)
    storage = f'sqlite:///{db_path}'
    logging.info(f"Optuna storage: {db_path}")
    logging.info(f"To view results, run: optuna-dashboard {db_path}")
    
    # Create sampler and pruner
    sampler = TPESampler(seed=args.random_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10) if args.pruning else optuna.pruners.NopPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize NDCG
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )
    
    # Create objective function
    objective = create_objective(args, corpus, model_class)
    
    # Run optimization
    logging.info(f"Starting Optuna optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    logging.info("\n" + "="*60)
    logging.info("OPTIMIZATION FINISHED")
    logging.info("="*60)
    
    logging.info(f"\nBest trial:")
    logging.info(f"  Value ({args.main_metric}): {study.best_trial.value:.4f}")
    logging.info(f"  Params:")
    for key, value in study.best_trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # Save results to CSV
    results_df = study.trials_dataframe()
    results_path = os.path.join(args.output_dir, f'optuna_{args.model_name}_{args.dataset}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"\nResults saved to: {results_path}")
    
    # Save best params
    best_params_path = os.path.join(args.output_dir, f'best_params_{args.model_name}_{args.dataset}_{timestamp}.txt')
    with open(best_params_path, 'w') as f:
        f.write(f"# Best hyperparameters for {args.model_name} on {args.dataset}\n")
        f.write(f"# {args.main_metric}: {study.best_trial.value:.4f}\n\n")
        for key, value in study.best_trial.params.items():
            f.write(f"--{key} {value}\n")
    logging.info(f"Best params saved to: {best_params_path}")
    
    return study


if __name__ == '__main__':
    study = main()
