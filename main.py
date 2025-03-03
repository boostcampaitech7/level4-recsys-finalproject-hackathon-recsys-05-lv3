import os
import sys
import time
import argparse
import ast
import multiprocessing
from os.path import join

import torch
import torch.optim as optimizer_module
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from src import utils, wandblogger
import src.models as model_module
from src.data import dataloader
from src.data.preprocessing import ML32M_PreprocessingData, ML1M_PreprocessingData
from src.lightgcn_utils.trainer import Trainer
import src.lightgcn_utils.loss as loss_module



def main(args) :
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    args.CODE_PATH = join(ROOT_PATH, 'code')
    args.DATA_PATH = join(args.CODE_PATH, 'data')
    args.BOARD_PATH = join(args.CODE_PATH, 'runs')
    args.FILE_PATH = join(args.CODE_PATH, 'checkpoints')
    args.CORES = multiprocessing.cpu_count() // 2

    if not os.path.exists(args.FILE_PATH):
        os.makedirs(args.FILE_PATH, exist_ok=True)

    if args.dataset.data == "MovieLens32M":
        preprocessing = ML32M_PreprocessingData(args)
    else:
        preprocessing = ML1M_PreprocessingData(args)
    print("preprocessing done")

    args.popular_items = preprocessing.popular_items
    print(f"Loaded {len(args.popular_items)} popular items.")

    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)


    dataset = dataloader.Loader(args,path="./data/"+args.dataset.data+args.dataset.preprocess_dir)

    Recmodel = getattr(model_module, args.model)(args, dataset)
    Recmodel = Recmodel.to(args.device)

    weight_file = utils.getFileName(args)

    print(f"load and save to {weight_file}")
    
    if args.train.resume:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    
    loss = getattr(loss_module, args.loss)(Recmodel,args)

    w = SummaryWriter(join(args.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.memo)) if args.tensorboard else None
    if not args.tensorboard:
        print("TensorBoard logging is disabled.")
    
    wandb_logger = wandblogger.WandbLogger(args)

    trainer = Trainer(args,dataset,Recmodel,loss, w)

    save_interval = args.train['save_interval']

    try:
        print("[TEST]")
        results = trainer.test()
        results = {key: value.tolist() for key, value in results.items()}
        
        print("[TEST ColdItem]")
        cold_results = trainer.test_cold()
        cold_results = {key: value.tolist() for key, value in cold_results.items()}


        print("[TRAIN]")
        for epoch in range(args.train.epochs):
            output_information, aver_loss = trainer.train()
            wandb_logger.log_metrics({"train_loss": aver_loss}, head="train", epoch = epoch+1)
            print(f'EPOCH[{epoch+1}/{args.train.epochs}] {output_information}')
            
            if (epoch + 1) % save_interval == 0:
                torch.save(Recmodel.state_dict(), weight_file)
                print(f"Model saved at epoch {epoch+1}")

            print("[TEST]")
            results = trainer.test()
            results = {key: value.tolist() for key, value in results.items()}
            wandb_logger.log_metrics(results,epoch=epoch+1)

            print("[TEST ColdItem]")
            cold_results = trainer.test_cold()
            cold_results = {key: value.tolist() for key, value in cold_results.items()}
            wandb_logger.log_metrics(cold_results,epoch=epoch+1, head="test_cold")
    finally:
        print("[TEST]")
        results = trainer.test()
        wandb_logger.log_metrics({**results}, head="test")
        
        print("[TEST ColdItem]")
        cold_results = trainer.test_cold()
        wandb_logger.log_metrics({**cold_results}, head="test_cold_result")

        if args.tensorboard:
            w.close()

        # if args.wandb :
        #     wandb_logger.finish()


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')

    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments (no default value)
    parser.add_argument('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    parser.add_argument('--model', '-m', '--m', type=str, 
        choices=['LightGCN', 'CLCRec'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    parser.add_argument('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    parser.add_argument('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    parser.add_argument('--model_experiment_name', '--men','-men',type=str,
        help='model 저장 이름을 설정할 수 있습니다.')
    parser.add_argument('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    parser.add_argument('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    parser.add_argument('--wandb_experiment_name', '--wen', '-wen', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    parser.add_argument('--tensorboard','--tb','-tb',type=str,
        help='Tensorboard를 사용할 지 선택합니다.')
    parser.add_argument('--model_args', '--ma', '-ma', type=ast.literal_eval)
    parser.add_argument('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    parser.add_argument('--dataset', '--dset', '-dset', type=ast.literal_eval)
    parser.add_argument('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    parser.add_argument('--loss', '-l', '--l', type=str)
    parser.add_argument('--metrics', '-met', '--met', type=ast.literal_eval)
    parser.add_argument('--train', '-t', '--t', type=ast.literal_eval)

    args = parser.parse_args()


    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    config_yaml['model_args'] = config_yaml.model_args[config_yaml.model]

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))
    
    ######################## MAIN
    main(config_yaml)