from src import preprocessing, world, utils, Procedure, register, wandblogger
from src.data import dataloader
import torch
from tensorboardX import SummaryWriter
import time
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

preprocessing.data2txt() # 해당 경로에 txt가 없는 경우 최초 한번 실행

if world.dataset in ['MovieLens1M', 'MovieLens32M']:
    dataset = dataloader.Loader(path="./data/"+world.dataset+"/final")


Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

wandblogger = wandblogger.WandbLogger(world.config)

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            world.cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information, aver_loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        wandblogger.log_metrics({"train_loss": aver_loss}, head="train", epoch = epoch+1)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)

    world.cprint("[TEST]")
    results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
    wandblogger.log_metrics({**results}, head="test")
finally:
    if world.tensorboard:
        w.close()