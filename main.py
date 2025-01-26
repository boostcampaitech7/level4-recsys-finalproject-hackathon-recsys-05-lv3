from src import preprocessing, world, utils, Procedure, register
import torch
from tensorboardX import SummaryWriter
import time
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

preprocessing.data2txt() # 해당 경로에 txt가 없는 경우 최초 한번 실행

Recmodel = register.MODELS[world.model_name](world.config, register.dataset)
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


try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            world.cprint("[TEST]")
            Procedure.Test(register.dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(register.dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)

    world.cprint("[TEST]")
    Procedure.Test(register.dataset, Recmodel, epoch, w, world.config['multicore'])
finally:
    if world.tensorboard:
        w.close()