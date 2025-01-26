from src.lightgcn_utils import world
from src.data import dataloader
from src.models import lightgcn
from pprint import pprint

if world.dataset in ['MovieLens1M', 'MovieLens32M']:
    dataset = dataloader.Loader(path="./data/"+world.dataset+"/final")

print('==================config=======================')
pprint(world.config)
print("device:", world.device)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('==================end=======================')

MODELS = {
    'mf': lightgcn.PureMF,
    'lgn': lightgcn.LightGCN
}