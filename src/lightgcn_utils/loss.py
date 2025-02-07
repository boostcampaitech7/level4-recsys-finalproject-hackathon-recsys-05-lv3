import torch
import torch.optim as optim


class BPRLoss:
    def __init__(self, recmodel, args):
        self.config = args.optimizer
        self.model = recmodel
        self.weight_decay = self.config.args["weight_decay"]
        self.lr = self.config.args["lr"]
        optimizer_class = getattr(optim, self.config.type)
        self.opt = optimizer_class(self.model.parameters(), lr=self.lr)
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

        self.args = args

    def predict(self, users, pos, neg):

        if self.args.model == "CLCRec":
            loss, reg_loss = self.model.clc_loss(users, pos, neg)
        else:
            loss, reg_loss = self.model.bpr_loss(users, pos, neg)

        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
