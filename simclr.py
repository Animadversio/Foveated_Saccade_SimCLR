"""This is the core traininig loop of simclr"""
import logging
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import numpy as np
from utils import accuracy, save_checkpoint, save_config_file
from linear_eval import evaluation

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir_tmp = os.path.join(self.args.log_root, self.args.run_label + "_" + current_time + "-%03d"%np.random.randint(1000))
        if os.path.isdir(log_dir_tmp):
            log_dir_tmp += "-%03d"%np.random.randint(1000)
        self.writer = SummaryWriter(log_dir=log_dir_tmp, )
        # self.writer.log_dir = self.args.log_root + self.writer.log_dir
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # labels is a matrix of shape (batch * n_views, batch * n_views); 
        # It's (batch, batch) identity matrix blocks arranged in (n_views, n_views) fashion. 
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) # cosine similarity
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)  # (batch * n_views, batch * n_views - 1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (batch * n_views, batch * n_views - 1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (batch * n_views, n_views - 1), here it's (batch * n_views, 1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (batch * n_views, (batch - 1) * n_views)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  
        # after rearrangement, only first entry is positive, so label is int(0); This used the assumption of `n_views=2` or the objective is wrong. 

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        final_train_loss, final_train_acc, final_test_loss, final_test_acc = evaluation(self.model.backbone, self.args, \
                        logistic_batch_size=256, logistic_epochs=500, print_every_epoch=100)
        logging.debug(f"Random Initialization \t After Logistic Fitting, Train Loss: {final_train_loss}\tTop1 accuracy: {final_train_acc}")
        logging.debug(f"Random Initialization \t After Logistic Fitting, Test  Loss: {final_test_loss}\tTop1 accuracy: {final_test_acc}")
        self.writer.add_scalar('eval/train_loss', final_train_loss, global_step=-1)
        self.writer.add_scalar('eval/train_acc', final_train_acc, global_step=-1)
        self.writer.add_scalar('eval/test_loss', final_test_loss, global_step=-1)
        self.writer.add_scalar('eval/test_acc', final_test_acc, global_step=-1)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader): 
                # Train loader have performed the augmentation; it returns a list of different augmented images. 
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('epoch', epoch_counter, global_step=n_iter)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            
            if epoch_counter % self.args.ckpt_every_n_epocs == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                            'args': self.args,
                            'arch': self.args.arch,
                            'state_dict': self.model.backbone.state_dict(), # containing the repr and the projections 
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                
                final_train_loss, final_train_acc, final_test_loss, final_test_acc = evaluation(self.model.backbone, self.args, \
                        logistic_batch_size=256, logistic_epochs=500, print_every_epoch=100)
                
                logging.debug(f"Epoch: {epoch_counter}\t After Logistic Fitting, Train Loss: {final_train_loss}\tTop1 accuracy: {final_train_acc}")
                logging.debug(f"Epoch: {epoch_counter}\t After Logistic Fitting, Test  Loss: {final_test_loss}\tTop1 accuracy: {final_test_acc}")
                self.writer.add_scalar('eval/train_loss', final_train_loss, global_step=n_iter)
                self.writer.add_scalar('eval/train_acc', final_train_acc, global_step=n_iter)
                self.writer.add_scalar('eval/test_loss', final_test_loss, global_step=n_iter)
                self.writer.add_scalar('eval/test_acc', final_test_acc, global_step=n_iter)


        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'args': self.args,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
        final_train_loss, final_train_acc, final_test_loss, final_test_acc = evaluation(self.model.backbone, self.args, \
                        logistic_batch_size=256, logistic_epochs=500, print_every_epoch=100)
        logging.debug(f"Final Evaluation \t After Logistic Fitting, Train Loss: {final_train_loss}\tTop1 accuracy: {final_train_acc}")
        logging.debug(f"Final Evaluation \t After Logistic Fitting, Test  Loss: {final_test_loss}\tTop1 accuracy: {final_test_acc}")
        self.writer.add_scalar('eval/train_loss', final_train_loss, global_step=n_iter)
        self.writer.add_scalar('eval/train_acc', final_train_acc, global_step=n_iter)
        self.writer.add_scalar('eval/test_loss', final_test_loss, global_step=n_iter)
        self.writer.add_scalar('eval/test_acc', final_test_acc, global_step=n_iter)
        print("Finished pipeline")
