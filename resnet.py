import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from misc import AverageMeter
from random import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_weights = './checkpoints/batchsize64/checkpoint_50perc.pth' # checkpoint to a pretrained model. If not available, set it to False
freeze = True # To speed up the learning process, freeze some layers of the model

class resnet18:

    def __init__(self):
        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)

    def load_model(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 4)
        # Start from checkpoint, if specified
        if pretrained_weights:
            model.load_state_dict(torch.load(pretrained_weights))
            print('pretrained weights loaded!')
        #Freeze layers from 1 to 9, if specified
        if freeze:
            ct = 0
            for child in model.children():
                ct += 1
                if ct < 10: # Freeze 1-9 layers
                    for param in child.parameters():
                        param.requires_grad = False
            print('Freezed layers!')

        #(For GradCam to work) Set the requires_grad to True only for the last convolution
        #for param in model.layer4[1].conv2.parameters():
         #   param.requires_grad = True

        model = model.to(device)
        return model

    def fit(self, known_set, ite, max_ite, n_epochs, checkpoint):
        if checkpoint:
            # load from checkpoints
            self.model.load_state_dict(torch.load("./checkpoints/ckpt_resnet18.pth")) 

        self.model.train()

        known_loader = torch.utils.data.DataLoader(known_set, 
                                                   batch_size=int(np.sqrt(len(known_set))),
                                                   shuffle=True, num_workers=2, pin_memory=True) # DataLoader for known set

        for epoch in range(n_epochs):
            for i, batch in enumerate(known_loader):
                # Training
                self.optimizer.zero_grad()
                loss, loss_info = self.sord_function(self.model, batch, epoch)
               
                loss.backward()
                self.optimizer.step()

                print("\n---- Epoch: [{0}/{1}] batch: [{2}/{3}] iteration: [{4}/{5}] ----\t".format((epoch+1), n_epochs, (i+1), len(known_loader), (ite+1), max_ite))
        # Save checkpoint
        torch.save(self.model.state_dict(),"./checkpoints/ckpt_resnet18.pth")

        return 


    def sord_function(self, model, batch, epoch):

        wide_gap_loss = True # It could be False as well

        img, metadata = batch
    
        img = img.to(device)  
        label_multiclass = metadata["multiclass_label"].type(torch.LongTensor).to(device, non_blocking=True) 
        batch_size = label_multiclass.size(0)
        labels_sord = np.zeros((batch_size, 4))
        for element_idx in range(batch_size):
            current_label = label_multiclass[element_idx].item()
            for class_idx in range(4):
                if wide_gap_loss:
                    wide_label = current_label
                    wide_class_idx = class_idx

                  # Increases the gap between positive and negative
                    if wide_label == 0:
                         wide_label = -0.5
                    if wide_class_idx == 0:
                        wide_class_idx = -0.5

                    labels_sord[element_idx][class_idx] = 2 * abs(wide_label - wide_class_idx) ** 2
                else:
                    labels_sord[element_idx][class_idx] = 2 * abs(current_label - class_idx) ** 2

        labels_sord = torch.from_numpy(labels_sord).to(device, non_blocking=True)
        labels_sord = F.softmax(-labels_sord, dim=1)
        
        class_predictions = model(img)
        log_predictions = F.log_softmax(class_predictions, 1)

        # Computes cross entropy
        loss = (-labels_sord * log_predictions.double()).sum(dim=1).mean()

        return loss, {"loss1": loss.item()}

    def evaluate(self, data_loader, writer, step):

        def _compute_scores(y_true, y_pred):

            folder = "test"

            labels = list(range(4)) # 4 is the number of classes: {0,1,2,3}
            confusion = confusion_matrix(y_true, y_pred, labels=labels)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)

            print(confusion)

            scores = {}
            scores["{}/accuracy".format(folder)] = accuracy
            scores["{}/precision".format(folder)] = precision
            scores["{}/recall".format(folder)] = recall
            scores["{}/f1".format(folder)] = fscore

            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels, average=None)

            for i in range(len(labels)):
                prefix = "{}_{}/".format(folder, i)
                scores[prefix + "precision"] = precision[i]
                scores[prefix + "recall"] = recall[i]
                scores[prefix + "f1"] = fscore[i]

            return scores

        self.model.eval()
        val_loss_meter = AverageMeter('loss1', 'loss2')
        classification_loss_function = torch.nn.CrossEntropyLoss()

        total_samples = 0

        all_predictions = []
        all_labels = []

        total_time = 0

        with torch.no_grad():
            for (img, metadata) in data_loader:

                start = time.time()
                labels_classification = metadata["multiclass_label"].type(torch.LongTensor).to(device)
                total_samples += img.size()[0]

                img = img.to(device)

                class_probabilities = self.model(img)
                class_predictions = torch.argmax(class_probabilities, dim=1).cpu().numpy()
                total_time += time.time() - start

                classification_loss = classification_loss_function(class_probabilities, labels_classification)

                labels_classification = labels_classification.cpu().numpy()

                val_loss_meter.add({'classification_loss': classification_loss.item()})
                all_labels.append(labels_classification)
                all_predictions.append(class_predictions)

        inference_time = total_time / total_samples
        print("Inference time: {}".format(inference_time))

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Computes and logs classification results
        scores = _compute_scores(all_labels, all_predictions)

        avg_classification_loss = val_loss_meter.pop('classification_loss')


        print("- accuracy: {:.3f}".format(scores["test/accuracy"]))
        print("- precision: {:.3f}".format(scores["test/precision"]))
        print("- recall: {:.3f}".format(scores["test/recall"]))
        print("- f1: {:.3f}".format(scores["test/f1"]))
        print("- classification_loss: {:.3f}".format(avg_classification_loss))
        writer.add_scalar("Validation_f1/", scores["test/f1"], step)
        writer.add_scalar("Validation_accuracy/", scores["test/accuracy"], step)
        writer.add_scalar("Validation_precision/", scores["test/precision"], step)
        writer.add_scalar("Validation_classification_loss/", avg_classification_loss, step)

        return  

    def explain(self, grad_cam, explain_loader, writer, step):  
        tot_prec = 0
        tot_rec = 0 
        tot_corr = 0

        for (expl_test, metadata) in explain_loader:

            test_masks = grad_cam(expl_test, Training=False)
            hospitals = metadata['hospital']
            gts_strings = metadata['explanation']
            groundtruth_list = getGroundTruths(hospitals, gts_strings, Training=False)

            black_and_white_expls = process_masks(test_masks)

            correlation, precision, recall = calculate_measures(groundtruth_list, black_and_white_expls)
            tot_corr = tot_corr + correlation
            tot_prec = tot_prec + precision
            tot_rec = tot_rec + recall


        writer.add_scalar("correlation_gradcam", (tot_corr / 418), step) # 418 is the size of the grad_test_set in makeDS.py
        writer.add_scalar("precision_gradcam", (tot_prec / 418), step)
        writer.add_scalar("recall_gradcam", (tot_rec / 418), step)

        return   

    def predict_proba(self, unknown_set):
        self.model.load_state_dict(torch.load("./checkpoints/ckpt_resnet18.pth"))
        self.model.eval()
        unknown_loader = torch.utils.data.DataLoader(unknown_set, 
                                                     batch_size=1,
                                                     shuffle=True, num_workers=2, pin_memory=True)
        probs = torch.zeros([len(unknown_set), 4])
        with torch.no_grad():
            for idx, (img, _) in enumerate(unknown_loader):
                img = img.to(device)
                class_prediction = self.model(img)
                prob = F.softmax(class_prediction, dim=1) 
                probs[idx] = prob.cpu()
        return probs

    def calculate_measures(self, gts, masks):  
        final_corr = 0
        final_prec = 0
        final_rec = 0
        div = len(masks)

        for mask, gt in zip(masks, gts): 

            precision = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum((1-gt)*mask)) 
            final_prec = final_prec + precision

            recall = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum(gt*(1-mask)))
            final_rec = final_rec + recall 

            correlation = (1 / (gt.shape[0]*gt.shape[1])) * np.sum(gt*mask)
            final_corr = final_corr + correlation

        return final_corr/div, final_prec/div, final_rec/div 


        return precision, recall

    def process_masks(masks):
        bw_expls = []

        for mask in masks:
            mask = np.float32(mask*255)
            mask = np.uint8(np.around(mask,decimals=0))
            th, dst = cv2.threshold(mask, 200, 225, cv2.THRESH_BINARY)
            mask = dst / 255
            bw_expls.append(mask)
        
        return bw_expls