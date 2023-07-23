import random
import re
import os
from os import listdir
from os.path import join
import datetime
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from random import Random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomResizedCrop, ColorJitter
import seaborn as sn
import pandas as pd
import shutil
import random
from math import e
import wandb
import zipfile
import subprocess
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from functionNclass.class_collection import VideoDataset, VideoDatasetSourceAndTarget, ClassObservationsSamplerVideoDatasetSourceAndTarget, ClassObservationsSamplerVideoDatasetTarget
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_best_model(h_score, model, config, entropy_val):
    # Get the file path of the saved model
    model_path = config["model_dir"]
    if(config['baseline_or_proposed']=='baseline'):
      model_dir = os.path.join(model_path, 'baseline')
    else:
      model_dir = os.path.join(model_path, 'proposed')

    os.makedirs(model_dir, exist_ok=True)
    model_name = f"model_entropy_{entropy_val:.4f}_hscore_{h_score:.4f}_direction_{config['adaptation_direction']}_{config['baseline_or_proposed']}_seed_{config['seed']}.pth"
    model_path = os.path.join(model_dir, model_name)

    # If there's no saved model yet, save the current model
    if not os.listdir(model_dir):
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with h_score: {h_score}")
        return

    model_list = [m for m in os.listdir(model_dir) if config['adaptation_direction'] and config['baseline_or_proposed'] and str(config['seed']) in m]
    if(model_list==[]):
      torch.save(model.state_dict(), model_path)
      print(f"No previous model saved. Saving model with seed {config['seed']}, h_score {h_score}, entropy {entropy_val:.4f}, direction {config['adaptation_direction']}, and type {config['baseline_or_proposed']}")
      return model_name

    # If there is a saved model, load it and compare the h_scores
    for file in model_list:
        saved_h_score = float(file.split('_')[-1][:-4])
        if h_score > saved_h_score:
            # Save the current model with higher h_score
            torch.save(model.state_dict(), model_path)
            # Remove the saved model with lower h_score
            os.remove(os.path.join(model_dir, file))


            print(f"Previous model with seed {config['seed']}, h_score {saved_h_score}, entropy {entropy_val:.4f}, direction {config['adaptation_direction']}, and type {config['baseline_or_proposed']} replaced with model with h_score: {h_score} nd entropy {entropy_val:.4f} ")
            return model_name
    print(f"Model not saved, h_score: {h_score} is not better than existing model's h_score: {saved_h_score}, entropy {entropy_val:.4f}, direction {config['adaptation_direction']}, and type {config['baseline_or_proposed']}")
    return 'no_model'


def plot_tsne(features, labels, epoch, entropy_val, config, filename, perplexity=30):
    # Adapt perplexity according to the size of the input
    if len(features) <= perplexity:
        perplexity = len(features) - 1
    # Set a seed for the t-SNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=300, random_state=int(config['seed']))
    tsne_results = tsne.fit_transform(features)

    df = pd.DataFrame()
    df['x-tsne'] = tsne_results[:,0]
    df['y-tsne'] = tsne_results[:,1]
    df['labels'] = labels

    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="x-tsne", y="y-tsne",
        hue="labels",
        palette=sns.color_palette("hsv", len(set(labels))),
        data=df,
        legend="full",
        alpha=0.6
    )
    # tsne_name = f'{filename}/tsne_epoch_{epoch}_entropy_{entropy_val}.png'
    tsne_name = f'{filename}/tsne_entropy_{entropy_val}.png'
    plt.title(f't-SNE plot at epoch {epoch} with entropy {entropy_val}')
    plt.savefig(tsne_name)
    # wandb.log({"t-SNE plot": wandb.Image(plt)})
    plt.close()
    wandb.log({"t-SNE plot": wandb.Image(tsne_name)})



def baseline(config, source_n_target_train_loader, target_test_loader, entropy_list, filename, run_id):
  model = config["model"]
  criterion = config["criterion"]
  device = config["device"]
  optimizer = config["optimizer"]
  num_epochs = config["num_epochs"]
  num_classes = config["num_classes"]
  wandb.watch(model)
  for entropy_val in entropy_list:
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_start_time = time.time()

        for X_index, X_source, y_source, x_index, X_target, y_target in source_n_target_train_loader: #should we use the target?
            X_source, y_source = X_source.to(device), y_source.to(device)
            optimizer.zero_grad()
            pred_source = model(X_source)
            loss = criterion(pred_source, y_source)
            loss.backward()
            optimizer.step()
            pred_labels = torch.argmax(pred_source, dim=1)
            # print('pred_labels: ', pred_labels)
            train_loss += loss.item() * X_source.size(0)
            train_total += y_source.size(0)
            train_correct += (pred_labels == y_source).sum().item()

        train_accuracy = train_correct / train_total
        train_time = time.time() - train_start_time
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Train Time": train_time
        })

        # Evaluation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_start_time = time.time()
        correct_per_class = [0 for _ in range(num_classes)]
        instances_per_class = [0 for _ in range(num_classes)]

        with torch.no_grad():
            predicted_all = []
            labels_all = []
            entropy_values = []
            hist, bin_edges = np.histogram(entropy_values, bins=30)


            for batch in target_test_loader: #should we have data from both datasets?
                if config['subset_flag']:
                    index, target_data, target_label = batch
                else:
                    target_data, target_label = batch

                target_data, target_label = target_data.to(device), target_label.to(device)
                pred_target = model(target_data)

                # adding tsne
                # Extract 2D features for t-SNE
                features_2d = pred_target.detach().cpu().numpy()
                labels = target_label.detach().cpu().numpy()
                # Plot t-SNE
                plot_tsne(features_2d, labels, epoch, entropy_val, config, filename)

                probs = F.softmax(pred_target, dim=1)
                entropy = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)
                # print('entropy: ', entropy)
                entropy_values.extend(entropy.tolist())
                pred_labels = torch.argmax(pred_target, dim=1)
                pred_labels_entrop = pred_labels.clone()
                pred_labels_entrop[entropy > entropy_val] = num_classes - 1
                loss = criterion(pred_target, target_label)

                for i in range(len(target_label)):
                    label = target_label[i]
                    predicted_label = pred_labels_entrop[i]
                    instances_per_class[label] += 1
                    if label.item() == predicted_label.item():
                        correct_per_class[label] += 1

                predicted_all.extend(pred_labels_entrop.cpu().tolist())
                labels_all.extend(target_label.cpu().tolist())

            accuracy_per_class = np.divide(np.array(correct_per_class), np.array(instances_per_class), out=np.zeros_like(np.array(correct_per_class), dtype=float), where=np.array(instances_per_class)!=0)
            closed_accuracy = (accuracy_per_class[:num_classes-1].mean())
            open_accuracy = (accuracy_per_class[-1])
            h_score = (2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)) if (closed_accuracy + open_accuracy) > 0 else 0
            val_time = time.time() - val_start_time

            # Save the best model based on h_score
            model_name = save_best_model(h_score, model, config, entropy_val)
            if(model_name!='no_model'):
              model_id = model_name + '_' + run_id
              filename = os.path.join("/content/drive/MyDrive/datasets-thesis/runs", model_id)

            wandb.log({
                'Epoch': epoch,
                "Entropy": entropy_val,
                "Validation Loss": val_loss,
                "Validation Closed Accuracy": closed_accuracy,
                "Validation Open Accuracy": open_accuracy,
                "Validation H Score": h_score,
                "Validation Time": val_time,
                "Model Name" : model_name
            })
            all_classes = sorted(set(int(val) for val in config["target_test_classes"].values()))
            plot_confusion_matrix(labels_all, predicted_all, all_classes, epoch, entropy_val, filename)
            wandb.log({"wandb_confusion_matrix": wandb.Image(f"{filename}/confusion_matrix_entropy_val_{entropy_val}_epoch_{epoch}.png")})

            print("#################### - EVALUATION - ##########################")
            print(f'Entropy: {entropy_val}')
            print(f'Validation loss: {val_loss:.4f}')
            print(f'Validation accuracy_per_class: {accuracy_per_class}')
            print(f'Validation closed_accuracy: {closed_accuracy:.2%}')
            print(f'Validation open_accuracy: {open_accuracy:.2%}')
            print(f'Validation h_score: {h_score:.2%}')
            print(f'Validation time: {val_time:.2f}')
            print(f'PREDICTED LABELS: {predicted_all}')
            print(f'TARGET LABELS   : {labels_all}')

        # Draw and save the entropy histogram
        plt.hist(entropy_values, bins=30)
        plt.title('Histogram of entropy values')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        entropy_histogram_path = f"{filename}/entropy_histogram_epoch_{epoch}.png"
        plt.savefig(entropy_histogram_path)
        plt.close()
        # log the image to Weights & Biases
        wandb.log({"entropy_histogram": wandb.Image(entropy_histogram_path)})

def calculate_new_labels(source_dataset, target_dataset):
    # Combine the source and target dataset into one
    combined_dataset = source_dataset + target_dataset
    
    # Extract the labels
    old_labels = [label for video_path, label in combined_dataset]

    # Get the unique labels
    unique_labels = set(old_labels)

    # Create a mapping from the old labels to the new labels
    new_labels = {old_label: i for i, old_label in enumerate(unique_labels)}

    return new_labels

def calculate_new_labels(source_dir, target_dir, source_txt, target_txt):
    source_classes = set(os.listdir(source_dir))
    target_classes = set(os.listdir(target_dir))

    # Find extra classes in the target dataset
    extra_classes = target_classes - source_classes

    if len(extra_classes) == 0:
        # The source and target datasets have the same classes, so no need to change anything
        print("Classes are identical, no need to update labels.")
        return

    # Load the source dataset labels
    with open(source_txt, 'r') as file:
        source_data = file.readlines()

    # Find the maximum integer label in the source dataset
    max_label = max(int(line.split(' ')[1]) for line in source_data)

    # Assign the unknown label
    unknown_label = max_label + 1

    # Load the target dataset labels
    with open(target_txt, 'r') as file:
        target_data = file.readlines()

    # # Write the new labels to a new .txt file
    # with open('new_source_labels.txt', 'w') as file:
    #     for line in source_data:
    #         path, label = line.strip().split(' ')
    #         if label in extra_classes:
    #             file.write(f'{path} {unknown_label}\n')
    #         else:
    #             file.write(line)

    # Write the new labels to a new .txt file
    with open('new_target_labels.txt', 'w') as file:
        for line in target_data:
            path, label = line.strip().split(' ')
            if label in extra_classes:
                file.write(f'{path} {unknown_label}\n')
            else:
                file.write(line)

    print("Labels have been updated.")

def prepare_datasets(
    source_dataset,
    target_dataset,
    val_dataset,
    source_txt_file_path,
    target_train_txt,
    target_test_txt,
    n_frames=16, #4
    n_clips=1, #4
    frame_size=224,
    normalize=True
):
    
    source_dataset = VideoDataset(
        source_dataset,
        source_txt_file_path,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
        train=True,
        augmentation=True
    )

    target_dataset = VideoDataset(
        target_dataset,
        target_train_txt,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
        train=False,
        augmentation=True
    )
    source_n_target_dataset = VideoDatasetSourceAndTarget(
        source_dataset, target_dataset
    )

    val_dataset = VideoDataset(
        val_dataset,
        target_test_txt,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
        train=False,
        augmentation=False
    )

    return source_n_target_dataset, val_dataset


# # prepares datasets and dataloaders
# def prepare_data(
#     source_dataset,
#     target_dataset,
#     val_dataset,
#     n_frames=16,
#     n_clips=1,
#     frame_size=224,
#     normalize=True,
#     batch_size=8,
#     num_workers=2
# ):
#     source_n_target_dataset, val_dataset = prepare_datasets(
#         source_dataset=source_dataset,
#         target_dataset=target_dataset,
#         val_dataset=val_dataset,
#         n_frames=n_frames,
#         n_clips=n_clips,
#         frame_size=frame_size,
#         normalize=normalize
#     )
#     source_n_target_loader, val_loader = prepare_dataloaders(
#         source_n_target_dataset,
#         val_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#     )
#     return source_n_target_loader, val_loader

# prepares dataloaders given input datasets
def prepare_dataloaders(train_dataset, val_dataset, batch_size=8, num_workers=2):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def get_paths_dataset(config):

  if(config['adaptation_direction']=='ucf2hmdb'): #source -> target
    #source  UCF
    path_source_train = '/content/openset_domain_adaptation/hmdb_ucf/ucf/train'
    path_source_test = '/content/openset_domain_adaptation/hmdb_ucf/ucf/test'
    config['source_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train_source.txt'
    config['source_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test_source.txt'

    #target HMDB
    path_target_train = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/train'
    path_target_test = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/test'
    config['target_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train_target.txt'
    config['target_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test_target.txt'

  else:
    #source HMDB
    path_source_train = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/train'
    path_source_test = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/test'
    config['source_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train_source.txt'
    config['source_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test_source.txt'
    
    #target UCF
    path_target_train = '/content/openset_domain_adaptation/hmdb_ucf/ucf/train'
    path_target_test = '/content/openset_domain_adaptation/hmdb_ucf/ucf/test' 
    config['target_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train_target.txt'
    config['target_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test_target.txt'   



  config['path_source_train'] = path_source_train
  config['path_source_test'] = path_source_test
  config['path_target_train'] = path_target_train
  config['path_target_test'] = path_target_test
  return config
  
def create_datasets(config):

  path_source_train = config['path_source_train']
  path_target_train = config['path_target_train']
  path_target_test = config['path_target_test']
  source_txt = config['source_train_txt'] #need to create a file with all data for source. source_txt_file_path = source_train_txt + source_test_txt_file_path
  target_train_txt = config['target_train_txt']
  target_test_txt =config['target_test_txt']


  source_n_target_train_dataset, target_test_dataset = prepare_datasets(path_source_train,
                                                                      path_target_train,
                                                                      path_target_test, 
                                                                      source_txt,
                                                                      target_train_txt,
                                                                      target_test_txt)

  if(config['g_open_set']==True): 
      num_classes_to_remove = config['num_classes_to_remove']
      source_old_mapping = map_classes_to_labels(path_source_train)
      target_old_mapping = map_classes_to_labels(path_target_train)
      classes_to_remove, new_mapping, unknown_label = select_classes_to_remove_and_create_new_mapping(path_source_train,path_target_train, source_old_mapping, num_classes_to_remove)
      modify_labels_in_datasets(source_txt, target_train_txt, target_test_txt, source_old_mapping, target_old_mapping, new_mapping, classes_to_remove, unknown_label)
      #updating the class with new labels.
      source_n_target_train_dataset, target_test_dataset = prepare_datasets(path_source_train,
                                                                          path_target_train,
                                                                          path_target_test, 
                                                                          source_txt,
                                                                          target_train_txt,
                                                                          target_test_txt)  

  if(config['subset_flag']==True): ########################### ADDING A SAMPLER
    source_n_target_train_dataset = ClassObservationsSamplerVideoDatasetSourceAndTarget(source_n_target_train_dataset, config['obs_num'])
    target_test_dataset = ClassObservationsSamplerVideoDatasetTarget(target_test_dataset, config['obs_num'])
  return source_n_target_train_dataset, target_test_dataset

def classes_validation(source_n_target_train_dataset, target_test_dataset):
  # classes validation
  verification1 = source_n_target_train_dataset.source_dataset.classes==source_n_target_train_dataset.target_dataset.classes==target_test_dataset.classes
  source_classes_verification = list(set(source_n_target_train_dataset.source_dataset[idx][1] for idx in range(0,len(source_n_target_train_dataset.source_dataset))))
  target_train_classes_verification = list(set(source_n_target_train_dataset.target_dataset[idx][1] for idx in range(0,len(source_n_target_train_dataset.target_dataset))))
  target_test_verification = list(set(target_test_dataset[idx][2] for idx in range(0,len(target_test_dataset))))
  verification2 = source_classes_verification==target_train_classes_verification==target_test_verification

  if(config['g_open_set']==False):
    print('CLOSED SET VALIDATION!')
    print('ARE ALL CLASSES EQUAL?')
    if(verification1==True and verification2==True):
      print('ALL GOOD, PROCEED!')
    else:
      raise ValueError("Classes are DIFFERENT for source and target. Not allowed for CLOSE set!") 
  else:
    print('OPEN SET VALIDATION!')
    if(verification1==False and verification2==False):
      print('ALL GOOD, PROCEED!')
    else:
      raise ValueError("Classes are EQUAL for source and target. Not allowed for OPEN set!") 

def train_model(config, source_n_target_train_loader, target_test_loader, entropy_val, filename):

    model = config["model"]
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    num_epochs = config["num_epochs"]

    eval_interval = 5  # Run evaluation every 5 epochs
    wandb.watch(model)

    for epoch in range(num_epochs):
        with open(filename, "a") as f:
          f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Initialize training variables
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_start_time = time.time()
        # Set the model to train mode
        model.train()


        for X_index, X_source, y_source, x_index, X_target, y_target in source_n_target_train_loader: #y_target cannot be used to 
          X_source, y_source = X_source.to(device), y_source.to(device)
          
          # Zero out the optimizer gradients
          optimizer.zero_grad()

          # Forward pass
          pred_source = model(X_source)
          
          # Calculate the cross-entropy loss
          loss = criterion(pred_source, y_source)
          # print(torch.argmax(pred_source, dim=1), y_source)

          # Backward pass
          loss.backward()
          optimizer.step()

          pred_labels = torch.argmax(pred_source, dim=1)
          # Calculate training statistics
          train_loss += loss.item() * X_source.size(0)
          train_total += y_source.size(0)
          train_correct += ( pred_labels== y_source).sum().item()
            
        train_accuracy = train_correct / train_total
        train_time = time.time() - train_start_time

        with open(filename, "a") as f:
          f.write(f"Training loss: {train_loss:.4f}\n")
          f.write(f"Training accuracy: {train_accuracy:.2%}\n")
          f.write(f"Training time: {train_time:.2f}s\n")
          f.write(f"PREDICTED LABELS: {pred_labels}\n")
          f.write(f"SOURCE LABELS   : {y_source}\n")
          
          
        print(f'Training loss: {train_loss:.4f}')
        print(f'Training accuracy: {train_accuracy:.2%}')
        print(f'Training time: {train_time:.2f}s')
        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Train Time": train_time
        })
        

        # Run evaluation only if the current epoch is a multiple of eval_interval
        eval_model(config, target_test_loader, entropy_val, filename, epoch)
        # if (epoch + 1) % eval_interval == 0:
        #   eval_model(config, dataset, target_test_loader, entropy_val, filename)


def eval_model(config, target_test_loader, entropy_val, filename, epoch):

    model = config["model"]
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    num_classes = config["num_classes"]

    # Evaluate the model on the validation dataset
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_start_time = time.time()
        predicted_all = []
        labels_all = []
        # at the beginning, initialize the lists
        correct_per_class = [0 for _ in range(config["num_classes"])] #unk already being counted
        instances_per_class = [0 for _ in range(config["num_classes"])] #unk already being counted

        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        for batch in target_test_loader:
              if config['subset_flag']:
                  index, target_data, target_label = batch
              else:
                  target_data, target_label = batch

              # print('init: ',target_label)
              target_data, target_label = target_data.to(device), target_label.to(device)
              pred_target = model(target_data)

              # Forward pass
              probs = F.softmax(pred_target, dim=1)
              entropy = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)
              pred_labels = torch.argmax(pred_target, dim=1)
              # print('Forward pass: ',target_label)
              # Set label to 'unknown class' if entropy is greater than the threshold
              pred_labels_entrop = pred_labels.clone()
              pred_labels_entrop[entropy > entropy_val] = config["num_classes"]-1  #'unknown_class_id'
              loss = criterion(pred_target, target_label)

              # Calculate validation statistics
              for i in range(len(target_label)):
                  label = target_label[i]
                  predicted_label = pred_labels_entrop[i]
                  instances_per_class[label] += 1
                  # print('label: ', label, 'predicted_label: ', predicted_label, 'instances_per_class: ',instances_per_class)
                  if label.item() == predicted_label.item():
                      correct_per_class[label] += 1

              # # Store predictions and labels for confusion matrix
              predicted_all.extend(pred_labels_entrop.cpu().tolist())
              labels_all.extend(target_label.cpu().tolist())

           


        # at the end compute h score
        # accuracy_per_class = np.divide(np.array(correct_per_class), np.array(instances_per_class), out=np.zeros_like(np.array(correct_per_class)), where=np.array(instances_per_class)!=0)
        accuracy_per_class = np.divide(np.array(correct_per_class), np.array(instances_per_class), out=np.zeros_like(np.array(correct_per_class), dtype=float), where=np.array(instances_per_class)!=0)


        # Let's handle the edge case where all elements in accuracy_per_class are zero
        if np.count_nonzero(accuracy_per_class) > 0:
            closed_accuracy = (accuracy_per_class[:num_classes-1].mean())
            open_accuracy = (accuracy_per_class[-1])
            h_score = (2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy))
        else:
            closed_accuracy = 0
            open_accuracy = 0
            h_score = 0

        # val_loss = val_loss / len(target_test_loader.dataset)
        # val_accuracy = 100 * val_correct / val_total
        val_time = time.time() - val_start_time
        print("#################### - EVALUATION - ##########################")
        print(f'Entropy VAL: {entropy_val}')
        print(f'Entropy: {entropy}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Validation accuracy_per_class: {accuracy_per_class}')
        print(f'Validation closed_accuracy: {closed_accuracy:.2%}')
        print(f'Validation open_accuracy: {open_accuracy:.2%}')
        print(f'Validation h_score: {h_score:.2%}')
        print(f'Validation time: {val_time:.2f}')
        print(f'PREDICTED LABELS: {predicted_all}')
        print(f'TARGET LABELS   : {labels_all}')
        with open(filename, "a") as f:
          f.write(f"################### - EVALUATION - ########################## \n")
          f.write(f"-------------  Entropy VAL: {entropy_val} ------------- \n")
          f.write(f"Entropy: {entropy}\n")
          f.write(f"Validation loss: {val_loss:.4f}\n")
          f.write(f"Validation accuracy_per_class: {accuracy_per_class}\n")
          f.write(f"Validation closed_accuracy: {closed_accuracy:.2%}\n")
          f.write(f"Validation open_accuracy: {open_accuracy:.2%}\n")
          f.write(f"Validation h_score: {h_score:.2%}\n")
          f.write(f"Validation time: {val_time:.2f}\n")
          f.write(f"PREDICTED LABELS: {predicted_all}\n")
          f.write(f"TARGET LABELS   : {labels_all}\n")
          f.write(f"################### - EVALUATION - ########################## \n")

        # Log metrics to wandb
        wandb.log({
            "Entropy VAL": entropy_val,
            "Validation Loss": val_loss,
            "Validation Closed Accuracy": closed_accuracy,
            "Validation Open Accuracy": open_accuracy,
            "Validation H Score": h_score,
            "Validation Time": val_time
        })
        # # Log confusion matrix
        # class_names = [str(i) for i in range(config["num_classes"] - 1)] + ["unknown"]
        # wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        #     preds=predicted_all,
        #     y_true=labels_all,
        #     class_names=class_names
        # )})
        # class_names = [str(i) for i in range(config["num_classes"] - 1)] + ["unknown"]
        # if(epoch==config['num_epochs']):
        all_classes = sorted(set(int(val) for val in config["target_test_classes"].values()))
        plot_confusion_matrix(labels_all, predicted_all, all_classes, epoch, entropy_val)
        print("#################### - EVALUATION - ##########################")



def plot_confusion_matrix(labels_all, predicted_all, all_classes, epoch, entropy_val, filename):
    cm = confusion_matrix(labels_all, predicted_all, labels=all_classes)
    df_cm = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix, Entropy: {}'.format(entropy_val))
    cm_name = f'{filename}/confusion_matrix_entropy_val_{entropy_val}_epoch_{epoch}.png'
    plt.savefig(cm_name)
    plt.close()
    wandb.log({"confusion_matrix": wandb.Image(cm_name)})


# def plot_confusion_matrix(y_true, y_pred, class_names):
#     cm = confusion_matrix(y_true, y_pred)
#     df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
#     plt.figure(figsize=(10,7))
#     sn.heatmap(df_cm, annot=True, fmt='d')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')


# def are_classes_identical(dir1, dir2):
#     classes_dir1 = set(os.listdir(dir1))
#     classes_dir2 = set(os.listdir(dir2))

#     return classes_dir1 == classes_dir2


# def remove_random_classes_from_source(source_train_dir, hmdb_train_dir, num_classes_to_remove):
#     # Check if classes in both directories are identical
#     if not are_classes_identical(source_train_dir, hmdb_train_dir):
#         print("Classes are not identical, skipping deletion.")
#         return []

#     # Get list of all classes
#     all_classes = os.listdir(source_train_dir)

#     # Randomly select certain classes to remove
#     classes_to_remove = random.sample(all_classes, num_classes_to_remove)

#     # Prepare class to label mappings
#     class_label_map = {class_name: i for i, class_name in enumerate(all_classes)}
#     unknown_class_label = max(class_label_map.values()) + 1
#     old_new_label_map = {class_name: (class_label_map[class_name], unknown_class_label if class_name in classes_to_remove else class_label_map[class_name]) for class_name in all_classes}

#     # Update labels in the txt files for source and target datasets
#     txt_files = ['/content/ucf_train_source.txt', '/content/hmdb_test_target.txt']
#     for txt_file_path in txt_files:
#         with open(txt_file_path, 'r') as f:
#             lines = f.readlines()
#         new_lines = []
#         for line in lines:
#             class_name, path_to_observation, label = line.strip().split()
#             old_label, new_label = old_new_label_map[class_name]
#             if int(label) == old_label:
#                 new_lines.append(f'{class_name} {path_to_observation} {new_label}\n')
#             else:
#                 print(f'Old and new labels do not match for class {class_name}. Label in file: {label}, Old label: {old_label}, New label: {new_label}')

#         print(f'New lines for file {txt_file_path}: {new_lines}')
        
#         # Write new lines back to file
#         txt_mod_path = txt_file_path.replace('.txt', '_mod.txt')
#         with open(txt_mod_path, 'w') as f:
#             f.writelines(new_lines)

#     # Remove classes from the source directory
#     for class_name in classes_to_remove:
#         class_dir_train = os.path.join(source_train_dir, class_name)
#         if os.path.exists(class_dir_train):
#             shutil.rmtree(class_dir_train)

#     return classes_to_remove


# def update_class_labels(source_and_target_dataset, target_test_dataset):
#     """
#     Update class labels in source, target and validation datasets.

#     Args:
#         source_and_target_dataset (Dataset): The dataset containing both source and target datasets.
#         target_test_dataset (Dataset): The target validation dataset.
#     """

#     # Get the sorted class names of the source dataset
#     source_classes = sorted(list(source_and_target_dataset.source_dataset.classes.keys()))

#     # Create a new mapping for the source dataset
#     source_mapping = {class_name: str(i) for i, class_name in enumerate(source_classes)}

#     # Update the classes in the source dataset
#     source_and_target_dataset.source_dataset.classes = source_mapping

#     # Get the sorted class names of the target datasets
#     target_classes = sorted(list(source_and_target_dataset.target_dataset.classes.keys()))
#     target_test_classes = sorted(list(target_test_dataset.classes.keys()))

#     # Initialize the target mappings with the source mapping
#     target_mapping = source_mapping.copy()
#     target_test_mapping = source_mapping.copy()

#     # The label for the classes that are not in the source dataset
#     unknown_class_label = str(max(map(int, source_mapping.values())) + 1)

#     # Update the target mappings with the classes that only exist in the target datasets
#     for class_name in target_classes:
#         if class_name not in target_mapping:
#             target_mapping[class_name] = unknown_class_label

#     for class_name in target_test_classes:
#         if class_name not in target_test_mapping:
#             target_test_mapping[class_name] = unknown_class_label

#     # Update the classes in the target datasets
#     source_and_target_dataset.target_dataset.classes = target_mapping
#     target_test_dataset.classes = target_test_mapping

#     # Update the video_label in the datasets
#     for video in source_and_target_dataset.source_dataset.video_label:
#         video[1] = source_and_target_dataset.source_dataset.classes[source_and_target_dataset.source_dataset.class_id_to_name[video[1]]]

#     for video in source_and_target_dataset.target_dataset.video_label:
#         video[1] = source_and_target_dataset.target_dataset.classes[source_and_target_dataset.target_dataset.class_id_to_name[video[1]]]

#     for video in target_test_dataset.video_label:
#         video[1] = target_test_dataset.classes[target_test_dataset.class_id_to_name[video[1]]]

def get_classes_from_dir(dir_path):
    return sorted(os.listdir(dir_path))


def select_classes_to_remove_and_create_new_mapping(source_train_dir, target_train_dir, old_mapping, num_classes_to_remove):
    all_classes_source = get_classes_from_dir(source_train_dir)
    all_classes_target = get_classes_from_dir(target_train_dir)

    # If classes in source and target are different, it's already an open set
    if set(all_classes_source) != set(all_classes_target):
        classes_to_remove = list(set(all_classes_target) - set(all_classes_source))
    # Otherwise, we're transitioning from a closed set to an open set
    else:
        classes_to_remove = random.sample(all_classes_source, num_classes_to_remove)

    classes_to_keep = [class_name for class_name in all_classes_source if class_name not in classes_to_remove]
    new_mapping = {class_name: i for i, class_name in enumerate(classes_to_keep)}
    unknown_label = max(new_mapping.values()) + 1

    print("Summary of changes:")
    print("Label for unknown classes: ", unknown_label)

    # Detailed label changes for each class
    print("\n###### SOURCE ######")
    print("Old mapping: ", old_mapping)
    print("New mapping: ", new_mapping)
    print("Classes removed: ", classes_to_remove)

    print("Classes that had changes:")
    for class_name in all_classes_source:
        old_label = old_mapping[class_name]
        if class_name in new_mapping:
            new_label = new_mapping[class_name]
            print(f"[{class_name}, {old_label} -> {new_label}]")

    return classes_to_remove, new_mapping, unknown_label




# def select_classes_to_remove_and_create_new_mapping(source_train_dir, old_mapping, num_classes_to_remove):
#     all_classes = sorted(os.listdir(source_train_dir))
#     classes_to_remove = random.sample(all_classes, num_classes_to_remove)
#     classes_to_keep = [class_name for class_name in all_classes if class_name not in classes_to_remove]
#     new_mapping = {class_name: i for i, class_name in enumerate(classes_to_keep)}
#     unknown_label = max(new_mapping.values()) + 1

#     print("Summary of changes:")
#     print("Label for unknown classes: ", unknown_label)

#     # Detailed label changes for each class
#     print("\n###### SOURCE ######")
#     print("Old mapping: ", old_mapping)
#     print("New mapping: ", new_mapping)
#     print("Classes removed: ", classes_to_remove)

#     print("Classes that had changes:")
#     for class_name in all_classes:
#         old_label = old_mapping[class_name]
#         if class_name in new_mapping:
#             new_label = new_mapping[class_name]
#             print(f"[{class_name}, {old_label} -> {new_label}]")

#     return classes_to_remove, new_mapping, unknown_label

#go back to this if something goes wrong!
# def modify_labels_in_datasets(source_txt_path, target_txt_path, source_old_mapping, target_old_mapping, new_mapping, classes_to_remove, unknown_label):
#     # Modify labels in source text file
#     mod_lines_source = modify_labels_in_file(source_txt_path, source_old_mapping, new_mapping, unknown_label, classes_to_remove, source=True)
#     mod_source_txt_path = source_txt_path.replace('.txt', '_mod.txt')
#     with open(mod_source_txt_path , 'w') as f:
#         f.writelines(mod_lines_source)

#     print("\n###### TARGET ######")
#     print("Old mapping: ", target_old_mapping)
#     # Include classes mapped to the unknown label in the new mapping
#     new_mapping_with_unknown = new_mapping.copy()
#     for class_name in classes_to_remove:
#         new_mapping_with_unknown[class_name] = unknown_label
#     print("New mapping: ", new_mapping_with_unknown)

#     print("Classes that had changes:")
#     for class_name in sorted(target_old_mapping.keys()):
#         old_label = target_old_mapping[class_name]
#         new_label = new_mapping_with_unknown[class_name]
#         print(f"[{class_name}, {old_label} -> {new_label}]")

#     # Modify labels in target text file
#     mod_lines_target = modify_labels_in_file(target_txt_path, target_old_mapping, new_mapping, unknown_label, classes_to_remove, source=False)
#     mod_target_txt_path = target_txt_path.replace('.txt', '_mod.txt')
#     with open(mod_target_txt_path , 'w') as f:
#         f.writelines(mod_lines_target)

def map_classes_to_labels(dir_path):
    classes = sorted(os.listdir(dir_path))
    return {class_name: i for i, class_name in enumerate(classes)}

def modify_labels_in_file(file_path, old_mapping, new_mapping, unknown_label, classes_to_remove, source=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    mod_lines = []
    for line in lines:
        class_name, path, old_label = line.strip().split()
        old_label = int(old_label)

        if source:
            # Skip lines for classes to remove in source dataset
            if class_name in classes_to_remove:
                continue
        else:
            # Assign unknown label to classes not in the new mapping in target dataset
            if class_name not in new_mapping:
                new_label = unknown_label
                mod_line = f"{class_name} {path} {new_label}\n"
                mod_lines.append(mod_line)
                continue

        # Confirm that the old label in the file matches the old label in the mapping
        if old_mapping[class_name] != old_label:
            print(f"Old labels do not match for class {class_name}. Label in file: {old_label}, Old label in mapping: {old_mapping[class_name]}")
            continue

        # Assign new label
        if class_name in new_mapping:
            new_label = new_mapping[class_name]
        else:
            new_label = unknown_label

        mod_line = f"{class_name} {path} {new_label}\n"
        mod_lines.append(mod_line)

    return mod_lines



def modify_labels_in_datasets(source_txt, target_train_txt, target_test_txt, source_old_mapping, target_old_mapping, new_mapping, classes_to_remove, unknown_label):
    # Modify labels in source text file
    mod_lines_source = modify_labels_in_file(source_txt, source_old_mapping, new_mapping, unknown_label, classes_to_remove, source=True)
    mod_source_txt_path = source_txt.replace('.txt', '_mod.txt')
    with open(mod_source_txt_path , 'w') as f:
        f.writelines(mod_lines_source)

    print("\n###### TARGET ######")
    print("Old mapping: ", target_old_mapping)
    # Include classes mapped to the unknown label in the new mapping
    new_mapping_with_unknown = new_mapping.copy()
    for class_name in classes_to_remove:
        new_mapping_with_unknown[class_name] = unknown_label
    print("New mapping: ", new_mapping_with_unknown)

    print("Classes that had changes:")
    known_classes = []
    unknown_classes = []
    for class_name in sorted(target_old_mapping.keys(), key=lambda x: (new_mapping_with_unknown.get(x, float('inf')) == unknown_label, target_old_mapping[x])):
        old_label = target_old_mapping[class_name]
        new_label = new_mapping_with_unknown[class_name]
        if new_label == unknown_label:
            unknown_classes.append(f"[{class_name}, {old_label} -> {new_label}]")
        else:
            known_classes.append(f"[{class_name}, {old_label} -> {new_label}]")

    for class_entry in known_classes:
        print(class_entry)

    print("---------------- UNKNOWN CLASSES ----------------")
    for class_entry in unknown_classes:
        print(class_entry)

    # Modify labels in target text file
    mod_lines_target = modify_labels_in_file(target_train_txt, target_old_mapping, new_mapping, unknown_label, classes_to_remove, source=False)
    mod_target_txt_path = target_train_txt.replace('.txt', '_mod.txt')
    with open(mod_target_txt_path , 'w') as f:
        f.writelines(mod_lines_target)
        
    mod_lines_target = modify_labels_in_file(target_test_txt, target_old_mapping, new_mapping, unknown_label, classes_to_remove, source=False)
    mod_target_txt_path = target_test_txt.replace('.txt', '_mod.txt')
    with open(mod_target_txt_path , 'w') as f:
        f.writelines(mod_lines_target)


def create_temp_file_with_new_labels(path, new_labels):
    # create a temporary file in the same directory as the original file
    base_dir = os.path.dirname(path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=base_dir)
    
    with open(path, 'r') as original_file, open(temp_file.name, 'w') as new_file:
        for line in original_file:
            line_split = line.strip().split()
            new_label = new_labels.get(line_split[0])
            if new_label is not None:
                # replace the label in the line
                line_split[-1] = new_label
            # write the line to the new file
            new_file.write(' '.join(line_split) + '\n')

    return temp_file.name

def get_new_labels(source_mapping, target_mapping, source_validation_mapping, target_validation_mapping):
    # Start with the source classes and their IDs
    new_labels = source_mapping.copy()

    # Create a mapping for new target classes not in the source dataset
    for target_class in target_mapping:
        if target_class not in source_mapping:
            new_labels[target_class] = str(max(map(int, source_mapping.values())) + 1)
            source_mapping[target_class] = new_labels[target_class]

    # For validation set, assign new labels for classes not in source or target training sets
    for validation_class in target_validation_mapping:
        if validation_class not in source_mapping and validation_class not in target_mapping:
            new_labels[validation_class] = str(max(map(int, new_labels.values())) + 1)
            
    return new_labels


# def open_set_loss(outputs, labels, num_classes, device):
#     # Create a tensor of shape (batch_size, num_classes) with all zeros
#     # This will be the target tensor for the source data
#     source_targets = torch.zeros_like(outputs[:, :num_classes])

#     # Create a tensor of shape (batch_size, 1) with all ones
#     # This will be the target tensor for the target data
#     target_targets = torch.ones_like(outputs[:, num_classes:])

#     # Concatenate the source and target targets tensors along the second dimension
#     targets = torch.cat([source_targets, target_targets], dim=1)

#     # Move the targets tensor to the device
#     targets = targets.to(device)

#     # Calculate the cross-entropy loss between the outputs and targets tensors
#     loss = nn.CrossEntropyLoss()(outputs, labels)

#     return loss

def apply_data_augmentation(img):
    # Define the data augmentation transformations
    augmentations = [
        RandomHorizontalFlip(p=0.5),
        RandomCrop(size=(224, 224), padding=4),
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
    ]

    # Apply the transformations to the input image
    for transform in augmentations:
        img = transform(img)

    return img

# def get_frames_by_class(dataset_path, n_frames):
#     classes = {}
#     video_label = []
#     classes_root = sorted(os.listdir(dataset_path))
#     for i, class_name in enumerate(classes_root):
#         observation_folder = sorted(os.listdir(os.path.join(dataset_path, class_name)))
#         classes[i] = class_name
#         for obs in observation_folder:
#             frames = os.listdir(os.path.join(dataset_path,class_name,obs))
#             if len(frames) >= n_frames:
#               class_obs = class_name + "/" + obs
#               video_label.append((class_obs, i))
#     return video_label, classes

def get_frames_by_class(txt_file_path, n_frames):
  # if(train):
  #   # p = '/content/drive/MyDrive/datasets-thesis/ucf_train_source.txt'
  #   p = '/content/ucf_train_source.txt'
  # else:
  #   p = '/content/hmdb_test_target.txt'
  #   # p = '/content/drive/MyDrive/datasets-thesis/hmdb_test_target.txt'
  p = txt_file_path

  video_label = []
  classes = {}
  # Open the text file for reading
  with open(p, 'r') as file:
      # Iterate over each line in the file
      for line in file:
          # Strip any newline characters from the line
          line_split = line.strip().split()
          # Append the line to the self.video_label and classes list
          classes[line_split[0]]=line_split[-1]
          video_label.append(tuple(line_split[1:]))
  # create the inverse mapping
  class_id_to_name = {v: k for k, v in classes.items()}

  return video_label, classes, class_id_to_name

# def get_frames_by_class(train, n_frames, new_labels):
#     if(train):
#         p = '/content/ucf_train_source.txt'
#     else:
#         p = '/content/hmdb_test_target.txt'

#     # create a temporary file with new labels
#     temp_file_path = create_temp_file_with_new_labels(p, new_labels)

#     video_label = []
#     classes = {}
#     with open(temp_file_path, 'r') as file:
#         for line in file:
#             line_split = line.strip().split()
#             classes[line_split[0]] = line_split[-1]
#             video_label.append(tuple(line_split[1:]))
#     return video_label, classes
def changeTXT(root = '/content/openset_domain_adaptation/paths', destination = '/content', pattern_search="/content/hmdb_ucf_giacomo/data", pattern_out="/content/openset_domain_adaptation/hmdb_ucf"):  
  for file_name in [file for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]:
    file_path = os.path.join(root, file_name)
    with open(file_path, "r") as f:
        lines = f.readlines()

    file_path = os.path.join(destination, file_name)
    with open(file_path, "w") as f:
      
      for line in lines:
          f.write(line.replace(pattern_search, pattern_out))

# def push(message):
#   os.system('git config --global user.email "sander.martinsgoncalves@gmail.com"')
#   os.system('git config --global user.name "sangoncalves"')
#   os.system('git remote set-url origin https://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@github.com/sangoncalves/openset_domain_adaptation.git')
#   os.system('git add /content/openset_domain_adaptation/')
#   result = subprocess.run(['git', 'status'], stdout=subprocess.PIPE)
#   print(result.stdout.decode('utf-8'))
#   os.system(f'git commit -m "{message}"')
#   print('#######################################')
#   result = subprocess.run(['git', 'status'], stdout=subprocess.PIPE)
#   print(result.stdout.decode('utf-8'))
#   os.system('git push origin main')
#   print('########## GIT PUSH COMPLETE ##########')
#   result = subprocess.run(['git', 'status'], stdout=subprocess.PIPE)
#   print(result.stdout.decode('utf-8'))


         
def get_txt_classes(txt_list, txt_folder):
  df = pd.DataFrame(columns=['txt_name', 'classes'])
  for f in txt_list:
    with open(os.path.join(txt_folder,f), 'r') as f_txt:
      lines = f_txt.readlines()
      df_classes = pd.DataFrame([{'txt_name': f, 'classes': sorted(list(set([l.split()[-1] for l in lines])), key=int)}])
      df = pd.concat([df, df_classes] , ignore_index=True)
  return df

txt_folder ='/content/openset_domain_adaptation/paths' 
txt_list = [f for f in os.listdir(txt_folder) if os.path.isfile(os.path.join(txt_folder,f))]
# df = get_txt_classes(txt_list, txt_folder)
