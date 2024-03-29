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
from functionNclass.class_collection import VideoDataset_frame_analysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json


def create_datasets_frame_analysis_one_dataset(config, n_frames=16, frame_strategy="uniform"):
    
    path_train = config['path_source_train']
    path_test = config['path_source_test']
    source_txt = config['source_train_txt']
    test_txt =config['source_test_txt']

    # Create the datasets
    train_data, test_data = prepare_datasets_frame_analysis_one_dataset(
        path_train,
        path_test, 
        source_txt,
        test_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    if(config['subset_flag']==True): ########################### ADDING A SAMPLER
      train_data = ClassObservationsSamplerVideoDatasetSourceAndTarget(train_data, config['obs_num'])
      test_data = ClassObservationsSamplerVideoDatasetTarget(test_data, config['obs_num'])
      
    return train_data, test_data


def prepare_datasets_frame_analysis_one_dataset(path_train,  path_test,
                                    source_txt,  test_txt,
                                    n_frames=16, frame_strategy="uniform"):
    
    # Initialize the source and target datasets using the VideoDataset_frame_analysis class
    train_data = VideoDataset_frame_analysis(
        dataset_path=path_train,
        txt_file_path=source_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    
    test_data = VideoDataset_frame_analysis(
        dataset_path=path_test,
        txt_file_path=test_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    return train_data, test_data    



def load_or_create_class_to_label_map(base_path, output_path):
    mapping_file_path = os.path.join(output_path, 'class_to_label_map.json')

    # Check if the mapping file already exists
    if os.path.isfile(mapping_file_path):
        # Load the existing mapping
        with open(mapping_file_path, 'r') as file:
            return json.load(file)
    else:
        # Generate a new mapping
        unique_classes = set(os.listdir(base_path))
        class_to_label_map = {class_name: i for i, class_name in enumerate(sorted(unique_classes))}

        # Save the new mapping
        with open(mapping_file_path, 'w') as file:
            json.dump(class_to_label_map, file)

        return class_to_label_map

def generate_dataset_labels(base_path, output_path):
    # Extract dataset name and category from the path
    parts = base_path.split('/')
    dataset_name = parts[-2] # Extract dataset name
    category = 'train' if 'train' in base_path else 'test'

    # Load or create the class-to-label map
    class_to_label_map = load_or_create_class_to_label_map(base_path, output_path)

    # Output file name
    output_file_name = f"{dataset_name}_{category}.txt"

    # Full path for the output file
    full_output_path = os.path.join(output_path, output_file_name)

    # Open the output file
    with open(full_output_path, 'w') as file:
        # Iterate through each class folder
        for class_folder in os.listdir(base_path):
            class_folder_path = os.path.join(base_path, class_folder)
            if os.path.isdir(class_folder_path):
                # Check if class is in the map
                if class_folder not in class_to_label_map:
                    print(f"Warning: Class '{class_folder}' not found in the class-to-label map. Skipping.")
                    continue  # Skip this class

                # Get the class label from the mapping
                class_label = class_to_label_map[class_folder]

                # Iterate through each observation in the class
                for observation_folder in os.listdir(class_folder_path):
                    # Writing the pattern "class_folder/observation_folder class_label"
                     file.write(f"{class_folder} {base_path}/{class_folder}/{observation_folder} {class_label}\n")

    return full_output_path


# generate_dataset_labels('/content/openset_domain_adaptation/hmdb_ucf/ucf/test','/content/drive/MyDrive/datasets-thesis/path_datasets')
# generate_dataset_labels('/content/openset_domain_adaptation/hmdb_ucf/ucf/train','/content/drive/MyDrive/datasets-thesis/path_datasets')
# generate_dataset_labels('/content/openset_domain_adaptation/hmdb_ucf/hmdb/test','/content/drive/MyDrive/datasets-thesis/path_datasets')
# generate_dataset_labels('/content/openset_domain_adaptation/hmdb_ucf/hmdb/train','/content/drive/MyDrive/datasets-thesis/path_datasets')





def create_datasets_frame_analysis(config, n_frames=16, frame_strategy="uniform"):
    
    path_source_train = config['path_source_train']
    path_target_train = config['path_target_train']
    path_target_test = config['path_target_test']
    source_txt = config['source_train_txt']
    target_train_txt = config['target_train_txt']
    target_test_txt =config['target_test_txt']

    # Create the datasets
    source_n_target_train_dataset, target_test_dataset = prepare_datasets_frame_analysis(
        path_source_train,
        path_target_train,
        path_target_test, 
        source_txt,
        target_train_txt,
        target_test_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    if(config['g_open_set'] == True): 
      source_classes = source_n_target_train_dataset.source_dataset.classes
      print('source_classes: ',type(source_classes), source_classes)
      num_classes_to_remove = config['num_classes_to_remove']
      fake_source_label_or_remove_class = config['fake_source_label_or_remove_class']
      source_old_mapping = map_classes_to_labels(path_source_train)
      target_old_mapping = map_classes_to_labels(path_target_train)
      classes_to_remove, new_mapping, unknown_label = select_classes_to_remove_and_create_new_mapping(
          list(source_classes.keys()), 
          path_source_train,
          path_target_train,
          source_old_mapping,
          num_classes_to_remove,
          fake_source_label_or_remove_class
      )
      modify_labels_in_datasets(source_txt, target_train_txt, target_test_txt, source_old_mapping, target_old_mapping, new_mapping, unknown_label)
      #updating the class with new labels.
      source_n_target_train_dataset, target_test_dataset = prepare_datasets_frame_analysis(path_source_train,
                                                                          path_target_train,
                                                                          path_target_test, 
                                                                          source_txt,
                                                                          target_train_txt,
                                                                          target_test_txt, fake_label=False, n_frames=n_frames)  
      source_n_target_train_dataset.unknown_label = unknown_label
      target_test_dataset.unknown_label = unknown_label

    if(config['subset_flag']==True): ########################### ADDING A SAMPLER
      source_n_target_train_dataset = ClassObservationsSamplerVideoDatasetSourceAndTarget(source_n_target_train_dataset, config['obs_num'])
      target_test_dataset = ClassObservationsSamplerVideoDatasetTarget(target_test_dataset, config['obs_num'])
      
    return source_n_target_train_dataset, target_test_dataset


def prepare_datasets_frame_analysis(path_source_train, path_target_train, path_target_test,
                                    source_txt, target_train_txt, target_test_txt,
                                    n_frames=16, frame_strategy="uniform"):
    
    # Initialize the source and target datasets using the VideoDataset_frame_analysis class
    source_dataset = VideoDataset_frame_analysis(
        dataset_path=path_source_train,
        txt_file_path=source_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    target_train_dataset = VideoDataset_frame_analysis(
        dataset_path=path_target_train,
        txt_file_path=target_train_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )
    
    target_test_dataset = VideoDataset_frame_analysis(
        dataset_path=path_target_test,
        txt_file_path=target_test_txt,
        n_frames=n_frames,
        frame_strategy=frame_strategy  # New parameter for frame strategy
    )

    # Combine source and target train datasets
    source_n_target_train_dataset = VideoDatasetSourceAndTarget(source_dataset, target_train_dataset)
    
    return source_n_target_train_dataset, target_test_dataset    





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_best_model(h_score, model, config, entropy_val, epoch):
    model_path = config["model_dir"]
    if config['baseline_or_proposed'] == 'baseline':
        model_dir = os.path.join(model_path, 'baseline')
        model_name = f"model_entropy_{entropy_val:.4f}_hscore_{h_score:.4f}_direction_{config['adaptation_direction']}_{config['baseline_or_proposed']}_seed_{config['seed']}_epoch_{str(epoch)}.pth"
        current_run_dir = os.path.join(model_dir, f"{config['adaptation_direction']}_seed_{config['seed']}_entropy_{entropy_val}")
    else:
        model_dir = os.path.join(model_path, 'proposed')
        model_name = f"model_name_{entropy_val}_hscore_{h_score:.4f}_direction_{config['adaptation_direction']}_{config['baseline_or_proposed']}_seed_{config['seed']}_epoch_{str(epoch)}.pth"
        current_run_dir = os.path.join(model_dir, f"{config['adaptation_direction']}_seed_{config['seed']}_model-name_{entropy_val}")

    current_run_tmp_dir = current_run_dir + f"_tmp"
    
    # If it's the beginning of the training (epoch 0), and the temporary directory already exists, remove its contents
    if epoch == 0 and os.path.exists(current_run_tmp_dir):
        shutil.rmtree(current_run_tmp_dir)
        os.makedirs(current_run_tmp_dir)  # Recreate the directory after removing it

    os.makedirs(current_run_tmp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_tmp_path = os.path.join(current_run_tmp_dir, model_name)
    torch.save(model.state_dict(), model_tmp_path)


    if epoch == config["num_epochs"] - 1:
        saved_model_list = [m for m in os.listdir(current_run_dir) if m.endswith(".pth")]
        saved_model_list_tmp = [m for m in os.listdir(current_run_tmp_dir) if m.endswith(".pth")]
        if config['baseline_or_proposed'] == 'baseline':
          max_h_score_saved = max([float(m.split('_')[4]) for m in saved_model_list]) if saved_model_list else -1
          max_h_score_tmp = max([float(m.split('_')[4]) for m in saved_model_list_tmp]) if saved_model_list_tmp else -1
        else:
          max_h_score_saved = max([float(m.split('_')[2]) for m in saved_model_list]) if saved_model_list else -1
          max_h_score_tmp = max([float(m.split('_')[2]) for m in saved_model_list_tmp]) if saved_model_list_tmp else -1

        if max_h_score_tmp > max_h_score_saved:
            shutil.rmtree(current_run_dir)
            os.rename(current_run_tmp_dir, current_run_dir)
            print(f"Model saved with seed {config['seed']}, h_score {h_score}, direction {config['adaptation_direction']}, and type {config['baseline_or_proposed']}")
        else:
            shutil.rmtree(current_run_tmp_dir)
            print(f"Model not saved, h_score: {h_score} is not better than existing model's h_score: {max_h_score_saved}, direction {config['adaptation_direction']}, and type {config['baseline_or_proposed']}")

    return model_name

def plot_known_unknown_confusion_matrix(y_target, known_mask, unknown_label):
    true_known_unknown_labels = (y_target != unknown_label)
    predicted_known_unknown_labels = known_mask

    # Compute the confusion matrix
    cm_known_unknown = confusion_matrix(true_known_unknown_labels, predicted_known_unknown_labels)

    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(cm_known_unknown, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Known/Unknown Confusion Matrix')
    
    wandb.log({"Known-Unknown Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

def plot_class_labels_confusion_matrix(y_target, pred_labels, true_known_unknown_labels, all_classes):
    true_class_labels = y_target[true_known_unknown_labels]
    predicted_class_labels = pred_labels[true_known_unknown_labels]

    # Compute the confusion matrix
    cm_class_labels = confusion_matrix(true_class_labels, predicted_class_labels, labels=all_classes)
    
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(cm_class_labels, annot=True, fmt='g', xticklabels=all_classes, yticklabels=all_classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Class Labels Confusion Matrix')

    wandb.log({"Class Labels Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)


def plot_tsne(features, labels, epoch, entropy_val, config, filename, perplexity=30, known_unknown_labels=None):
    # Adapt perplexity according to the size of the input
    if len(features) <= perplexity:
        perplexity = len(features) - 1
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=300, random_state=int(config['seed']))
    tsne_results = tsne.fit_transform(features)

    df = pd.DataFrame()
    df['x-tsne'] = tsne_results[:,0]
    df['y-tsne'] = tsne_results[:,1]
    df['labels'] = labels

    # If known_unknown_labels is provided, add it to the DataFrame
    if known_unknown_labels is not None:
        df['known_unknown'] = known_unknown_labels

    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="x-tsne", y="y-tsne",
        hue="labels",
        style="known_unknown" if known_unknown_labels is not None else None,
        palette=sns.color_palette("hsv", len(set(labels))),
        data=df,
        legend="full",
        alpha=0.6
    )
    tsne_name = f'{filename}/tsne_epoch_{epoch}'
    title = f't-SNE plot at epoch {epoch}'
    if entropy_val is not None:
        tsne_name += f'_entropy_{entropy_val}'
        title += f' with entropy {entropy_val}'
    tsne_name += '.png'
    plt.title(title)
    plt.savefig(tsne_name)
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
            model_name = save_best_model(h_score, model, config, entropy_val, epoch)
            if(model_name!='no_model'):
              model_id = model_name + '_' + run_id
              filename = os.path.join("/content/drive/MyDrive/datasets-thesis/runs", model_id)
              if(not os.path.exists(filename)):
                os.mkdir(filename)
              

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

#n_frames=16, #4 and n_clips=1, #4
def prepare_datasets(source_dataset, target_dataset, val_dataset, source_txt_file_path, target_train_txt, target_test_txt, n_frames=16, n_clips=1, frame_size=224, normalize=True, fake_label = False):
    
    source_dataset = VideoDataset(
        source_dataset,
        source_txt_file_path,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
        train=True,
        augmentation=True,
        fake_label = False
    )

    target_dataset = VideoDataset(
        target_dataset,
        target_train_txt,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
        train=False,
        augmentation=True,
        fake_label = False
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
        augmentation=False,
        fake_label = False
    )

    return source_n_target_dataset, val_dataset

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
    # config['source_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train_source.txt'
    # config['source_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test_source.txt'
    config['source_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train.txt'
    config['source_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test.txt'

    #target HMDB
    path_target_train = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/train'
    path_target_test = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/test'
    # config['target_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train_target.txt'
    # config['target_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test_target.txt'
    config['target_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train.txt'
    config['target_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test.txt'

  else:
    #source HMDB
    path_source_train = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/train'
    path_source_test = '/content/openset_domain_adaptation/hmdb_ucf/hmdb/test'
    # config['source_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train_source.txt'
    # config['source_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test_source.txt'
    config['source_train_txt'] = '/content/openset_domain_adaptation/paths/hmdb_train.txt'
    config['source_test_txt'] = '/content/openset_domain_adaptation/paths/hmdb_test.txt'    
    
    #target UCF
    path_target_train = '/content/openset_domain_adaptation/hmdb_ucf/ucf/train'
    path_target_test = '/content/openset_domain_adaptation/hmdb_ucf/ucf/test' 
    # config['target_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train_target.txt'
    # config['target_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test_target.txt'   
    config['target_train_txt'] = '/content/openset_domain_adaptation/paths/ucf_train.txt'
    config['target_test_txt'] = '/content/openset_domain_adaptation/paths/ucf_test.txt' 


  config['path_source_train'] = path_source_train
  config['path_source_test'] = path_source_test
  config['path_target_train'] = path_target_train
  config['path_target_test'] = path_target_test
  return config
  
  
def create_datasets(config, n_frames=16):

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
                                                                      target_test_txt, n_frames=n_frames)

  if(config['g_open_set'] == True): 
      source_classes = source_n_target_train_dataset.source_dataset.classes
      print('source_classes: ',type(source_classes), source_classes)
      num_classes_to_remove = config['num_classes_to_remove']
      fake_source_label_or_remove_class = config['fake_source_label_or_remove_class']
      source_old_mapping = map_classes_to_labels(path_source_train)
      target_old_mapping = map_classes_to_labels(path_target_train)
      classes_to_remove, new_mapping, unknown_label = select_classes_to_remove_and_create_new_mapping(
          list(source_classes.keys()), 
          path_source_train,
          path_target_train,
          source_old_mapping,
          num_classes_to_remove,
          fake_source_label_or_remove_class
      )
      modify_labels_in_datasets(source_txt, target_train_txt, target_test_txt, source_old_mapping, target_old_mapping, new_mapping, unknown_label)
      #updating the class with new labels.
      source_n_target_train_dataset, target_test_dataset = prepare_datasets(path_source_train,
                                                                          path_target_train,
                                                                          path_target_test, 
                                                                          source_txt,
                                                                          target_train_txt,
                                                                          target_test_txt, fake_label=False, n_frames=n_frames)  
      source_n_target_train_dataset.unknown_label = unknown_label
      target_test_dataset.unknown_label = unknown_label

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
    title = 'Confusion Matrix'
    cm_name = f'{filename}/confusion_matrix_epoch_{epoch}'
    if entropy_val is not None:
        title += f', Entropy: {entropy_val}'
        cm_name += f'_entropy_val_{entropy_val}'
    cm_name += '.png'
    plt.title(title)
    plt.savefig(cm_name)
    plt.close()
    wandb.log({"confusion_matrix": wandb.Image(cm_name)})


def get_classes_from_dir(dir_path):
    return sorted(os.listdir(dir_path))

def select_classes_to_remove_and_create_new_mapping(all_classes_source, source_train_dir, target_train_dir, old_mapping, num_classes_to_remove, fake_source_label_or_remove_class='fake_source_label'):
    """
    Selects classes to be removed or faked (based on the "fake_source_label_or_remove_class" parameter) from the source dataset.
    Creates a new mapping for the labels of the source and target datasets.
    
    Args:
    - source_classes (dict): Old mapping of source class names to labels.
    - target_classes (dict): Old mapping of target class names to labels.
    - num_classes_to_remove (int): Number of classes to be removed or faked.
    - fake_source_label_or_remove_class (str): Either "fake_source_label" or "remove_class".

    Returns:
    - Tuple of:
        - New mapping for source dataset.
        - New mapping for target dataset.
        - List of classes to be removed or faked.
    """

    
    all_classes_target = get_classes_from_dir(target_train_dir)

    # Create a new mapping for the intersection classes with sequential labels
    new_mapping = {class_name: idx for idx, class_name in enumerate(all_classes_source)}
    
    # Identify classes in the target but not in the source
    target_only_classes = list(set(all_classes_target) - set(all_classes_source))
    
    forcibly_removed_common_classes = []
    forcibly_removed_distinct_source_classes = []

    if fake_source_label_or_remove_class == 'fake_source_label':
        only_in_source = list(set(all_classes_source) - set(all_classes_target))
        common_classes = list(set(all_classes_source).intersection(set(all_classes_target)))
        
        half_classes_to_remove = num_classes_to_remove // 2
        num_from_source_only = half_classes_to_remove if only_in_source else num_classes_to_remove
        num_from_both = half_classes_to_remove if common_classes else num_classes_to_remove
        
        if num_classes_to_remove % 2 != 0:
            if len(only_in_source) > len(common_classes):
                num_from_source_only += 1
            elif common_classes:
                num_from_both += 1

        num_from_source_only = min(num_from_source_only, len(only_in_source))
        num_from_both = min(num_from_both, len(common_classes))

        forcibly_removed_distinct_source_classes = random.sample(only_in_source, num_from_source_only) if only_in_source else []

        # If forcibly_removed_distinct_source_classes is not empty but less than half_classes_to_remove
        if forcibly_removed_distinct_source_classes and len(forcibly_removed_distinct_source_classes) < half_classes_to_remove:
            num_from_both += half_classes_to_remove - len(forcibly_removed_distinct_source_classes)
        elif not forcibly_removed_distinct_source_classes:
            num_from_both = num_classes_to_remove

        forcibly_removed_common_classes = random.sample(common_classes, num_from_both) if common_classes else [] #

    unknown_label = len(all_classes_source) - len(forcibly_removed_common_classes) - len(forcibly_removed_distinct_source_classes)


    # Classes to consider as unknown
    unknown_classes = forcibly_removed_common_classes + forcibly_removed_distinct_source_classes + target_only_classes
    
    # Separate classes into known and unknown
    known_classes = list(set(all_classes_source) - set(unknown_classes))
    
    # Create mappings for known and unknown classes
    dict_known = {class_name: idx for idx, class_name in enumerate(known_classes)}
    dict_unknown = {class_name: unknown_label for class_name in unknown_classes}

    # Merge the dictionaries
    new_mapping = {**dict_known, **dict_unknown}

    print("Summary of changes:")
    print("Label for unknown classes: ", unknown_label)

    print("\n###### SOURCE ######")
    print("Old mapping: ", old_mapping)
    print("New mapping: ", new_mapping)
    print("Fake label common classes: ", forcibly_removed_common_classes)
    print("Fake label distinct source classes: ", forcibly_removed_distinct_source_classes)
    print("Classes that had changes:")
    for class_name in all_classes_source:
        old_label = old_mapping.get(class_name, "Not in Old Mapping")
        new_label = new_mapping.get(class_name, unknown_label)
        print(f"[{class_name}, {old_label} -> {new_label}]")

    print("\n###### TARGET ######")
    print("Old mapping: ", old_mapping)
    print("New mapping: ", new_mapping)
    print("Classes that had changes:")
    for class_name in all_classes_target:
        old_label = old_mapping.get(class_name, "Not in Old Mapping")
        new_label = new_mapping.get(class_name, unknown_label)
        print(f"[{class_name}, {old_label} -> {new_label}]")

    return forcibly_removed_common_classes + forcibly_removed_distinct_source_classes, new_mapping, unknown_label

def map_classes_to_labels(dir_path):
    classes = sorted(os.listdir(dir_path))
    return {class_name: i for i, class_name in enumerate(classes)}

def modify_labels_in_file(filepath, new_mapping, unknown_label):
    """
    Modify the labels in a file according to a given mapping.

    Args:
    - filepath (str): Path to the file to be modified.
    - new_mapping (dict): A dictionary where keys are original class names and values are the new labels.
    - unknown_label (int): Label to assign to classes not found in the new mapping.

    Returns:
    - List of modified lines.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    mod_lines = []
    for line in lines:
        class_name, path, old_label = line.strip().split()
        
        if class_name in new_mapping:
            new_label = str(new_mapping[class_name])
        else:
            new_label = str(unknown_label)
        
        mod_line = f"{class_name} {path} {new_label}\n"
        mod_lines.append(mod_line)

    # Overwrite the file with modified labels
    with open(filepath, 'w') as file:
        file.writelines(mod_lines)

    return mod_lines

def modify_labels_in_datasets(source_txt, target_train_txt, target_test_txt, source_old_mapping, target_old_mapping, new_mapping, unknown_label):
    """
    Modify the labels in the dataset based on the new mapping.

    Args:
    - source_txt (str): Path to the source dataset txt file.
    - target_train_txt (str): Path to the target training dataset txt file.
    - target_test_txt (str): Path to the target test dataset txt file.
    - source_old_mapping (dict): Old mapping for the source dataset.
    - target_old_mapping (dict): Old mapping for the target dataset.
    - new_mapping (dict): New mapping for the classes.
    - unknown_label (int): Label for the unknown classes.

    Returns:
    - None
    """
    
    # Define paths for the modified txt files
    source_txt_mod = source_txt.replace('.txt', '_mod.txt')
    target_train_txt_mod = target_train_txt.replace('.txt', '_mod.txt')
    target_test_txt_mod = target_test_txt.replace('.txt', '_mod.txt')

    # Modify labels in the source dataset and write to _mod.txt
    mod_lines_source = modify_labels_in_file(source_txt, new_mapping, unknown_label)
    with open(source_txt_mod, 'w') as file:
        file.writelines(mod_lines_source)

    # Modify labels in the target training dataset and write to _mod.txt
    mod_lines_target_train = modify_labels_in_file(target_train_txt, new_mapping, unknown_label)
    with open(target_train_txt_mod, 'w') as file:
        file.writelines(mod_lines_target_train)

    # Modify labels in the target test dataset and write to _mod.txt
    mod_lines_target_test = modify_labels_in_file(target_test_txt, new_mapping, unknown_label)
    with open(target_test_txt_mod, 'w') as file:
        file.writelines(mod_lines_target_test)

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

def get_frames_by_class(txt_file_path, n_frames):
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

def changeTXT(root = '/content/openset_domain_adaptation/paths', destination = '/content', pattern_search="/content/hmdb_ucf_giacomo/data", pattern_out="/content/openset_domain_adaptation/hmdb_ucf"):  
  for file_name in [file for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]:
    file_path = os.path.join(root, file_name)
    with open(file_path, "r") as f:
        lines = f.readlines()

    file_path = os.path.join(destination, file_name)
    with open(file_path, "w") as f:
      
      for line in lines:
          f.write(line.replace(pattern_search, pattern_out))

def get_txt_classes(txt_list, txt_folder):
  df = pd.DataFrame(columns=['txt_name', 'classes'])
  for f in txt_list:
    with open(os.path.join(txt_folder,f), 'r') as f_txt:
      lines = f_txt.readlines()
      df_classes = pd.DataFrame([{'txt_name': f, 'classes': sorted(list(set([l.split()[-1] for l in lines])), key=int)}])
      df = pd.concat([df, df_classes] , ignore_index=True)
  return df

