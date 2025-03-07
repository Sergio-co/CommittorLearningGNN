#!/usr/bin/env python3
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from copy import deepcopy


#@torch.compile
def train_one_epoch(epoch_index, tb_writer, train_loader, model_to_train, optimizer, loss_function, kforce=1.0):
    report_step = max(1, len(train_loader) // 10)
    print(f'EPOCH {epoch_index + 1} ({len(train_loader)} batches):')

    running_loss = 0.0
    total_loss = 0.0

    model_to_train.train()
    for i, (data_t0, data_t1, labels) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        loss, q0, qt = loss_function(model_to_train, data_t0, data_t1, labels, kforce)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        total_loss += loss_value

        if (i + 1) % report_step == 0:
            avg_loss = running_loss / report_step
            q0_val = q0.item() if q0.numel() == 1 else q0.mean().item()
            qt_val = qt.item() if qt.numel() == 1 else qt.mean().item()

            print(f'  batch {i + 1} loss: {avg_loss:.4f} q0: {q0_val:.4f} qt: {qt_val:.4f}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
            running_loss = 0.0

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss

#@torch.compile
def val_one_epoch(epoch_index, val_loader, model_to_train, loss_function, kforce=1.0):
    running_loss = 0.0

    model_to_train.eval()
    with torch.no_grad():
        for data_t0, data_t1, labels in val_loader:
            loss, q0, qt = loss_function(model_to_train, data_t0, data_t1, labels, kforce)
            running_loss += loss.item()
        avg_loss = running_loss / len(val_loader)
    
    return avg_loss


def save_checkpoint(epoch, model, optimizer, writer_filename, best_epochs, best_model_state_dict, best_vloss, path):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'writer_filename': writer_filename,
                'best_epochs': best_epochs,
                'best_model_state_dict': best_model_state_dict,
                'best_vloss': best_vloss}, path)


def train_model(model_to_train, output_prefix, train_set, val_set, loss_function,
                epochs=1000, patience=20, batch_size=500, batch_size_factor=0.6, old_checkpoint=None,
                epoch_metrics_callback=None, dataloader=None, load_old_model_only=False, lr=1e-4, kforce=1.0):
    if dataloader is None:
        raise RuntimeError('Please provide a valid dataloader')
    # compute an appropriate batch size
    # see https://machine-learning.paperspace.com/wiki/epoch
    num_samples = len(train_set)
    print(f'Batch size: {batch_size}')
    train_loader = dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_set, batch_size=int(len(val_set)/100), shuffle=False)
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr, weight_decay=1e-5)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if old_checkpoint is not None:
        model_to_train.load_state_dict(old_checkpoint['model_state_dict'])
        if load_old_model_only is False:
            print('Will load the number of epochs and optimizer from previous training')
            epoch_number = old_checkpoint['epoch']
            optimizer.load_state_dict(old_checkpoint['optimizer_state_dict'])
            writer_filename = old_checkpoint['writer_filename']
            best_epochs = old_checkpoint['best_epochs']
            best_model_state_dict = deepcopy(old_checkpoint['best_model_state_dict'])
            best_vloss = old_checkpoint['best_vloss']
        else:
            print('Only the model is loaded from previous training')
    if old_checkpoint is None or load_old_model_only is True:
        epoch_number = 0
        writer_filename = f'{output_prefix}_trainer_{timestamp}'
        best_epochs = 0
        best_model_state_dict = deepcopy(model_to_train.state_dict())
        best_vloss = float('inf')
    writer = SummaryWriter(writer_filename)
    checkpoint_filename = f'{output_prefix}_best_model.pt'
    model_name = f'{output_prefix}_best_model.ptc'
    while epoch_number < epochs:
        avg_loss = train_one_epoch(epoch_number, writer, train_loader, model_to_train, optimizer, loss_function, kforce)
        avg_vloss = val_one_epoch(epoch_number, val_loader, model_to_train, loss_function, kforce)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        if epoch_metrics_callback is not None:
            results = epoch_metrics_callback(model_to_train, train_set, val_set)
            writer.add_scalars('Epoch metrics callback', results, epoch_number + 1)
        writer.flush()
        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_epoch_number = epoch_number
            best_vloss = avg_vloss
            #best_model_state_dict = deepcopy(model_to_train.state_dict())
            #torch.jit.script(model_to_train).save(model_name)
            torch.save(model_to_train, model_name)
            best_epochs = 0
        else:
            best_epochs += 1
            print(f'The best model has kept {best_epochs} epochs')
            if best_epochs >= patience:
                print(f'Early stopping after {best_epochs} epochs without improvement!')
                print(f'Best model obtained at epoch {best_epoch_number}')
                break
        # save the checkpoint
        #save_checkpoint(
        #    epoch_number + 1, model_to_train, optimizer, writer_filename,
        #    best_epochs, best_model_state_dict, best_vloss, checkpoint_filename)
        epoch_number += 1
    #model_to_train.load_state_dict(best_model_state_dict)
    return model_to_train
