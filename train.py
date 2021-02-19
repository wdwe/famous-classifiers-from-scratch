import os
import sys
import torch
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import models
import yaml
from data.dataset import TrainDataset, EvalDataset
from tensorboardX import SummaryWriter
from tools.utils import AverageMeter, add_weight_decay
from tools.eval import evaluate
from tools.eval import accuracy as compute_accuracy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



class Params:
    def __init__(self, train_params_file):
        with open(train_params_file) as f:
            self.params = yaml.safe_load(f)

    def __getattr__(self, item):
        return self.params.get(item, None)

# ModelWithLoss class has two purposes:
# 1. When we use data parallelism for training, there will be unbalanced gpu usage as explained
#   later. This wrapper forces pytorch to compute loss on the respective gpus instead of sending
#   the whole output back to the default gpus and compute the loss there.
# 2. Later when we train GoogLeNet, we will have to use a few side heads, i.e. attach fully connected
#   layers to the feature maps at a few intermediate stages of the network and let them make prediction.
#   The loss will be a weighted sum of the loss due to the final prediciton and intermediate predictions.
#   By using ModelWithLoss, we can let the class take care of the loss computation and easily reuse our training
#   script.

class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images, labels):
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        return preds, loss


def train(model, params):
    
    # helper function to print and save logs
    def print_log(string, print_time = True):
        if print_time:
            curr_time = time.asctime(time.localtime(time.time()))
            string = "[ " + curr_time + " ] " + string
        print(string)
        log_file = os.path.join(params.work_dir, "train_log.txt")
        with open(log_file, "a+") as log:
            log.write(string + "\n")
    
    # helper function to save checkpoints
    def save_checkpoint(best = False):
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.model.state_dict()
        else:
            model_state_dict = model.model.state_dict()
        ckpt_dict = {
            "epoch": e,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": best_accuracy
        }
        ckpt_dir = os.path.join(params.work_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok = True)
        if best:
            torch.save(ckpt_dict, os.path.join(ckpt_dir, "best.pth"))
        else:
            torch.save(ckpt_dict, os.path.join(ckpt_dir, f"epoch_{e}_step_{step}.pth"))

    # set up our project working directory
    os.makedirs(params.work_dir, exist_ok = True)
    # and save our training configuration
    with open(os.path.join(params.work_dir, "train_args.yaml"), "w+") as f:
        yaml.dump(params.params, f)

    # print out the settings for training
    print_log("Below are the training settings", print_time = False)
    for k, v in params.params.items():
        print_log(f"{k} : {v}", print_time = False)
    

    # tensorboard summary writer
    writer = SummaryWriter(os.path.join(params.work_dir, "events"))


    # initiating dataset and loader
    train_dir = os.path.join(params.data_root, "train")
    train_set = TrainDataset(
        imdir = train_dir,
        input_size = params.input_size,
        color_jitter = params.color_jitter,
        resize_scale = params.resize_scale,
        ratio = params.ratio,
        interpolation = params.interpolation,
        horizontal_flip = params.horizontal_flip,
        mean = params.mean,
        std = params.std,
        fname = True
    )

    train_loader = DataLoader(
        train_set, 
        batch_size = params.train_bs, 
        num_workers = params.num_workers,
        shuffle = True
    )


    # we will use center crop to evaluate the model's accuracy every epoch
    val_dir = os.path.join(params.data_root, "val")
    val_set = EvalDataset(
        imdir = val_dir,
        input_size = params.input_size,
        mean = params.mean,
        std = params.std,
        rescale_sizes = params.test_rescales,
        center_square = False,
        crop = "center",
        horizontal_flip = False
    )

    val_loader = DataLoader(
        val_set,
        batch_size = params.test_bs,
        shuffle = False,
        num_workers = params.num_workers
    )
    
    # GPU(s) or CPU usage
    if params.gpus:
        assert len(params.gpus) >= 1, "Please provide at least one gpu id for gpu training"
        if len(params.gpus) == 1:
            device = torch.device(f"cuda:{params.gpus[0]}")
            model = model.to(device)
            print_log(f"Training model on cuda: {params.gpus[0]}")
        else:
            device = torch.device(f"cuda:{params.gpus[0]}")

            # for parallelism, the model on the default gpu is still the one being updated
            # however, we replicate it to the other gpu every forward and backward pass
            # for gradient computation on the data that we allocated to those gpus
            model = model.to(torch.device(f"cuda:{params.gpus[0]}")) # it seems params.gpus must be like [0, 1] instead of [1, 0]
            model = nn.DataParallel(model, params.gpus)
            print_log(f"Data Parallelism is used across cuda: {params.gpus}")

    else:
        # the model stays on cpu
        print_log("Using cpu for training")
    

    # define optimizer
    # add in separate bn parameters
    if params.weight_decay is not None:
        # add_weight_decay separate bias and weight and bias in batchnorm from other parameters
        # because bias terms and and weight and bias in bn should not be decayed towards zero-norm
        # check here https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        param_groups = add_weight_decay(model, params.weight_decay)
    else:
        param_groups = model.parameters()

    # it is recommended to construct an optimizer after you have done the model.cuda(),
    # as some optimizer might create buffers of the type same as the model parameters.
    # since we will put our model to gpu, it is better that the model parameters have the type cuda 
    # instead of cpu before optimizer construction.
    optimizer = torch.optim.SGD(param_groups, lr = params.lr, weight_decay = params.weight_decay, momentum = params.momentum, nesterov=params.nesterov)

    # Let's define a learning rate scheduler that helps us reduce the learning rate by 10 times if our model's performance
    # on the validation set ceases to increase for 6 epochs
    # The mode should be min, so that it stores the min previous validation loss and compares that to the new validation loss
    # that we will provide when calling scheduler.step(<new_value>).
    # You may use "max" and validation accuracy too.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = 0.1, patience = 6)

    # resume from previous training
    if params.resume_path:
        ckpt_dict = torch.load(params.resume_path)
        model_state_dict = ckpt_dict["model_state_dict"]
        if isinstance(model, nn.DataParallel):
            model.module.model.load_state_dict(model_state_dict)
        else:
            model.model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
        start_epoch = ckpt_dict["epoch"]
        step = ckpt_dict["step"]
        if step != 0 and step % len(train_loader) == 0:
            # if we have finished the whole epoch last time before we saved the checkpoint
            # we move on to the next epoch
            start_epoch += 1
        best_accuracy = ckpt_dict["best_accuracy"]
        print_log(f"Loaded checkpoint {params.resume_path}")
        print_log(f"Resuming from epoch {start_epoch} step {step}")
    else:
        start_epoch = 0
        step = 0
        best_accuracy = 0


    batch_loss = AverageMeter()
    batch_accu = AverageMeter()
    try:
        # the try clause is for the except below where if we use ctrl-c/cmd-c to stop the program
        # it will save a checkpoint before exiting
        for e in range(start_epoch, params.num_epochs):
            model.train()
            progress_bar = tqdm(range(len(train_loader)))
            step_in_last_epoch = step % len(train_loader)
            
            loader = iter(train_loader)
            for i in progress_bar:
                # If we saved weights and stopped training halfway in an epoch, let's finish the remaining data in 
                # that epoch before moving on.
                # As the train_loader will be shuffled, we cannot really train the model on the data 
                # we left behind last time. However, it is easier for us to track training, as with these few lines
                # of code, we can align the stored epoch number correctly with the number of images trained (suppose the batch size is
                # the same, so the number of images trained per step is the same)
                if i < step_in_last_epoch:
                    progress_bar.update()
                    continue

                if i == len(train_loader) - step_in_last_epoch:
                    # if we have finished the equivalent amount of what we left behind last time
                    # stop this epoch and move on to the next
                    break

                
                data = next(loader)
                images = data['image']
                labels = data['label']
                
                # sending images and labels to gpus
                if params.gpus and len(params.gpus) == 1:
                    # if we are using one gpu
                    images = images.to(device)
                    labels = labels.to(device)
                # else:
                    # if the model is on cpu, then nothing needs to be done
                    # if multiple gpus are used, the data will be scattered to the corresponding gpus
                    # inside the nn.DataParallel class directly from CPU. Nothing needs to be done here.


                # forward pass the images to get prediction and loss
                # remember now our model is an instance of the wrapper ModelWithLoss.
                # It computes the loss inside its own forward() method.
                # Do not write as model(images = images, labels = labels) with DataParallel, as
                # they will then be counted as kwargs instead of tensor inputs
                preds, loss = model(images, labels)

                # compute accuracy for the training batch
                # Note: for the last batch or batch of odd number, the dataparallel may skip the remainder
                # when dividing the batch evenly among the gpus, resulting in different dimension between
                # preds and labels. Therefore, we need to take labels[:len(preds)]
                batch_accu.update(compute_accuracy(preds, labels[:len(preds)].view((-1, 1)))[0])
                
                # if we use data parallelism, the loss will be a vector with elements corresponding
                # to the loss on each gpu
                # it does not hurt if we are not using data parallelism
                loss = loss.mean()
                # compute dloss/dx for every parameter x that has requires_grad = True
                # and add this dloss/dx to the parameter's gradient
                # Initially, the parameters' gradients are all zero, loss.backward() adds the newly computed gradient
                # to the existing gradient. It will accumulate unless we call optimizer.zero_grad() to clear them.
                loss /= params.grad_accu_steps # see comments right below
                loss.backward()

                batch_loss.update(loss.item())
                
                step += 1
                # params.grad_accu_steps specifies for how many mini-batches we wish to accumulate gradients.
                # It is a work-around if we cannot fit a desirable size of mini-batch in GPU, we can simply accumulate
                # 2 or 3 batches' gradient before we call optimizer.step() (backpropagation).
                # However, this work-around has a difference when you have batch normalization layers,
                # as the running averages/variances of these are computed as exponential moving average. So
                # the running averages/variances statistics may deviate from using a larger batch.
                if step % params.grad_accu_steps == 0:
                    # Backpropagation to update parameters
                    optimizer.step()
                    # Set the gradients to zero, so that we can accumulate gradients from fresh
                    optimizer.zero_grad()
                
                if step % params.logging_interval == 0:
                    print_log(f"Epoch {e} Step {step}: Average loss is {batch_loss.avg:.4f} Training accuracy is {batch_accu.avg:.4f}")
                    for j, param_group in enumerate(optimizer.param_groups):
                        print_log(f"lr_{j} is {param_group['lr']}")
                    batch_loss.reset()
                    batch_accu.reset()
                
                writer.add_scalars("accuracy", {"train": batch_accu.val}, step)
                writer.add_scalars("loss", {"train": batch_loss.val}, step)
                for j, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"lr/lr_{j}", param_group["lr"], step)
                # update the information of the 
                progress_bar.set_description(f"Epoch {e}/{params.num_epochs} Step {step} Loss: {batch_loss.avg:.4f} Accuracy: {batch_accu.avg:.4f}")

            if (e + 1) % params.saving_interval == 0:
                save_checkpoint()

            # evaluate our model on the validation set
            # remember, our evaluate function can take care of ModelWithLoss wrapper
            if params.gpus and len(params.gpus) == 1:
                accu_meters, loss_meter = evaluate(model, val_loader, topk = (1, ), device = device)
            else:
                # dataparallel or cpu, let the data stay on cpu, see relevants comments above during training
                accu_meters, loss_meter = evaluate(model, val_loader, topk = (1, ))

            accuracy = accu_meters[0].avg
            print_log(f"Accuracy is {accuracy:.4f}, loss is {loss_meter.avg:.4f} for Epoch {e} Step {step} ")
            writer.add_scalars("accuracy", {"val": accuracy}, step)
            writer.add_scalars("loss", {"val": loss_meter.avg}, step)

            # update learning rate scheduler
            scheduler.step(loss_meter.avg)
            # scheduler.step(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(best = True)

            

    except KeyboardInterrupt:
        print_log("KeyboardInterrupt: Saving a checkpoint")
        save_checkpoint()







def train_process(gpu_id, model, train_params_file, nr):
    params = Params(train_params_file)
    rank = nr * params.gpus_per_node + gpu_id
    world_size = params.gpus_per_node * params.total_nodes
                       
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=world_size,                              
    	rank=rank                                               
    )


    # let's define the specific device in the node we are on 
    device = torch.device(f"cuda: {gpu_id}")

    # wrap the model
    model = model.to(device)
    model = DDP(model, device_ids=[gpu_id])

    # helper function to print and save logs
    def print_log(string, print_time = True):
        if print_time:
            curr_time = time.asctime(time.localtime(time.time()))
            string = "[ " + curr_time + " ] " + string
        print(string)
        log_file = os.path.join(params.work_dir, "train_log.txt")
        with open(log_file, "a+") as log:
            log.write(string + "\n")
    
    # helper function to save checkpoints
    def save_checkpoint(best = False):
        
        model_state_dict = model.module.model.state_dict()

        ckpt_dict = {
            "epoch": e,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": best_accuracy
        }
        ckpt_dir = os.path.join(params.work_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok = True)
        if best:
            torch.save(ckpt_dict, os.path.join(ckpt_dir, "best.pth"))
        else:
            torch.save(ckpt_dict, os.path.join(ckpt_dir, f"epoch_{e}_step_{step}.pth"))


    # tensorboard summary writer
    if gpu_id == 0:
        # save our training config
        os.makedirs(params.work_dir, exist_ok = True)
        with open(os.path.join(params.work_dir, "train_args.yaml"), "w+") as f:
            yaml.dump(params.params, f)

        # print out the settings for training
        print_log("Below are the training settings", print_time = False)
        for k, v in params.params.items():
            print_log(f"{k} : {v}", print_time = False)
        
        writer = SummaryWriter(os.path.join(params.work_dir, "events"))


    # initiating dataset and loader
    train_dir = os.path.join(params.data_root, "train")
    train_set = TrainDataset(
        imdir = train_dir,
        input_size = params.input_size,
        color_jitter = params.color_jitter,
        resize_scale = params.resize_scale,
        ratio = params.ratio,
        interpolation = params.interpolation,
        horizontal_flip = params.horizontal_flip,
        mean = params.mean,
        std = params.std,
        fname = True
    )

    # torch.utils.data.distributed.DistributedSampler()
    # Note: the batch_size is for each gpu, so one forward pass
    # takes in params.world_size * batch_size
    train_sampler = DistributedSampler(
    	train_set,
    	num_replicas=params.world_size,
    	rank=rank
    )

    train_loader = DataLoader(
        train_set, 
        batch_size = params.train_bs, 
        num_workers = params.num_workers,
        sampler = train_sampler,
        shuffle = False
    )

    # we will use center crop to evaluate the model's accuracy every epoch
    val_dir = os.path.join(params.data_root, "val")
    val_set = EvalDataset(
        imdir = val_dir,
        input_size = params.input_size,
        mean = params.mean,
        std = params.std,
        rescale_sizes = params.test_rescales,
        center_square = False,
        crop = "center",
        horizontal_flip = False
    )

    val_loader = DataLoader(
        val_set,
        batch_size = params.test_bs,
        shuffle = False,
        num_workers = params.num_workers
    )
    

    # define optimizer
    if params.weight_decay is not None:
        # add_weight_decay separate bias and weight and bias in batchnorm from other parameters
        # because bias terms and and weight and bias in bn should not be decayed towards zero-norm
        # check here https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        param_groups = add_weight_decay(model, params.weight_decay)
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.SGD(model.parameters(), lr = params.lr, weight_decay = params.weight_decay, momentum = params.momentum, nesterov=params.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 6)

    # resume from previous training
    # and map the gpu device for the weights
    # As pytorch tensors have their own associated devices, and we only (need to) save the model
    # on rank 1 gpu (which is always cuda: 0), our weights are all associated with cuda: 0.
    # Therefore, for processes in other gpus, we need to map the saved tensors' locations to 
    # other cuda device before we can use them. Hence the map_location dictionary
    if params.resume_path:
        map_location = {"cuda:0": f"cuda:{gpu_id}"}
        ckpt_dict = torch.load(params.resume_path, map_location = map_location)

        model_state_dict = ckpt_dict["model_state_dict"]
        model.module.model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
        start_epoch = ckpt_dict["epoch"]
        step = ckpt_dict["step"]
        if step != 0 and step % len(train_loader) == 0:
            start_epoch += 1
        best_accuracy = ckpt_dict["best_accuracy"]
        print_log(f"Loaded checkpoint {params.resume_path}")
        print_log(f"Resuming from epoch {start_epoch} step {step}")
    else:
        start_epoch = 0
        step = 0
        best_accuracy = 0


    batch_loss = AverageMeter()
    batch_accu = AverageMeter()
    try:
        for e in range(start_epoch, params.num_epochs):
            model.train()
            progress_bar = tqdm(range(len(train_loader)))
            step_in_last_epoch = step % len(train_loader)
            loader = iter(train_loader)
            for i in progress_bar:
                if i < step_in_last_epoch:
                    progress_bar.update()
                    continue

                if i == len(train_loader) - step_in_last_epoch:
                    break

                data = next(loader)
                images = data['image']
                labels = data['label']
                
                # remember the device here is torch.device(f"cuda:{gpu_id}")
                # i.e. the specific gpu we are on
                images = images.to(device)
                labels = labels.to(device)

                preds, loss = model(images, labels)

                batch_accu.update(compute_accuracy(preds, labels[:len(preds)].view((-1, 1)))[0])
                
                loss.backward()

                batch_loss.update(loss.item())
                
                step += 1

                optimizer.step()
                optimizer.zero_grad()
                
                # we only log if the process is on the first gpu of the node
                if gpu_id == 0:
                    if step % params.logging_interval == 0 :
                        print_log(f"Epoch {e} Step {step}: Average loss is {batch_loss.avg:.4f} Training accuracy is {batch_accu.avg:.4f}")
                        for j, param_group in enumerate(optimizer.param_groups):
                            print_log(f"lr_{j} is {param_group['lr']}")
                        batch_loss.reset()
                        batch_accu.reset()
                    
                    writer.add_scalars("accuracy", {"train": batch_accu.val}, step)
                    writer.add_scalars("loss", {"train": batch_loss.val}, step)
                    for j, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"lr/lr{j}", param_group["lr"], step)

            # we only save the checkpoint on the device with rank 0
            if (e + 1) % params.saving_interval == 0 and rank == 0:
                save_checkpoint()

            accu_meters, loss_meter = evaluate(model, val_loader, topk = (1, ), device = device)
            accuracy = accu_meters[0].avg

            # log if we are on gpu 0 of the node
            if gpu_id == 0:
                print_log(f"Accuracy is {accuracy:.4f}, loss is {loss_meter.avg:.4f} for Epoch {e} Step {step} ")
                writer.add_scalars("accuracy", {"val": accuracy}, step)
                writer.add_scalars("loss", {"val": loss_meter.avg}, step)

            # update learning rate scheduler
            scheduler.step(loss_meter.avg)
            
            # save the best accuracy check point if we are on the device with rank 0
            if accuracy > best_accuracy and rank == 0:
                best_accuracy = accuracy
                save_checkpoint(best = True)


    except KeyboardInterrupt:
        if gpu_id == 0:
            print_log("KeyboardInterrupt: Saving a checkpoint")
        if rank == 0:
            save_checkpoint()
        dist.destroy_process_group()
    
    dist.destroy_process_group()



def distributed_train(model, gpus_per_node, train_params_file, nr):
    os.environ['MASTER_ADDR'] = '192.168.2.1'
    os.environ['MASTER_PORT'] = '8888'

    # use seed to ensure the models are initialised with the same weights
    # it is in fact not necessary in our scripts, as I went for the not-reccommended
    # path of sending the model parameters to other process -- in mp.spawn, the args
    # includes the model
    # However, if the model initiation happens in train_process(), then we need this
    # seeding to ensure identical initialisation.
    torch.manual_seed(0)

    # talk about using docker
    mp.spawn(train_process, nprocs=gpus_per_node, args=(model, train_params_file, nr))  
    # Note: all the args are deep copied and sent to each functions

    # Function is called as the entrypoint of the spawned process. This function must be defined at 
    # the top level of a module so it can be pickled and spawned. This is a requirement imposed by 
    # multiprocessing.

    # The function is called as fn(i, *args), where i is the process index and args is the passed 
    # through tuple of arguments.



# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# why wrap model with loss can speed up and reduce memory on GPU:0

if __name__ == "__main__":
    from models import alexnet
    # create arg parser here
    train_params_file = sys.argv[1]
    params = Params(train_params_file)

    model = alexnet()
    model = ModelWithLoss(model)

    if params.distributed:
        distributed_train(model, params.gpus_per_node, train_params_file, nr = 0)
    else:
        train(model, params)


        # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html