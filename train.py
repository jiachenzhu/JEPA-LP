import os
import argparse

import torch
import torch.distributed as dist
from torchvision import datasets, transforms
from einops import rearrange

from helper import read_config, save_checkpoint, super_print, visualize_kernel
from augmentation import train_transform
from models import Encoder, Predictor, LatentGenerator, latent_generator
from lars import LARS
from scheduler import Scheduler
from loss import VICRegLossModule

def main():
    parser = argparse.ArgumentParser(description='JEPA')
    parser.add_argument('config_paths', nargs='+')
    args = parser.parse_args()
    config = read_config(args.config_paths)
    
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = int(os.getenv('SLURM_NNODES', '1')) * ngpus_per_node
    
    if 'SLURM_JOB_NODELIST' in os.environ:
        host_name = os.getenv('SLURM_JOB_NODELIST').split(',')[0].strip()
        config.dist_url = f'tcp://{host_name}:56384'
    else:
        config.dist_url = f'tcp://localhost:56384'
    
    print(config)
    print(f"node id: {int(os.getenv('SLURM_NODEID', '0'))} {ngpus_per_node}")
    
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

def main_worker(gpu, ngpus_per_node, config):
    device = torch.device(f"cuda:{gpu}")
    config.rank = gpu + int(os.getenv('SLURM_NODEID', '0')) * ngpus_per_node
    super_print(f"Rank {config.rank}")

    dist.init_process_group(backend='nccl', init_method=config.dist_url, world_size=config.world_size, rank=config.rank)
    torch.backends.cudnn.benchmark = True

    train_dataset = datasets.ImageFolder(f"{config.dataset_dir}/train", train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    
    device_batch_size = config.batch_size // config.world_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=device_batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    
    encoder = Encoder(config.projector_type, config.projector_dims).to(device)
    predictor = Predictor(config.projector_dims[-1], config.latent_dim).to(device)
    latent_generator = LatentGenerator(config.projector_dims[-1], config.hidden_dim, config.num_layers, config.latent_dim).to(device)

    encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
    latent_generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(latent_generator)

    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device])
    predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[device])
    latent_generator = torch.nn.parallel.DistributedDataParallel(latent_generator, device_ids=[device])

    encoder.train()
    predictor.train()
    latent_generator.train()

    optimizer = LARS(
        list(encoder.parameters()) + list(predictor.parameters()) + list(latent_generator.parameters()),
        lr=config.start_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        eta=config.eta,
        weight_decay_filter=config.weight_decay_filter,
        lars_adaptation_filter=config.lars_adaptation_filter
    )
    
    lr_scheduler = Scheduler(
        "lr",
        config.num_epochs, len(train_loader),
        config.start_lr * config.batch_size / 256, config.end_lr * config.batch_size / 256,
        config.lr_num_warmup_epochs,
        decay=config.lr_decay
    )

    loss_module = VICRegLossModule()

    super_print(f"Number of Steps per Epoch: {len(train_loader)}", rank=config.rank)
    if not os.path.exists(os.path.join(f"{config.checkpoint_dir}/{config.comment}", "checkpoint.pt")):
        if config.rank == 0:
            checkpoint = {
                'epoch': 0,
                'config': config,
                'encoder_state_dict': encoder.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'latent_generator_state_dict': latent_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint=checkpoint,
                directory=f"{config.checkpoint_dir}/{config.comment}",
                filename=f"checkpoint.pt"
            )
    dist.barrier()
    
    # load checkpoint
    ckpt = torch.load(os.path.join(f"{config.checkpoint_dir}/{config.comment}", "checkpoint.pt"), map_location='cpu')
    start_epoch = ckpt['epoch'] + 1
    super_print(f'resuming from checkpoint {start_epoch}', rank=config.rank)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    latent_generator.load_state_dict(ckpt['latent_generator_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        
        super_print(f"epoch {epoch} starts", rank=config.rank)
        for step, inputs in enumerate(train_loader, start=(epoch - 1) * len(train_loader)):
            # get lr
            lr = lr_scheduler.get_value(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad()
            
            x, _ = inputs
            x = x.to(device, non_blocking=True)

            representation = encoder(x)
            representation = rearrange(representation, "b c h w -> b h w c")
            representation_right = rearrange(representation, "b h w c -> (b h) w c")
            representation_left = torch.flip(representation_right, [1,])
            representation_down = rearrange(representation, "b h w c -> (b w) h c")
            representation_up = torch.flip(representation_down, [1,])
            
            # representation shape b, c, h, w
            # right, down
            latent_right = latent_generator(representation_right)
            latent_left = latent_generator(representation_left)
            latent_down = latent_generator(representation_down)
            latent_up = latent_generator(representation_up)

            prediction_right, ground_true_right = predictor(representation_right, latent_right, config.num_steps)
            prediction_left, ground_true_left = predictor(representation_left, latent_left, config.num_steps)
            prediction_down, ground_true_down = predictor(representation_down, latent_down, config.num_steps)
            prediction_up, ground_true_up = predictor(representation_up, latent_up, config.num_steps)
            
            loss = loss_module(representation, prediction_right, ground_true_right, prediction_left, ground_true_left, prediction_down, ground_true_down, prediction_up, ground_true_up, config.sim_coeff, config.std_coeff, config.cov_coeff)

            if loss.isnan().sum() > 0:
                print("loss is nan")
                exit()
            
            loss.backward()
            optimizer.step()

            super_print(f"{epoch:03d}-{step:04d} {lr_scheduler} -- {loss_module}", rank=config.rank)

        loss_module.reset_meters()

        if config.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'config': config,
                'encoder_state_dict': encoder.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'latent_generator_state_dict': latent_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint=checkpoint,
                directory=f"{config.checkpoint_dir}/{config.comment}",
                filename=f"checkpoint.pt"
            )
            if epoch % config.checkpoint_frequency == 0:
                save_checkpoint(
                    checkpoint=checkpoint,
                    directory=f"{config.checkpoint_dir}/{config.comment}",
                    filename=f"checkpoint_{epoch}.pt"
                )
            
            visualize_kernel(
                encoder.module.backbone[0].weight.detach().cpu(),
                directory=f"vis/{config.comment}",
                filename=f"vis_{epoch}.png"
            )
    
    dist.barrier()
    dist.destroy_process_group()
            
if __name__ == '__main__':
    main()    
