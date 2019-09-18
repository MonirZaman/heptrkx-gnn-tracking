"""
Main training script for NERSC PyTorch examples
"""

# System
import os
import sys
import argparse
import logging
import pickle

# Externals
import yaml
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Locals
from datasets import get_data_loaders
from trainers import get_trainer
import distributed

def parse_args():
    """Parse command line arguments."""
    hpo_warning = 'Flag overwrites config value if set, used for HPO and PBT runs primarily'
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed', choices=['ddp-file', 'ddp-mpi', 'cray'])
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--ranks-per-node', default=8)
    add_arg('--gpu', type=int)
    add_arg('--rank-gpu', action='store_true')
    add_arg('--resume', action='store_true', help='Resume from last checkpoint')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--load_checkpoint', type=str, default=None,
            help='Location where checkpoint can be loaded from')
    add_arg('--save_checkpoint', type=str, default=None,
            help='Location to save a checkpoint')
    add_arg('--real-weight', type=int, default=None,
            help='class weight of real to fake edges for the loss. %s' % hpo_warning)
    add_arg('--lr', type=float, default=None,
            help='Learning rate. %s' % hpo_warning)
    add_arg('--hidden-dim', type=int, default=None,
            help='Hidden layer dimension size. %s' % hpo_warning)
    add_arg('--n_iters', type=int, default=None,
            help='Number of graph iterations. %s' % hpo_warning)
    return parser.parse_args()

def config_logging(verbose, output_dir, append=False, rank=0):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if output_dir is not None:
        log_dir = "%s_%s" % (output_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'out_%i.log' % rank)
        mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    # Suppress annoying matplotlib debug printouts
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

def init_workers(dist_mode):
    """Initialize worker process group"""
    if dist_mode == 'ddp-file':
        from distributed.torch import init_workers_file
        return init_workers_file()
    elif dist_mode == 'ddp-mpi':
        from distributed.torch import init_workers_mpi
        return init_workers_mpi()
    elif dist_mode == 'cray':
        from distributed.cray import init_workers_cray
        return init_workers_cray()
    return 0, 1

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

        
def update_config(config, args):
    """
    Updates config values with command line overrides. This is used primarily
    for HPO and PBT runs where hyperparameters must be exposed via command line flags.
    Returns the updated config.
    """
    if args.real_weight is not None:
        config['data']['real_weight'] = args.real_weight
    if args.lr is not None:
        config['optimizer']['learning_rate'] = args.lr
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.n_iters is not None:
        config['model']['n_graph_iters'] = args.n_iters

    return config

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Initialize distributed workers
    rank, n_ranks = init_workers(args.distributed)

    # Load configuration
    config = load_config(args.config)
    config = update_config(config, args)
    config['output_dir'] = os.path.expandvars(config.get('output_dir', None))
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=rank)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if args.show_config and (rank == 0):
        logging.info('Command line config: %s' % args)
    if rank == 0:
        logging.info('Configuration: %s', config)
        logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    if rank == 0:
        save_config(config)

    # Load the datasets
    is_distributed = (args.distributed is not None)
    # Workaround because multi-process I/O not working with MPI backend
    if args.distributed in ['ddp-mpi', 'cray']:
        logging.info('Disabling I/O workers because of MPI issue')
        config['data']['n_workers'] = 0
    train_data_loader, valid_data_loader = get_data_loaders(
        distributed=is_distributed, rank=rank, n_ranks=n_ranks, **config['data'])
    logging.info('Loaded %g training samples', len(train_data_loader.dataset))
    if valid_data_loader is not None:
        logging.info('Loaded %g validation samples', len(valid_data_loader.dataset))

    # Load the trainer
    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    logging.info('Choosing GPU %s', gpu)
    trainer = get_trainer(distributed_mode=args.distributed,
                          output_dir=config['output_dir'],
                          rank=rank, n_ranks=n_ranks,
                          gpu=gpu, save_checkpoint=args.save_checkpoint,
                          load_checkpoint=args.load_checkpoint, **config['trainer'])

    # Build the model and optimizer
    model_config = config.get('model', {})
    optimizer_config = config.get('optimizer', {})
    logging.debug("Building model")
    trainer.build_model(optimizer_config=optimizer_config, **model_config)
    if rank == 0:
        trainer.print_model_summary()

    # Checkpoint resume
    if args.resume:
        trainer.load_checkpoint()

    # Run the training
    logging.debug("Training")
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **config['training'])

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = summary.train_time.mean()
    logging.info('Train samples %g time %g s rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = summary.valid_time.mean()
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)

    # Drop to IPython interactive shell
    if args.interactive and (rank == 0):
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    if rank == 0:
        print("FoM: %e" % summary['valid_loss'].min())
        logging.info('All done!')

if __name__ == '__main__':
    main()
