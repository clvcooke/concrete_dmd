import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='cdmd')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=256,
                      help='# of images in each batch of data')
data_arg.add_argument('--resolution', type=int, default=32,
                      help='resolution of reconstruction')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=50,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=5,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--temp', type=float, default=1.0,
                       help="CONCRETE temperature adjustment")
train_arg.add_argument('--hidden_size', type=int, default=32,
                       help="hidden size of the networks")

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--task', type=str, default='mnist',
                      help='Task to train on')
misc_arg.add_argument('--init_strategy', type=str, default="flat",
                      help='initialization strategy for dmds')
misc_arg.add_argument('--adaptive', type=str2bool, default=False,
                      help="whether to use an adaptive (recurrent) approach or fixed set of patterns")
misc_arg.add_argument('--num_patterns', type=int, default=2,
                      help='# of dmd patterns to use')
misc_arg.add_argument('--adaptive_multi', type=int, default=1,
                      help='# of DMD patterns to use at a time for adaptive approaches')
misc_arg.add_argument('--noise', type=float, default=0.0,
                      help='sensor readout noise (absolute gaussian)')
misc_arg.add_argument('--dmd_type', type=str, default='spatial',
                      help='Type of DMD to use (spatial or frequency)')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
