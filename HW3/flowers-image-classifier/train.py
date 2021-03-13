import argparse
from utils_ic import load_data, read_jason
from model_ic import NN_Classifier, validation, make_NN, save_checkpoint

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")
parser.add_argument("--no_of_trainable", type=int, default=0, help="set top n layers as trainable")
pretrained_group = parser.add_mutually_exclusive_group()
pretrained_group.add_argument('--pretrained', dest='pretrained', action='store_true', help="use pretrained weights")
pretrained_group.add_argument('--no-pretrained', dest='pretrained', action='store_false', help="do not use pretrained weights")
parser.set_defaults(pretrained=True)

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model = make_NN(n_hidden=[args.hidden_units], \
                n_epoch=args.epochs, \
                labelsdict=cat_to_name, \
                lr=args.learning_rate, \
                device=args.gpu, \
                model_name=args.arch, \
                trainloader=trainloader, \
                validloader=validloader, \
                train_data=train_data, 
                no_of_trainable=args.no_of_trainable, \
                pretrained=args.pretrained)

if args.save_dir:
    save_checkpoint(model, args.save_dir)