import argparse
from utils_ic import load_data, read_jason
from model_ic import load_model, test_model

parser = argparse.ArgumentParser(description="Test image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--load_dir", help="load model")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model = load_model(args.load_dir)

test_model(model, testloader, device=args.gpu)