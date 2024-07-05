import argparse

import torch

from predictor import Predictor
from yolox.exp.build import get_exp

def make_parser():
    parser = argparse.ArgumentParser("YOLOX")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )

    return parser


def main(exp, args):
    print("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = args.ckpt
    print("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")

    predictor = Predictor(
        model=model,
        exp=exp,
        device=args.device, 
        fp16=args.fp16, 
        legacy=args.legacy,
    )

    outputs, _ = predictor.inference(args.path)
    
    return outputs
    



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(exp_name=args.name)
    outputs = main(exp, args)
    print(outputs)