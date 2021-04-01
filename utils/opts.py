import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('-c', metavar='case', type=int, nargs='?',
                        help='case', default=0)
    parser.add_argument('-s', metavar='model', type=str, nargs='?',
                        help='model type')
    parser.add_argument('-r', metavar='retrain', type=bool, nargs='?',
                        help='retrain old model', default=False)
    parser.add_argument('-o', metavar='optimizer', type=int, nargs='?',
                        help='type of optimizer', default=0)
    parser.add_argument('-n', metavar='number of epochs', type=int, nargs='?',
                        help='number of epochs', default=4000)
    parser.add_argument('-bs', metavar='batch size', type=int, nargs='?',
                        help='batch size', default=32)  # OR .4
    parser.add_argument('-l', metavar='learning rate', type=float, nargs='?',
                        help='learning rate for both SGD and ADAM', default=1e-4)
    parser.add_argument('-w', metavar='weight_decay', type=float, nargs='?',
                        help='weight decay for ADAM and  SGD L2 penalty', default=1e-4)
    parser.add_argument('-job', metavar='job', type=str, nargs='?',
                        help='Job Name', default="")

    # ADAM PARAMETERS
    parser.add_argument('-b1', metavar='beta1', type=float, nargs='?',
                        help='beta1 for ADAM', default=.9)  # OR .4
    parser.add_argument('-b2', metavar='beta2', type=float, nargs='?',
                        help='beta2 for ADAM', default=.999)  # OR .4

    # SGD PARAMETERS
    parser.add_argument('-m', metavar='momentum', type=float, nargs='?',
                        help='momentum for SGD', default=0)  # OR .4

    # When use script =================================================
    # args = parser.parse_args(args=['-c', '-n', '-bs',
    #                                '-l', '-w', '-b1',
    #                               '-b2', '-m'])
    # args.c=0 ; args.n=4; args.bs=100 ;args.l=1e-4;
    # args.w=0 ; args.b1=.9; args.b2=.999; args.m=0;
    # args.r=False; args.o=0
    # ================================================================

    return parser.parse_args()