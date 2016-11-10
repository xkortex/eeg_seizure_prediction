from __future__ import print_function, division
import sys
import argparse
import glob

from pipeline import switchboard
from tools import menu


def set_arg_parser():
    parser = argparse.ArgumentParser(description='Process eeg data. See docs/main.txt for more info')
    parser.add_argument('infile', type=str, nargs='?',
                       help='Positional argument: Input file or path. If left blank, then default to the internal test file')
    parser.add_argument("-l", "--limit", type=int, default=10,
                    help="Set the maximum number of files to put in the batcher queue")
    parser.add_argument("-v", "--verbose", action="store_true",  # choices=[0, 1, 2], default=0,
                    help="output verbosity")
    parser.add_argument("-d", "--diagnostics", action="store_true",  #
                    help="specific messages for diagnostic/debugging")
    parser.add_argument("-t", "--test", type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help="output verbosity")
    parser.add_argument("-D", "--nodataerror", action="store_true",
                        help="Disable all data-quality errors, e.g. minimum speed.")
    parser.add_argument("-M", "--maxbatch", action="store_true",
                        help="Load ALL available files into the queue, up to 99,999,999")

    parser.add_argument("-p", "--pathmenu", action="store_true",
                        help="select path from menu")
    parser.add_argument("-P", "--pickpath", type=int, choices=list(range(20)), default=0,
                        help="Pick target path")


    ## models

    parser.add_argument("-SM", "--simplemetric", action="store_true",
                        help="Run simple metrics model")
    parser.add_argument("-q", "--showqueue", action="store_true",
                        help="Show queue")
    parser.add_argument("-BS", "--brainsound", action="store_true",
                        help="Convert to WAV files")
    parser.add_argument("-G", "--generalvec", action="store_true",
                        help="General vectorizer")


    return parser

if __name__ == '__main__':
    # boilerplate
    parser = set_arg_parser()
    args = parser.parse_args()
    dataErrors = True if not args.nodataerror else False # invert because this is a default-on state

    paths = {'/media/mike/Elements/data/kaggle/melbourne/train_all/': None,
             '/media/mike/Elements/data/kaggle/melbourne/train_1/': None,
             '/media/mike/Elements/data/kaggle/melbourne/train_2/': None,
             '/media/mike/Elements/data/kaggle/melbourne/train_3/': None,
             '/home/mike/Downloads/test_new/': None,
             'X /home/mike/Downloads/test_1_new/': None,
             'X /home/mike/Downloads/test_2_new/': None,
             'X /home/mike/Downloads/test_3_new/': None,
             '/home/mike/Downloads/train_1/': None,
             '/media/mike/Elements/data/kaggle/melbourne/': None}

    mymenu = menu.MenuPicker(paths)
    if args.pathmenu:
        args.infile = mymenu.user_pick_menu()
    print('Selection: ', args.infile)

    if args.infile is None:
        # raise ValueError("No input file specified")
        print("No input file specified")
        sys.exit(1)
    if args.maxbatch:
        args.limit=99999999 # note: not actually limitless
    if args.diagnostics:
        print('Debug - Args: {}'.format(args))

    queue = glob.glob(args.infile + '/*.mat')

    # actual stuff
    processname = None
    if args.simplemetric:
        processname = 'simple_metric'
    elif args.showqueue:
        processname = 'show_queue'
    elif args.brainsound:
        processname = 'brainsound'
    elif args.generalvec:
        processname = 'generalvec'

    queue = queue[:args.limit]

    switchboard.run_a_process(processname, queue)
