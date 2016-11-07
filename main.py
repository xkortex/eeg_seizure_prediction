import argparse
import glob

from pipeline import switchboard


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


    ## models

    parser.add_argument("-SM", "--simplemetric", action="store_true",
                        help="Run simple metrics model")
    parser.add_argument("-q", "--showqueue", action="store_true",
                        help="Show queue")


    return parser

if __name__ == '__main__':
    # boilerplate
    parser = set_arg_parser()
    args = parser.parse_args()
    dataErrors = True if not args.nodataerror else False # invert because this is a default-on state

    if args.infile is None:
        raise ValueError("No input file specified")
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

    queue = queue[:args.limit]

    switchboard.run_a_process(processname, queue)
