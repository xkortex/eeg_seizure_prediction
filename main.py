"""
Process data via the command line.
DataPipeline is the main tool for automatically processing data through the whole flow. Otherwise, operations can be
accessed a-la-carte.
"""

from __future__ import print_function
from builtins import input
import argparse, os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import json

from exceptions import InsufficientDataError
# from .process.fileio import Batcher
from .tools.batcher import Batcher, wrapped_json
from .process.botobatcher import BotoBatcher, BotoQueue
from .process.pipeline import DataPipeline
from .process.reportcompile import reportcompile

def set_arg_parser():
    parser = argparse.ArgumentParser(description='Process accelerometer data. See docs/main.txt for more info')
    parser.add_argument('infile', type=str, nargs='?', default=os.getcwd(),
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
    parser.add_argument("-b", "--batch", action="store_true",
                        help="Run in (local) batch processing mode")
    parser.add_argument("-B", "--botobatch", action="store_true",
                        help="Run S3 files in batch processing mode using boto connection")
    parser.add_argument("-c", "--cyOn", action="store_true",
                        help="Use Cython to speed up (may be buggy)")
    parser.add_argument("-f", "--fallback", action="store_true",
                        help="Fallback mode. Uses a reference implementation of the EKF")
    parser.add_argument("-M", "--maxbatch", action="store_true",
                        help="Load ALL available files into the batch/boto queue, up to 99,999,999")
    parser.add_argument("-ne", "--noEKF", action="store_true",
                        help="Bypass the EKF. 30x faster processing, but not as sophisticated.")
    parser.add_argument("-NQ", "--newqueue", action="store_true",
                        help="Clear out the queue")
    parser.add_argument("-NC", "--newcache", action="store_true",
                        help="Clear out the S3 key cache")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot the resulting accelerometer data")
    parser.add_argument("-pi", "--pseudoinv", action="store_true",
                        help="Use linalg.pinv (Moore-Penrose Pseudoinverse) instead of linalg.inv for speedup")
    parser.add_argument("-zg", "--zerogyro", action="store_true",
                        help="Replace gyroscope data with all zeros, or fill the array if the gyro data is absent")
    parser.add_argument("-tj", "--testjson", action="store_true",
                        help="Use the path of the built-in test file(s)")
    parser.add_argument("-T", "--timeiso", action="store_true",
                        help="Set the time formatting from 'infer' to 'iso'")

    parser.add_argument("-s", "--speedHack", action="store_true",
                        help="Use matrix speed optimizations")
    parser.add_argument("-X", "--bypass", action="store_true",
                        help="Full bypass - generate report on dummy data")

    return parser

if __name__ == '__main__':
    parser = set_arg_parser()
    args = parser.parse_args()
    dataErrors = True if not args.nodataerror else False # invert because this is a default-on state
    if args.timeiso:
        timeFormat = 'iso'
    else:
        timeFormat = 'infer'
    if args.maxbatch:
        args.limit=99999999 # note: not actually limitless
    if args.diagnostics:
        print('Debug - Args: {}'.format(args))

    testpath = './zensmoothscoring/testing/trip_json/'
    if args.testjson or args.test == 1:
        # please don't edit the internal test file path, pass it in from the command line if you want to test a file
        path = testpath + 'daveioslondon01.json'

    elif args.test == 2:
        path = testpath + 'danielkuala2016-07-28.json'

    elif args.test == 3:
        path = testpath + 'stevelimioskuala2016-05-20.json'

    elif args.infile == os.getcwd(): # default behavior
        path = testpath + 'daveioslondon01.json'
    else:
        path  = args.infile

    success = 0
    runs = 0
    returnData = False
    report_update = {}
    if args.batch or args.botobatch:
        datestamp = datetime.now().isoformat()[:10]
        if args.batch:
            filelist = glob.glob(args.infile + '/*.json')[:args.limit]
            if args.verbose:
                print(filelist)
            resultspath = args.infile + '/results/'
            cfgpath='./zensmoothscoring/cfg/batch.cfg'
            batcher = Batcher(cfgpath, filenames=filelist, verbose=args.verbose, newQueue=args.newqueue)
            batcher.bind_output(wrapped_json, basepath=resultspath)
            batcher.bind_taskrun(DataPipeline.run_auto_process,
                                 noEKF=args.noEKF,      pseudoInv=args.pseudoinv,
                                 timeFormat=timeFormat, zerogyro=args.zerogyro,
                                 returnData=returnData, plot=args.plot,
                                 verbose=args.verbose,  dataErrors=dataErrors,
                                 cyOn=args.cyOn,        bypass=args.bypass)
            batcher.bind_finalize(reportcompile, basepath=resultspath)

        else:
            if not(args.newqueue):
                response = input("Reset queue? Warning this will overwrite the existing queue (default no) (y/[n]): ")
                if response.upper() == 'Y':
                    args.newqueue = True
                    args.newcache = True
                    print('Resetting queue.')
            cfgpath = './zensmoothscoring/cfg/default.cfg'
            # botobot = BotoBatcher(limit=args.limit, cfgpath=cfgpath, newQueue=args.newqueue)
            botobot = BotoQueue(limit=args.limit, cfgpath=cfgpath)
            queue = botobot.new_queue(newCache=args.newcache)
            batcher = Batcher(cfgpath, filenames=botobot.queue, verbose=args.verbose,
                              newQueue=args.newqueue)
            resultspath = botobot.resultspath + '/{}/'.format(datestamp)
            batcher.bind_output(wrapped_json, basepath=resultspath)
            batcher.bind_taskrun(botobot.pipewrapper,
                                 noEKF=args.noEKF,      pseudoInv=args.pseudoinv,
                                 timeFormat=timeFormat, zerogyro=args.zerogyro,
                                 returnData=returnData, plot=args.plot,
                                 verbose=args.verbose,  dataErrors=dataErrors,
                                 cyOn=args.cyOn,        bypass=args.bypass)
            batcher.bind_finalize(reportcompile, basepath=resultspath)

        batcher.run(verbose=args.verbose)

        if args.verbose:
            print('Batch complete')

    else:
        returnData = args.plot
        pipeout = DataPipeline.run_auto_process(path,
                                                noEKF=args.noEKF,       pseudoInv=args.pseudoinv,
                                                timeFormat=timeFormat,  zerogyro=args.zerogyro,
                                                returnData=returnData,  plot=args.plot,
                                                verbose=args.verbose,   dataErrors=dataErrors,
                                                cyOn=args.cyOn,         bypass=args.bypass,
                                                speedHack=args.speedHack,
                                                fallback=args.fallback)
        if args.plot:
            report, dataframes = pipeout
        else:
            report = pipeout
            dataframes = None
        # print(report)
        keylist = list(report.keys())
        keylist.sort()
        for key in keylist:
            print('{0: <16}: {1}'.format(key, report[key]))
#        if args.plot:
#           plt.plot(dataframes['carAcc'])
#           plt.show()

