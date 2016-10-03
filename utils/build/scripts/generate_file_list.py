#!/usr/bin/env python2

# INTEL CONFIDENTIAL
# Copyright 2016 Intel Corporation
#
# The source code contained or described herein and all documents related to the source code ("Material") are owned by
# Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual property right is granted to
# or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel
# in writing.
#
#
# For details about script please contact following people:
#  * [Version: 1.0] Walkowiak, Marcin <marcin.walkowiak@intel.com>


import argparse
import codecs
import os

import teamcity_utils as tcu


# Sending service messages that will add/update build parameters.
def main(parsedArgs):
    """ Main script function.

    The script generates file with file list from specific directory.

    :param parsedArgs: Arguments parsed by argparse.ArgumentParser class.
    :return: Exit code for script.
    :rtype: int
    """

    logger        = tcu.initLogger('FILE LIST', parsedArgs.log_file)
    scanDirectory = parsedArgs.dir if parsedArgs.dir is not None and parsedArgs.dir != '' else os.curdir
    baseDirectory = parsedArgs.base if parsedArgs.base is not None and parsedArgs.base != '' else os.curdir

    scanDirectory = tcu.cvtUni(os.path.normpath(scanDirectory))
    baseDirectory = tcu.cvtUni(os.path.normpath(baseDirectory))

    logger.debug('Scanning directory for files (recursive): "{0}"'.format(scanDirectory))

    allFilePaths = []
    for currentDir, subdirNames, fileNames in os.walk(scanDirectory):
        allFilePaths.extend([tcu.cvtUni(os.path.join(currentDir, fileName)) for fileName in fileNames])

    logger.debug('Found {0:>4d} files.'.format(len(allFilePaths)))

    # Absolute / relative.
    if parsedArgs.absolute:
        logger.debug('Absolute paths will be outputted.')

        allFilePaths = [tcu.cvtUni(os.path.abspath(filePath)) for filePath in allFilePaths]
    else:
        logger.debug('Relative paths will be outputted.')
        logger.debug('Paths are converted to relative form against base directory: "{0}"'.format(baseDirectory))

        allFilePaths = [tcu.cvtUni(os.path.relpath(filePath, baseDirectory)) for filePath in allFilePaths]
        if not parsedArgs.no_slash:
            allFilePaths = [filePath if filePath.startswith(os.sep) else ('/' + filePath) for filePath in allFilePaths]

    # Normalize separators.
    if os.sep == '\\':
        allFilePaths = [filePath.replace('\\', '/') for filePath in allFilePaths]

    # Write file.
    try:
        logger.debug('Writing file list to file: "{0}"'.format(parsedArgs.file))

        fileListFile = codecs.open(parsedArgs.file, 'w', 'utf-8')
        fileListFile.write('\n'.join(allFilePaths))  # Enforce Unix line endings.
        fileListFile.close()
    except:
        logger.error('Writing list of files failed (file: "{0}").'.format(parsedArgs.file))
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generates file with file list for specific directory.')
    parser.add_argument('-d',  '--directory',      dest = 'dir',      metavar = '<dir>',           type = unicode, default = None,                                        help = 'Directory to scan for files. Default/None specified: current working directory.')
    parser.add_argument('-b',  '--base-directory', dest = 'base',     metavar = '<base-dir>',      type = unicode, default = None,                                        help = 'Base directory (used when outputting relative file paths). Default/None specified: current working directory.')
    parser.add_argument('-f',  '--file',           dest = 'file',     metavar = '<filelist-file>', type = unicode, default = 'FileList.txt',                              help = 'Path to the file which will store file list. Default: FileList.txt.')
    parser.add_argument('-a',  '--absolute',       dest = 'absolute', metavar = '<absolute>',      type = int,     nargs = '?', const = 1, default = 0, choices = (0, 1), help = 'Indicates that absolute paths should be outputted to file.')
    parser.add_argument('-ns', '--no-slash',       dest = 'no_slash', metavar = '<no-slash>',      type = int,     nargs = '?', const = 1, default = 0, choices = (0, 1), help = 'Indicates that paths should not be prefixed with slash (only for relative paths).')
    parser.add_argument('-l',  '--log-file',       dest = 'log_file', metavar = '<log-file>',      type = unicode, default = None,                                        help = 'Path to log file.')
    parser.add_argument('--version',                                                                               action = 'version',                                    version = '%(prog)s 1.0')

    args = parser.parse_args()

    exitCode = main(args)
    parser.exit(exitCode)
