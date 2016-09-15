#!/usr/bin/env python2


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

    scanDirectory = unicode(os.path.normpath(scanDirectory))
    baseDirectory = unicode(os.path.normpath(baseDirectory))

    logger.debug('Scanning directory for files (recursive): "{0}"'.format(scanDirectory))

    allFilePaths = []
    for currentDir, subdirNames, fileNames in os.walk(scanDirectory):
        allFilePaths.extend([unicode(os.path.join(currentDir, fileName)) for fileName in fileNames])

    logger.debug('Found {0:>4d} files.'.format(len(allFilePaths)))

    # Absolute / relative.
    if parsedArgs.absolute:
        logger.debug('Absolute paths will be outputted.')

        allFilePaths = [unicode(os.path.abspath(filePath)) for filePath in allFilePaths]
    else:
        logger.debug('Relative paths will be outputted.')
        logger.debug('Paths are converted to relative form against base directory: "{0}"'.format(baseDirectory))

        allFilePaths = [unicode(os.path.relpath(filePath, baseDirectory)) for filePath in allFilePaths]
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
