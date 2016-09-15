#!/usr/bin/env python2


import argparse
import subprocess
from datetime import datetime

import teamcity_utils as tcu


# Sending service messages that will add/update build parameters.
def main(parsedArgs):
    """ Main script function.

    The script generates additional build information useful for TeamCity and Berta.

    :param parsedArgs: Arguments parsed by argparse.ArgumentParser class.
    :return: Exit code for script.
    :rtype: int
    """

    logger              = tcu.initLogger('BUILD INFO', parsedArgs.log_file)
    curUtcDateTime      = datetime.now() if parsedArgs.use_local_time != 0 else datetime.utcnow()
    curUtcDateTimeBerta = curUtcDateTime.strftime('%Y-%m-%d %H:%M:%S')

    tcu.updateTcParameter('my.build.info.utcdatetime.berta', curUtcDateTimeBerta)
    tcu.updateTcBuildNumber(parsedArgs.format.format(
        counter   = parsedArgs.counter,
        id        = parsedArgs.id,
        id_f4     = parsedArgs.id[:4],
        id_l4     = parsedArgs.id[-4:],
        id_f8     = parsedArgs.id[:8],
        id_l8     = parsedArgs.id[-8:],
        vcs_id    = parsedArgs.vcs_id,
        vcs_id_f4 = parsedArgs.vcs_id[:4],
        vcs_id_l4 = parsedArgs.vcs_id[-4:],
        vcs_id_f8 = parsedArgs.vcs_id[:8],
        vcs_id_l8 = parsedArgs.vcs_id[-8:],
        name      = parsedArgs.name,
    ))

    # Getting name and e-mail of the person who triggered build or author of last commit (if possible).
    lastCommitAuthId    = parsedArgs.change_auth  # ID -> AD SID
    lastCommitAuth      = ''
    lastCommitAuthEMail = ''

    if parsedArgs.change_auth != '' and parsedArgs.tc_url != '':
        conn = tcu.prepareTcRestConnection(parsedArgs.tc_url, parsedArgs.agent_user, parsedArgs.agent_pass)
        try:
            logger.debug("Trying to get author name and e-mail from TeamCity users information (REST API).")
            authInfo = conn('users/username:{userName}', userName = parsedArgs.change_auth)
            lastCommitAuthId    = authInfo['username'] if authInfo['username'] != '' else lastCommitAuthId
            lastCommitAuth      = authInfo['name']
            lastCommitAuthEMail = authInfo['email']
        except:
            logger.warning("Fetching TeamCity users information failed.")

    if lastCommitAuth == '':
        try:
            logger.debug("Trying to get author name from last (HEAD) commit in Git repository.")
            lastCommitAuth = subprocess.check_output(['git', 'show', '-s', '--format=%aN'],
                                                     universal_newlines = True).strip()
        except:
            logger.warning("Fetching Git users information failed.")

    if lastCommitAuthEMail == '':
        try:
            logger.debug("Trying to get author e-mail from last (HEAD) commit in Git repository.")
            lastCommitAuthEMail = subprocess.check_output(['git', 'show', '-s', '--format=%aE'],
                                                          universal_newlines = True).strip()
        except:
            logger.warning("Fetching Git users information failed.")

    lastCommitAuth = lastCommitAuth if lastCommitAuth != '' else lastCommitAuthId
    if lastCommitAuth != '':
        tcu.updateTcParameter('my.build.info.change.auth.user', lastCommitAuth)
    if lastCommitAuthEMail != '':
        tcu.updateTcParameter('my.build.info.change.auth.email', lastCommitAuthEMail)
        lastCommitPrettyAuth = u'{0} <{1}>'.format(lastCommitAuth, lastCommitAuthEMail)
    else:
        lastCommitPrettyAuth = u'{0}'.format(lastCommitAuth)
    if lastCommitPrettyAuth != '':
        tcu.updateTcParameter('my.build.info.change.auth.pretty', lastCommitPrettyAuth)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generates common build information. Updates build number to the more organized format.')
    parser.add_argument('counter',                                           metavar = '<counter>',             type = int,                                                            help = 'Current value of a (global) build counter.')
    parser.add_argument('id',                                                metavar = '<build-id>',            type = unicode,                                                        help = 'Unique identifier of current build.')
    parser.add_argument('vcs_id',                                            metavar = '<vcs-id>',              type = unicode,                                                        help = 'Unique identifier of main VCS (version control system) revision used in current build.')
    parser.add_argument('-n',   '--name',           dest = 'name',           metavar = '<build-name>',          type = unicode, default = 'manual',                                    help = 'Name of current build, e.g. ci-post-main, manual-pre, etc.')
    parser.add_argument('-f',   '--format',         dest = 'format',         metavar = '<build-number-format>', type = unicode, default = '{name}-{counter:0>5d}---{vcs_id_f8}',       help = 'Format for build number. Use "name", "counter", "id", "vcs_id", "vcs_id_f8" elements and .format() formatting.')
    parser.add_argument('-ult', '--use-local-time', dest = 'use_local_time', metavar = '<ult>',                 type = int,     nargs = '?', const = 1, default = 0, choices = (0, 1), help = 'Boolean int value / Flag which indicates that the local time should be used instead of UTC time in some build parameters.')
    parser.add_argument('-ca',  '--change-author',  dest = 'change_auth',    metavar = '<change-author>',       type = unicode, default = '',                                          help = 'TeamCity user name who triggered build. If the value is specified and verified by TeamCity REST API, VCS deduction (last commit author) is not performed.')
    parser.add_argument('-tc',  '--tc-server-url',  dest = 'tc_url',         metavar = '<teamcity-url>',        type = unicode, default = '',                                          help = 'URL to TeamCity server.')
    parser.add_argument('-au',  '--agent-user',     dest = 'agent_user',     metavar = '<agent-user-id>',       type = unicode, default = '',                                          help = 'Temporary agent user ID for TeamCity (to access TeamCity data).')
    parser.add_argument('-ap',  '--agent-password', dest = 'agent_pass',     metavar = '<agent-pass>',          type = unicode, default = '',                                          help = 'Temporary agent password for TeamCity (to access TeamCity data).')
    parser.add_argument('-l',   '--log-file',       dest = 'log_file',       metavar = '<log-file>',            type = unicode, default = None,                                        help = 'Path to log file.')
    parser.add_argument('--version',                                                                                            action = 'version',                                    version = '%(prog)s 1.0')

    args = parser.parse_args()

    exitCode = main(args)
    parser.exit(exitCode)
