#!/usr/bin/env python2


import argparse
import re
import subprocess

from datetime import datetime


def escapeTeamCityMsg(message):
    """ Escapes TeamCity server messages. """

    if not isinstance(message, (str, unicode)) or message == u'':
        return u'';
    return re.sub(ur'''['|\[\]]''', ur'|\g<0>', unicode(message)).replace(u'\n', u'|n').replace(u'\r', u'|r') \
        .replace(u'\u0085', '|x').replace(u'\u2028', '|l').replace(u'\u2029', '|p')


def updateTcParameter(paramName, message):
    """ Sends log service message to TeamCity which updates/adds build configuration parameter. """

    pName = escapeTeamCityMsg(paramName)
    msg   = escapeTeamCityMsg(message)
    print u"""##teamcity[setParameter name='{0}' value='{1}']""".format(pName, msg)


def updateTcBuildNumber(number):
    """ Sends log service message to TeamCity which updates build number for current build. """

    bNumber = escapeTeamCityMsg(number)
    print u"""##teamcity[buildNumber '{0}']""".format(bNumber)


# Sending service messages that will add/update build parameters.
def main(args):
    curUtcDateTime = datetime.now() if args.use_local_time != 0 else datetime.utcnow()
    curUtcDateTimeBerta = curUtcDateTime.strftime('%Y-%m-%d %H:%M:%S')

    updateTcParameter('my.build.info.utcdatetime.berta', curUtcDateTimeBerta)
    updateTcBuildNumber(args.format.format(
            counter   = args.counter,
            id        = args.id,
            id_f4     = args.id[:4],
            id_l4     = args.id[-4:],
            id_f8     = args.id[:8],
            id_l8     = args.id[-8:],
            vcs_id    = args.vcs_id,
            vcs_id_f4 = args.vcs_id[:4],
            vcs_id_l4 = args.vcs_id[-4:],
            vcs_id_f8 = args.vcs_id[:8],
            vcs_id_l8 = args.vcs_id[-8:],
            name      = args.name,
        ))

    # Getting last commit author name and e-mail (if possible).
    lastCommitAuth = args.change_auth
    try:
        lastCommitAuth = subprocess.check_output(['git', 'show', '-s', '--format=%aN'],
                                                 universal_newlines = True).strip()
    except:
        pass

    lastCommitAuthEMail = None
    try:
        lastCommitAuthEMail = subprocess.check_output(['git', 'show', '-s', '--format=%aE'],
                                                      universal_newlines = True).strip()
    except:
        pass

    
    updateTcParameter('my.build.info.change.auth', lastCommitAuth)
    if lastCommitAuthEMail != None:
        updateTcParameter('my.build.info.change.auth.email', lastCommitAuthEMail)
        updateTcParameter('my.build.info.change.auth.pretty', u'{0} <{1}>'.format(lastCommitAuth, lastCommitAuthEMail))
    else:
        updateTcParameter('my.build.info.change.auth.pretty', u'{0}'.format(lastCommitAuth))

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generates common build information. Updates build number to the more organized format.')
    parser.add_argument('counter',                                           metavar = '<counter>',             type = int,                                                            help = 'Current value of a (global) build counter.')
    parser.add_argument('id',                                                metavar = '<build-id>',            type = unicode,                                                        help = 'Unique identifier of current build.')
    parser.add_argument('vcs_id',                                            metavar = '<vcs-id>',              type = unicode,                                                        help = 'Unique identifier of main VCS (version control system) revision used in current build.')
    parser.add_argument('-n',   '--name',           dest = 'name',           metavar = '<build-name>',          type = unicode, default = 'manual',                                    help = 'Name of current build, e.g. ci-post-main, manual-pre, etc.')
    parser.add_argument('-f',   '--format',         dest = 'format',         metavar = '<build-number-format>', type = unicode, default = '{name}-{counter:0>5d}---{vcs_id_f8}',       help = 'Format for build number. Use "name", "counter", "id", "vcs_id", "vcs_id_f8" elements and .format() formatting.')
    parser.add_argument('-ult', '--use-local-time', dest = 'use_local_time', metavar = '<ult>',                 type = int,     nargs = '?', const = 1, default = 0, choices = (0, 1), help = 'Boolean int value / Flag which indicates that the local time should be used instead of UTC time in some build parameters.')
    parser.add_argument('-ca',  '--change-author',  dest = 'change_auth',    metavar = '<change-author>',       type = unicode, default = '',                                          help = 'Fall-back value for author of the last change. If the value can be deduced from VCS, this value will be ignored.')
    parser.add_argument('--version',                                                                                            action = 'version',                                    version = '%(prog)s 1.0')

    args = parser.parse_args()
    print repr(args)

    exitCode = main(args)
    parser.exit(exitCode)