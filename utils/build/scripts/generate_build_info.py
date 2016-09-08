#!/usr/bin/env python2


import argparse
import json
import re
import subprocess
import urllib2
import urlparse

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


def prepareTcRestConnection(teamCityUrl, agentUser, agentPass):
    if agentUser != None and agentUser != '':
        passMgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        passMgr.add_password(None, teamCityUrl, agentUser, agentPass)
        authHandler = urllib2.HTTPBasicAuthHandler(passMgr)
        restOpener = urllib2.build_opener(authHandler)
        urllib2.install_opener(restOpener)

    def prepareGetRequest(restGetRequest, **args):
        getRequestPart = unicode(restGetRequest).format(
            **{k: urllib2.quote(v, safe = '') for (k, v) in args.iteritems()})
        getRequestFragment = u'/httpAuth/app/rest/{0}'.format(getRequestPart)
        getRequestUrl = urlparse.urljoin(teamCityUrl, getRequestFragment)

        getRequest = urllib2.Request(getRequestUrl, headers = {'Accept' : 'application/json'})
        return json.loads(urllib2.urlopen(getRequest).read())

    return prepareGetRequest


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

    # Getting name and e-mail of the person who triggered build or author of last commit (if possible).
    lastCommitAuthId     = args.change_auth  #ID -> AD SID
    lastCommitAuth       = ''
    lastCommitAuthEMail  = ''
    lastCommitPrettyAuth = ''

    if args.change_auth != '' and args.tc_url != '':
        conn = prepareTcRestConnection(args.tc_url, args.agent_user, args.agent_pass)
        try:
            print "Trying to get author name and e-mail from TeamCity users information (REST API)."
            authInfo = conn('users/username:{userName}', userName = args.change_auth)
            lastCommitAuthId    = authInfo['username'] if authInfo['username'] != '' else lastCommitAuthId
            lastCommitAuth      = authInfo['name']
            lastCommitAuthEMail = authInfo['email']
        except:
            print "Fetching TeamCity users information failed."

    if lastCommitAuth == '':
        try:
            print "Trying to get author name from last (HEAD) commit in Git repository."
            lastCommitAuth = subprocess.check_output(['git', 'show', '-s', '--format=%aN'],
                                                     universal_newlines = True).strip()
        except:
            print "Fetching Git users information failed."

    if lastCommitAuthEMail == '':
        try:
            print "Trying to get author e-mail from last (HEAD) commit in Git repository."
            lastCommitAuthEMail = subprocess.check_output(['git', 'show', '-s', '--format=%aE'],
                                                          universal_newlines = True).strip()
        except:
            print "Fetching Git users information failed."

    lastCommitAuth = lastCommitAuth if lastCommitAuth != '' else lastCommitAuthId
    if lastCommitAuth != '':
        updateTcParameter('my.build.info.change.auth.user', lastCommitAuth)
    if lastCommitAuthEMail != '':
        updateTcParameter('my.build.info.change.auth.email', lastCommitAuthEMail)
        lastCommitPrettyAuth = u'{0} <{1}>'.format(lastCommitAuth, lastCommitAuthEMail)
    else:
        lastCommitPrettyAuth = u'{0}'.format(lastCommitAuth)
    if lastCommitPrettyAuth != '':
        updateTcParameter('my.build.info.change.auth.pretty', lastCommitPrettyAuth)

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
    parser.add_argument('--version',                                                                                            action = 'version',                                    version = '%(prog)s 1.0')

    args = parser.parse_args()
    print repr(args)

    exitCode = main(args)
    parser.exit(exitCode)