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

    if parsedArgs.tc_url != '':
        conn = tcu.prepareTcRestConnection(parsedArgs.tc_url, parsedArgs.agent_user, parsedArgs.agent_pass)
        if parsedArgs.change_auth != '':
            try:
                logger.debug("Trying to get author name and e-mail from TeamCity users information (REST API).")
                authInfo = conn('users/username:{userName}', userName = parsedArgs.change_auth)
                lastCommitAuthId    = authInfo['username'] if authInfo['username'] != '' else lastCommitAuthId
                lastCommitAuth      = authInfo['name']
                lastCommitAuthEMail = authInfo['email']
            except:
                logger.warning("Fetching TeamCity users information failed.")

        # Designate good correlation build.
        try:
            logger.debug("Trying to get information about latest good correlation build from TeamCity (REST API).")
            buildStepCount = 100
            buildStartIdx = 0
            buildCount = buildStepCount
            foundGoodCorrBuild = False

            corrBranchName = parsedArgs.corr_branch.strip()
            corrBranchLocator = '(name:{branchName})'.format(branchName = corrBranchName) if corrBranchName != '' else '(default:true)'

            while buildCount >= buildStepCount and not foundGoodCorrBuild:
                logger.debug("Scanning latest successful finished correlation builds run on correlation branch ({0}-{1})..."
                             .format(buildStartIdx + 1, buildStartIdx + buildCount))
                buildsInfo = conn("builds/?locator=buildType:(id:{corrConfigId}),status:SUCCESS,personal:false,canceled:false,failedToStart:false,running:false,branch:{corrBranch},start:{startIdx},count:{count}&fields=count,build(id,number,finishDate,tags(count,tag))",
                                  corrConfigId = parsedArgs.corr_config_id, corrBranch = corrBranchLocator,
                                  startIdx = buildStartIdx,
                                  count = buildCount)

                buildCount = buildsInfo["count"]
                buildStartIdx += buildCount
                if buildCount > 0:
                    for buildInfo in buildsInfo["build"]:
                        isGoodCorrBuild = True
                        print buildInfo
                        if buildInfo["tags"]["count"] > 0:
                            for buildTag in buildInfo["tags"]["tag"]:
                                if tcu.cvtUni(buildTag["name"]).lower().strip() == 'bad-corr':
                                    isGoodCorrBuild = False
                                    break

                        if not isGoodCorrBuild:
                            logger.debug("Build {0:>12d} (finished: {1:<20s}, number: {2}) was rejected as correlation build due to being tagged as 'BAD-CORR'."
                                         .format(buildInfo["id"], buildInfo["finishDate"], buildInfo["number"]))
                            continue

                        logger.debug("Build {0:>12d} (finished: {1:<20s}, number: {2}) is selected as correlation."
                                     .format(buildInfo["id"], buildInfo["finishDate"], buildInfo["number"]))
                        tcu.updateTcParameter('my.build.info.corr.berta', buildInfo["number"])
                        foundGoodCorrBuild = True
                        break

            if not foundGoodCorrBuild:
                logger.warning("Could not locate good correlation build in TeamCity.")
        except:
            logger.warning("Fetching TeamCity builds information failed.")

    if lastCommitAuth == '':
        try:
            logger.debug("Trying to get author name from last (HEAD) commit in Git repository.")
            lastCommitAuth = tcu.cvtUni(subprocess.check_output(['git', 'show', '-s', '--format=%aN'],
                                                                universal_newlines = True)).strip()
        except:
            logger.warning("Fetching Git users information failed.")

    if lastCommitAuthEMail == '':
        try:
            logger.debug("Trying to get author e-mail from last (HEAD) commit in Git repository.")
            lastCommitAuthEMail = tcu.cvtUni(subprocess.check_output(['git', 'show', '-s', '--format=%aE'],
                                                                     universal_newlines = True)).strip()
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
    #parser.add_argument('-f',   '--format',         dest = 'format',         metavar = '<build-number-format>', type = unicode, default = '{name}-{counter:0>5d}---{vcs_id_f8}',       help = 'Format for build number. Use "name", "counter", "id", "vcs_id", "vcs_id_f8" elements and .format() formatting.')
    parser.add_argument('-f',   '--format',         dest = 'format',         metavar = '<build-number-format>', type = unicode, default = '{name}-{counter:0>5d}',     				   help = 'Format for build number. Use "name", "counter", "id", "vcs_id", "vcs_id_f8" elements and .format() formatting.')
    parser.add_argument('-ult', '--use-local-time', dest = 'use_local_time', metavar = '<ult>',                 type = int,     nargs = '?', const = 1, default = 0, choices = (0, 1), help = 'Boolean int value / Flag which indicates that the local time should be used instead of UTC time in some build parameters.')
    parser.add_argument('-ca',  '--change-author',  dest = 'change_auth',    metavar = '<change-author>',       type = unicode, default = '',                                          help = 'TeamCity user name who triggered build. If the value is specified and verified by TeamCity REST API, VCS deduction (last commit author) is not performed.')
    parser.add_argument('-cci', '--corr-config-id', dest = 'corr_config_id', metavar = '<build-config-id>',     type = unicode, default = 'SEIgk1_GenGpuClDNN_CiMain',                 help = 'Identifier of build configuration that will be used as source of correlation builds.')
    parser.add_argument('-cb',  '--corr-branch',    dest = 'corr_branch',    metavar = '<branch-name-or-ref>',  type = unicode, default = '',                                          help = 'Name or reference of branch in VCS that will be used as source of correlation builds. If not specified or empty, default branch is used.')
    parser.add_argument('-tc',  '--tc-server-url',  dest = 'tc_url',         metavar = '<teamcity-url>',        type = unicode, default = '',                                          help = 'URL to TeamCity server.')
    parser.add_argument('-au',  '--agent-user',     dest = 'agent_user',     metavar = '<agent-user-id>',       type = unicode, default = '',                                          help = 'Temporary agent user ID for TeamCity (to access TeamCity data).')
    parser.add_argument('-ap',  '--agent-password', dest = 'agent_pass',     metavar = '<agent-pass>',          type = unicode, default = '',                                          help = 'Temporary agent password for TeamCity (to access TeamCity data).')
    parser.add_argument('-l',   '--log-file',       dest = 'log_file',       metavar = '<log-file>',            type = unicode, default = None,                                        help = 'Path to log file.')
    parser.add_argument('--version',                                                                                            action = 'version',                                    version = '%(prog)s 1.0')

    args = parser.parse_args()

    exitCode = main(args)
    parser.exit(exitCode)
