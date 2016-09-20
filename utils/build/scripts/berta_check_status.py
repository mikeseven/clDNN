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
#  * [Version: 1.0] Sluis, Benjamin <benjamin.sluis@intel.com>
#  * [Version: 1.1] Walkowiak, Marcin <marcin.walkowiak@intel.com>   (Adaptation for TeamCity and fetching regressions)


import argparse
import getpass
import json
import logging
import logging.handlers
import os
import sys
import time

import berta_api
import teamcity_utils as tcu


settings = {}        # Global settings file
log = None           # Global log handle
berta_server = None  # Global BertaApi class


def load_json(file, data = None):
    """ Loads a local JSON file using the specified file key. A default
    JSON data object can be optionally passed in so default properties
    can be guranteed to exist """
    if data is None:
        data = {}

    try:
        log.debug('load_json(%s)' % (file))

        if os.path.exists(file):
            new_data = json.load(open(file, 'r'))
            for k,v in new_data.items():
                data[k] = v
    except Exception as e:
        log.exception('load_json')
    return data

def save_json(file, data):
    """ Saves an object to a local JSON file """
    try:
        log.debug('save_json(%s)' % (file))

        with open(file, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except:
        log.exception('save_json')

def init_settings( args ):
    """ Initializes settings object from settings.json file """
    try:
        log.debug('init_settings()')

        # Load settings file.  If it fails to parse, bail
        global settings
        settings['berta_server'] = args.server
        settings['berta_streams'] = args.streams
        settings['product'] = args.product
        settings['build_version'] = args.buildversion

        log.debug('  berta_server = ' + args.server)
        log.debug('  berta_streams = ' + args.streams)
        log.debug('  product = ' + args.product)
        log.debug('  build_version = ' + args.buildversion)

        # Get username and password
        username = ''
        if args.username:
            username = args.username
        else:
            username = getpass.getuser()
        passwd = '' # TODO: Get a better password management.  Maybe pass as arg?  getpass.getpass("\nEnter your password: ")

        # Initialize the Berta class
        global berta_server
        berta_server = berta_api.BertaApi(settings['berta_server'], username, passwd)
        if berta_server == None:
            log.error('Failed to create berta server object for ' + settings['berta_server'])
            return False

        return True
    except:
        log.exception('init_settings')
        return False

def init_logging(fname='check_berta_status.log'):
    """ Initializes logging capabilities """
    global log
    log = logging.getLogger(fname)
    log.setLevel(logging.DEBUG)
    logformatter = logging.Formatter("%(asctime)-20s %(levelname)8s: %(message)s", datefmt='%b %d %H:%M:%S')
    handler = logging.handlers.RotatingFileHandler(
        fname, maxBytes=5*1024*1024, backupCount=2)
    handler.setFormatter(logformatter)
    log.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logformatter)
    log.addHandler(consoleHandler)    

    log.info('*' * 48)
    log.info('Starting')
    log.info('*' * 48)

def check_build_existance(build_version, product_id):
    """Check if berta includes build similar to build_version on given product_id"""
    log.debug('check_build_existance(%s)' % (build_version))
    try:
        build = berta_server.show_build_by_comment(product_id, build_version)
        return build
    except:
        return False

def get_product_info(product_name):
    """ Get information about product if it exists """
    log.debug('get_product_info(%s)' % (product_name))
    products = berta_server.show_products(product_name)

    for product in products:
        if product['name'] == product_name:
            return product

    return False

def get_build_status( build_id, stream_names ):
    """gets status of the berta build"""
    log.debug('get_build_status(build_id=%s, stream_names=%s)' %(build_id, stream_names))
    
    # Returns JSON array of test sessions
    test_sessions = berta_server.show_build_test_sessions(build_id)

    # Convert the stream name to it's id
    streams = berta_server.show_streams(stream_names)

    # Loop through the test sessions and find the one that matches the stream id
    status = True
    found_sessions = []
    # Walk through each test session and find the one that matches the specified session ID
    for session in test_sessions:
        # Loop over the streams
        for stream in streams:        
            if session['stream_id'] == stream['id']:
                # Found the session matching session_id.  Now check if it is still running or not
                found_sessions.append(session['id'])

                if session['still_testing']:
                    # Tests are still running
                    status = True
                else:
                    # Testing is complete
                    status = False 

    if len(found_sessions) == 0:
        # If test session not found then probably script didn't receive complete test_session list. 
        # Return True so it will try again next time
        log.error('Failed to find any test sessions associated with streams ' + stream_names )
        status = True

    return status, found_sessions


def getRegressions(sessionIds):
    """ Fetches information about all regression from specific test sessions.

    :param sessionIds: List of identifiers of test sessions to check.
    :type sessionIds: list[int]
    """

    log.debug('getRegressions(sessionIds = {0})'.format(repr(sessionIds)))
    try:
        allRegressions = []
        allRegressionsCount = 0
        for sessionId in sessionIds:
            regressions, regressionsCount = berta_server.show_test_session_regressions(sessionId)
            allRegressions.extend(regressions)
            allRegressionsCount += regressionsCount

        if allRegressionsCount > 0:
            log.debug('Analysing possible regressions (count: {0:>4d})'.format(allRegressionsCount))

            testSuite = tcu.reportTcTestSuiteStart('Berta Test Changes')
            for regression in allRegressions:
                testCase = testSuite.reportTestStart(regression['test_case_name'])

                isRegression = regression['prev_status_changes'] == 0
                if regression['perf']:
                    if regression['perf_status'] == 'perf_imp':
                        testStatus = 'IMPROVEMENT'
                        isRegression = False
                    elif regression['perf_status'] == 'perf_ok':
                        testStatus = 'NO CHANGE'
                        isRegression = False
                    elif regression['perf_status'] == 'perf_reg':
                        testStatus = 'REGRESSION'
                    else:
                        testStatus = 'UNKNOWN ({0})'.format(regression['perf_status'])

                    testMessage = 'Test:        {0}\nTest status: {1}\n\nTest change: {2} -> {3} [diff: {4}]' \
                        .format(regression['cmd_line'], testStatus, regression['prev_result'], regression['result'],
                                regression['perf_diff'])
                else:
                    if unicode(regression['result']).lower().startswith('pass'):
                        isRegression = False

                    testMessage = 'Test:        {0}\nTest status: {1}\n\nTest change: {2} -> {1}' \
                        .format(regression['cmd_line'], regression['result'], regression['prev_result'])

                if isRegression:
                    testCase.reportFailed('REGRESSION', testMessage)
                else:
                    testCase.reportStdOut(testMessage)
            testSuite.finish()
            sys.stdout.flush()

        return False
    except:
        log.error('Fetching list of regressions failed for one of test sessions (ids = {})'.format(sessionIds))
        return True


def main( args ):
    init_logging()
    success = init_settings( args )
    
    if not success:
        log.info('Exiting with status 1')
        return 1

    # Check that the product actually exists in Berta prior to trying to add a build.
    product = get_product_info(settings['product'])
    if not product:
        log.error('Specified product %s do not exists in Berta. Exiting with code (1)' % (settings['product']))
        return 1

    # Check if the build is available in Berta. If not, exit since there is no build check status of
    build = check_build_existance(settings['build_version'], product['id'])
    if not build:
        log.error('Specified build %s doesn\'t exist in Berta product_id = %. Exiting with code (1) ' %(settings['build_version'],settings['product']))
        return 1

    # Waiting for build to end.
    status, found_sessions = get_build_status(build['id'], settings['berta_streams'])
    while status and args.interval > 0:
        log.info('Build is still running. Waiting {0:>3d} seconds...'.format(args.interval))
        time.sleep(args.interval)

        status, found_sessions = get_build_status(build['id'], settings['berta_streams'])

    # Getting regressions.
    if not status:
        status = getRegressions(found_sessions)

    # Exit
    if status:
        log.info('Build is still running')
    else:
        log.info('Build is complete')
    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Berta Poller')
    parser.add_argument('-u', '--username', help='username to use')
    parser.add_argument('-s', '--server', help='berta server to connect to', required=True)
    parser.add_argument('-b', '--buildversion', help='berta build name to check if it has any running test sessions in the specified streams, ie. ci-dev-igc-12345', required=True)
    parser.add_argument('-m', '--streams', help='berta streams associated with product, comma separated. e.g. unified-smoke,dev-igc', required=True)
    parser.add_argument('-p', '--product', help='berta product associated with build', required=True)
    parser.add_argument('-i', '--interval', metavar = '<interval>', type = int, default = 60, help = 'Interval of checking for status in seconds. If zero is specified, the script will not wait for status. Default: 60.')
    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
