import os
import sys
import json
import bertaapi
import getpass
import textwrap
import logging
import logging.handlers
import time
import xmlrpclib
import argparse

settings = {}  # Global settings file
log = None  # Global log handle
berta_server = None  # Global BertaApi class


def load_json(file, data=None):
    """ Loads a local JSON file using the specified file key. A default
    JSON data object can be optionally passed in so default properties
    can be guranteed to exist """
    if data is None:
        data = {}

    try:
        log.debug('load_json(%s)' % (file))

        if os.path.exists(file):
            new_data = json.load(open(file, 'r'))
            for k, v in new_data.items():
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


def init_settings(args):
    """ Initializes settings object from settings.json file """
    try:
        log.debug('init_settings()')

        # Load settings file.  If it fails to parse, bail
        global settings
        settings['berta_server'] = args.server
        settings['berta_stream'] = args.stream
        settings['product'] = args.product
        settings['build_version'] = args.buildversion
        settings['test_plan_config_ids'] = args.testplanconfigids
        settings['check_session_exists'] = args.checksessionexists
        settings['username'] = args.username

        # Log this on the console so we can debug when things go wrong
        log.debug('  berta_server: %s' % str(args.server))
        log.debug('  berta_product: %s' % str(args.product))
        log.debug('  build_version: %s' % str(args.buildversion))
        log.debug('  test_plan_config_ids: %s' % str(args.testplanconfigids))
        log.debug('  check_session_exists: %s' % str(args.checksessionexists))
        log.debug('  berta_stream: %s' % str(args.stream))
        log.debug('  username: %s' % str(args.username))

        # Get username and password
        username = ''
        if args.username:
            username = args.username
        else:
            username = getpass.getuser()
        passwd = ''  # TODO: Get a better password management.  Maybe pass as arg?  getpass.getpass("\nEnter your password: ")

        # Initialize the Berta class
        global berta_server
        berta_server = bertaapi.BertaApi(settings['berta_server'], username, passwd)
        if berta_server is None:
            log.error('Failed to create berta server object for ' + settings['berta_server'])
            return False

        return True
    except:
        log.exception('init_settings')
        return False


def init_logging(fname='run_tests.log'):
    """ Initializes logging capabilities """
    global log
    log = logging.getLogger(fname)
    log.setLevel(logging.DEBUG)
    logformatter = logging.Formatter("%(asctime)-20s %(levelname)8s: %(message)s", datefmt='%b %d %H:%M:%S')
    handler = logging.handlers.RotatingFileHandler(
        fname, maxBytes=5 * 1024 * 1024, backupCount=2)
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


def get_test_data():
    """gets testing_plan_config ids linked with given stream for further execution"""
    log.debug('get_test_data()')

    tps = berta_server.get_testing_plans_configs(settings['berta_stream'])

    test_data = []
    if len(tps) > 0:
        for tps_row in tps:
            test_data.append(tps_row['testing_plan_config_id'])
    else:
        return False

    return test_data


def get_test_session_for_build(stream_id, build_id):
    test_session = None
    try:
        test_session = berta_server.show_test_session_by_build(stream_id, build_id)
        log.info('Test session found. Berta returned: %s' % str(test_session))
    except xmlrpclib.Fault as err:
        log.info('No test session found. Berta returned: %s' % err.faultString)
    return test_session


def trigger_execution(build_id):
    """Triggers selected testing_plan_configs to Berta for execution. Returns TestSession id"""
    log.debug('trigger_execution()')

    #testing_plan_configs = ",".join([str(x) for x in test_data])
    try:
        streams = berta_server.show_streams(settings['berta_stream'])
        if len(streams) == 1:
            stream_id = streams[0]['id']
            if settings['check_session_exists'] == 'True':
                test_session = get_test_session_for_build(stream_id, build_id)
                if test_session is not None:
                    log.warning('Test session %s already created for stream: %s' % (str(test_session['id']), settings['berta_stream']))
                    return test_session['id']
            session_id = berta_server.run_tasks_plans(stream_id, build_id, settings['test_plan_config_ids'], settings['username'])
        else:
            log.error('Could not find stream_id. Berta show_streams returned: %s' % str(streams))
            return False
    except:
        log.exception('trigger_execution')
        session_id = 0

    return session_id


def save_test_info_to_json(test_session_id, stream, file):
    """ Save the test session id to a json file in the local workspace.
    If the file already exists, it will load existing data and then
    append or update the file.
    Example JSON data:
    {
        "http://bertax-synergy.igk.intel.com": [
            {        
                "stream": "15.36-Regression",
                "test_session_id": 140010
            },
            {        
                "stream": "15.36-OGL",
                "test_session_id": 140011
            }
        ],
        "http://bertax-3dv.igk.intel.com": [
            {
                "stream": "15.36-ufo",
                "test_session_id": 394569
            }
        ]
    } """

    log.debug('save_test_info_to_json(%s, %s, %s)' % (test_session_id, stream, file))
    test_data = {}
    if os.path.exists(file):
        test_data = load_json(file)

    test_session_data = {}
    test_session_data['stream'] = stream
    test_session_data['test_session_id'] = test_session_id

    try:
        if not test_session_data in test_data[settings['berta_server']]:
            test_data[settings['berta_server']].append(test_session_data)
    except KeyError:
        test_data[settings['berta_server']] = []
        test_data[settings['berta_server']].append(test_session_data)

    save_json(file, test_data)

    return 0


def main(args):
    init_logging()
    success = init_settings(args)

    if not success:
        log.info('Exiting with status 1')
        return 1

    # Check that the product actually exists in Berta prior to trying to add a build.
    product = get_product_info(settings['product'])
    if not product:
        log.error('Specified product %s do not exists in Berta. Exiting with code (1)' % (settings['product']))
        return 1

    # Check if the build is available in Berta. If not, exit since there is no build to run tests against
    build = check_build_existance(settings['build_version'], product['id'])
    if not build:
        log.error('Specified build %s doesn\'t exist in Berta product_id = %s. Exiting with code (1) ' % (settings['build_version'], settings['product']))
        return 1

    # Getting test data to be executed
    test_data = get_test_data()
    if not test_data:
        log.warning(
            'No testing plans attached to stream \'%s\' in Berta - nothing to be executed. Exiting with code (0) ' % (
                settings['berta_stream']))
        return 0

    # Triggering execution to Berta
    if args.fake_session:
        session_id = args.fake_session
    else:
        session_id = trigger_execution(build['id'])
    if not session_id:
        log.error(
            'Attempted to triggered execution using testing_plan_config ids: %s , but Berta didn\'t respond with session_id. Exiting with code (1) ' % (
                str(test_data)))
        return 1
    else:
        log.info(
            'Successfully triggered execution, %s/test_session?id=%s' % (settings['berta_server'], str(session_id)))

    # Save test session information to disk
    status = save_test_info_to_json(session_id, settings['berta_stream'], 'tests.json')
    if status == 0:
        log.info('Saved test details to file tests.json')
    else:
        log.error('Failed to save test details to file tests.json')

    # Exit
    if status == 0:
        log.info('Success! Exiting with status 0')
    else:
        log.info('Failed! Exiting with status ' + str(status))
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Berta Poller')
    parser.add_argument('-u', '--username', help='username to use')
    parser.add_argument('-s', '--server', help='berta server to connect to', required=True)
    parser.add_argument('-b', '--buildversion',
                        help='quickbuild build version string associated with the build to upload to Berta, ie. ci-dev-igc-12345',
                        required=True)
    parser.add_argument('-m', '--stream', help='berta stream associated with product', required=True)
    parser.add_argument('-p', '--product', help='berta product associated with build', required=True)
    parser.add_argument('-t', '--testplanconfigids', help='berta testing plan configs id associated with stream',
                        default="", required=False)
    parser.add_argument('-c', '--checksessionexists', help='check if session for the stream already exists',
                        default=False, required=False)
    parser.add_argument('-fs', '--fake_session', type=int,
                        help='use this session id instead of triggering a new session')
    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
