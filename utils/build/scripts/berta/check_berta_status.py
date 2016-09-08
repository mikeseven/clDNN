import os
import sys
import json
import bertaapi
import getpass
import textwrap
import logging
import logging.handlers
import time
import argparse

settings = {} # Global settings file
log = None # Global log handle
berta_server = None # Global BertaApi class

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
        berta_server = bertaapi.BertaApi(settings['berta_server'], username, passwd)
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
    found_session = False
    # Walk through each test session and find the one that matches the specified session ID
    for session in test_sessions:
        # Loop over the streams
        for stream in streams:        
            if session['stream_id'] == stream['id']:
                # Found the session matching session_id.  Now check if it is still running or not
                found_session = True

                if session['still_testing']:
                    # Tests are still running
                    status = True
                else:
                    # Testing is complete
                    status = False 

    if found_session == False:
        # If test session not found then probably script didn't receive complete test_session list. 
        # Return True so it will try again next time
        log.error('Failed to find any test sessions associated with streams ' + stream_names )
        status = True

    return status
    
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

    status = get_build_status( build['id'], settings['berta_streams'])

    # Exit
    if status == True:
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
    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
