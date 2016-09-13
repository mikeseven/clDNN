#!/usr/bin/env python2


import os
import sys
import json
from datetime import datetime
import berta_api
import getpass
import textwrap
import logging
import logging.handlers
import time
import argparse

settings = {} # Global settings file
log = None # Global log handle
p4 = None # Global PerforceApi class
berta_server = None # Global BertaApi class
scenarios_dict = {} # Global dict of scenarios in format => (id:name)
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

def init_settings(args):
    """ Initializes settings object from settings.json file """
    try:
        log.debug('init_settings()')

        # Load settings file.  If it fails to parse, bail
        global settings
        settings['berta_server_url'] = args.server
        settings['product'] = args.product
        settings['build_version'] = args.buildversion
        settings['build_succeeded'] = True if args.build_succeeded == "True" else False
        settings['build_store'] = args.build_store
        settings['build_publish_dir'] = args.buildpublishdir
        settings['correlation_build'] = args.correlation_build
        settings['build_info_url'] = args.build_info_url
        settings['user_name'] = args.username
        settings['build_date'] = args.builddate
        settings['build_note'] = args.build_note

        # Log this on the console so we can debug when things go wrong
        log.debug('  berta_server: %s' % args.server)
        log.debug('  berta_product: %s' % args.product)
        log.debug('  build_version: %s' % args.buildversion)
        log.debug('  build_info_url: %s' % args.build_info_url)
        log.debug('  build_store: %s' % args.build_store)
        log.debug('  build_publish_dir: %s' % args.buildpublishdir)
        log.debug('  build_succeeded: %s' % args.build_succeeded)
        log.debug('  build_date: %s' % args.builddate)
        log.debug('  build_user_name: %s' % args.username)
        log.debug('  webcache_root: %s' % args.webcache)
        log.debug('  correlation_build: %s' % args.correlation_build)
        log.debug('  api_new: %s' % args.api_new)

        # Dump out the list of files
        with open(args.filelist) as file_names:
            file_list = []
            for file in file_names:
                file_list.append(file)
            log.debug('  file_list: %s' %(file_list))

        # Get username and password
        username = ''
        if args.username:
            username = args.username
        else:
            username = getpass.getuser()
        passwd = '' # TODO: Get a better password management.  Maybe pass as arg?  getpass.getpass("\nEnter your password: ")


        # Get the users domain (required by some systems)
        domain = ''
        if "USERDOMAIN" in os.environ:
            domain = os.environ['USERDOMAIN']        

        # Initialize the Berta class
        global berta_server
        berta_server = berta_api.BertaApi(settings['berta_server_url'], username, passwd)
        if berta_server == None:
            log.error('Failed to create berta server ' + settings['berta_server_url'])
            return False

        return True
    except:
        log.exception('init_settings')
        return False

def init_logging(fname='upload_build_to_berta.log'):
    """ Initializes logging capabilities """
    global log
    log = logging.getLogger(fname)
    log.setLevel(logging.DEBUG)
    logformatter = logging.Formatter("%(asctime)-20s %(levelname)8s: %(message)s", datefmt='%b %d %H:%M:%S')
    handler = logging.handlers.RotatingFileHandler(fname, maxBytes=5 * 1024 * 1024, backupCount=2)
    handler.setFormatter(logformatter)
    log.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logformatter)
    log.addHandler(consoleHandler)    

    log.info('*' * 48)
    log.info('Starting')
    log.info('*' * 48)

def upload_build_to_berta(use_new_berta_api, webcache_root, build_publish_dir, build_info_url, send_files_filename, build_version, product_name, build_owner, correlation_build, build_date, build_succeeded, build_note):
    """ Call Berta API to send files and driver build information to Berta server """
    log.debug('upload_build_to_berta(%s)' % (build_version))

    files = []
    if use_new_berta_api == True:
        # New API needs to remove '/' from the beggining of the file list since it re-assembling the path to the file
        # Before: /Android/GMIN_L/Debug/android64/cht_rvp/system/lib/libgrallocclient.so
        # After : Android/GMIN_L/Debug/android64/cht_rvp/system/lib/libgrallocclient.so
        with open(send_files_filename) as file_names:
            for file in file_names:
                if file != "\n":
                    file = file[1:] if file[0] == "/" else file
                    files.append(file.replace("\n",""))

        # Add the build in the Berta server
        log.debug("add_build_from_store()")
        build_id = berta_server.add_build_from_store(settings['build_store'], build_publish_dir, product_name, build_version, build_owner, build_info_url, files, correlation_build, build_date, build_succeeded, build_note)
    else:
        # The legacy API needs list of files as fully qualified URLs
        files = get_qb_files_with_store_path(send_files_filename, webcache_root)

        # Add the build in the Berta server
        log.debug("add_build_from_quickbuild()")
        build_id = berta_server.add_build_from_quickbuild(files, build_version, product_name, build_owner, build_info_url, correlation_build, build_note)

    return build_id

def check_build_existance(build_version, product_id):
    """Check if berta includes build similar to build_version on given product_id"""
    log.debug('check_build_existance(%s)' % (build_version))
    try:
        build = berta_server.show_build_by_comment(product_id, build_version)
        return build
    except:
        return False

def get_product_info(product_name, build_succeeded):
    """ Get information about product if it exists """
    log.debug('get_product_info(%s)' % (product_name))
    products = berta_server.show_products(product_name)

    for product in products:
        if product['name'] == product_name:
            return product

    return False

def get_qb_files_with_store_path(filename, webcache_root):
    """ Open the file, walk each line and add the file to an array of file paths.
    Before adding, replace the file root with the webcache url """
    qb_files_with_store_path = []
    with open(filename) as file_names:
        for file_name in file_names:

            # This should be something like /Windows/Driver-Release-32-bit.7z
            qb_file = file_name.strip()
            if qb_file:
                # Add in the build publish directory, /masterdata/Artifacts/CI/Run/CI-Main/builds/2800620/artifacts
                # So the full string becomes
                #   /masterdata/Artifacts/CI/Run/DEV/IGC/builds/1714742/artifacts/Windows/Driver-Release-32-bit.7z
                qb_file = settings['build_publish_dir'] + qb_file

                # Now replace /master/Artifacts with the full webcache root and store the string
                qb_files_with_store_path.append(qb_file.replace("/masterdata/Artifacts", webcache_root))
    return qb_files_with_store_path

def save_build_to_json(build_id, file):
    """ Save the build id to a json file in the local workspace.
    If the file already exists, it will load existing data and then
    append or update the file.
    Example JSON data:
    {
        "http://bertax-synergy.igk.intel.com": {
            "build_id": 140010
        },
        "http://bertax-3dv.igk.intel.com": {
            "build_id": 394569
        }        
    } """

    log.debug('save_build_to_json(%s, %s)' %(build_id, file))
    build_data = {}
    if os.path.exists(file):
        build_data = load_json(file)

    build_data[settings['berta_server_url']] = {}
    build_data[settings['berta_server_url']]['build_id'] = build_id
    save_json(file, build_data)

    return 0

def main(args):
    
    init_logging()
    success = init_settings(args)

    if not success:
        log.info('Exiting with status 1')
        return 1

    # Check that the product actually exists in Berta prior to trying to add a build.
    product = get_product_info(settings['product'], args.build_succeeded)
    if not product:
        log.error('Specified product %s do not exists in Berta. Exiting with code (1)' % (settings['product']))
        return 1

    if not args.fakebuild:
        # Check if the build is available in Berta.
        already_exists = check_build_existance(settings['build_version'], product['id'])
        if already_exists:
            log.info('Specified build %s exists in Berta product = %s.' % (settings['build_version'], settings['product']))
        else:
            log.info('Build %s does not exist in Berta' %(settings['build_version']))
    else:
        already_exists = None

    if not args.api_new and already_exists:
        # add_build_from_quickbuild (the old Berta function) can be called only once for a given build.
        log.warning('Old Berta API in use (add_build_from_quickbuild) and the specified build %s exists in Berta product = %s. Exiting with code (0) ' % (settings['build_version'], settings['product']))
        # Just exit with success (0).
        return 0

    log.info('Uploading details about the build %s to berta' %(settings['build_version']))

    # Upload build
    if args.fakebuild:
        build_id = args.fakebuild
    else:
        # Remove the /masterdata/Artifacts from the start of the build publish dir from Quickbuild.
        # Before: /masterdata/Artifacts/CI/Run/DEV/IGC/builds/1714742/artifacts
        # After : CI/Run/DEV/IGC/builds/1714742/artifacts
        build_publish_dir = settings['build_publish_dir'].replace("/masterdata/Artifacts/","")
        build_id = upload_build_to_berta(args.api_new,
                                         args.webcache,
                                         build_publish_dir,
                                         settings['build_info_url'],
                                         args.filelist,
                                         settings['build_version'],
                                         settings['product'],
                                         settings['user_name'],
                                         settings['correlation_build'],
                                         settings['build_date'],
                                         settings['build_succeeded'],
                                         settings['build_note'])
        log.info('Added new build id, ' + str(build_id))

    if build_id < 0:
        log.error('Build was not added to Berta. Exiting with code (1) ')
        return 1

    # Save build details for a JSON file so other tools can consume the results
    status = save_build_to_json( build_id, 'build.json')
    if status == 0:
        log.info('Saved build details to file build.json')
    else:
        log.error('Failed to save build details to file build.json')

    # Exit
    if status == 0:
        log.info('Success! Exiting with status 0')
    else:
        log.info('Failed! Exiting with status ' + str(status))
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload QuickBuild binaries to Berta')
    parser.add_argument('-u', '--username', help='username to use')
    parser.add_argument('-s', '--server', help='berta server to connect to', required=True)
    parser.add_argument('-b', '--buildversion', help='quickbuild build version string associated with the build to upload to Berta, e.g. ci-dev-igc-12345', required=True)
    parser.add_argument('-p', '--product', help='berta product associated with build', required=True)
    parser.add_argument('-q', '--buildpublishdir', help='build publish dir, e.g. /masterdata/Artifacts/CI/Run/CI-Main/builds/1431321/artifacts', required=True)
    parser.add_argument('-w', '--webcache', help='The path to the webcache root for the driver build', default="https://ubitstore.intel.com/webstores/fm/sfa/Artifacts/Graphics/Builds", required=True)
    parser.add_argument('-f', '--filelist', help='A text file with a list of files to send to berta', default="FileList.txt", required=True)
    # Defaulting --build_succeeded to true in order to make it easier to test on configs that are still modifying the product name in Quickbuild if the build is failed.
    parser.add_argument('-bs', '--build_succeeded', help='Define if build was successful or failed.', default="True", required=False)
    parser.add_argument('-bst', '--build_store', help='Build store which builds will be uploaded from.', default='QuickBuild-2', required=False)
    parser.add_argument('-c', '--correlation_build', help='correlation build for berta', default="", required=False)
    parser.add_argument('-l', '--build_info_url', help='Build info url, e.g. https://ubit-gfx.intel.com/build/1431321', default="", required=False)
    parser.add_argument('-d', '--builddate', help='build start date', default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), required=False)
    parser.add_argument('-a', '--api_new', help='use the new add_build_from_store api in Berta', action='store_true', default=False)
    parser.add_argument('-fb', '--fakebuild', help='Instead of submitting the build to berta, just use the specified build ID instead', required=False)
    parser.add_argument('-bn', '--build_note', help='Note about build, it can be some user note, or SCM information, etc', default="", required=False)

    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
