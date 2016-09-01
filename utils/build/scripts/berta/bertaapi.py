"""
bertapi.py

Description:
    Access Berta through xmlrpc
"""
#!/usr/bin/python
import os
import getpass
import json
import argparse
import xmlrpclib
import datetime
import re

class BertaApi():
    """Implements many helper functions for accessing data in Berta through its xmlrpc API"""
    domain = ''
    username = ''
    passwd = None
    server_url = ''

    def __init__(self, url, username, passwd, domain=''):
        """Initialize the class"""
        self.username = username
        self.passwd = passwd
        self.domain = domain
        self.server_url = url
        # if not self.server_url.endswith('/'):
        #     self.server_url += '/'

        # Authentication
        # Get the http string, ie 'http' or 'https'
        http = url.split('://')[0]
        # Get the host string
        host = url.split('://')[1]

        # http://docs.python.org/2/library/xmlrpclib.html
        # Both the HTTP and HTTPS transports support the URL syntax extension 
        # for HTTP Basic Authentication: http://user:pass@host:port/path. 
        # The user:pass portion will be base64-encoded as an HTTP 'Authorization' header, 
        # and sent to the remote server as part of the connection process when invoking an XML-RPC method.
        # You only need to use this if the remote server requires a Basic Authentication user and password.
        if http == "https":
            self.server_url = '%s://%s:%s@%s' %(http, username, passwd, host)

        self.proxy = xmlrpclib.ServerProxy(self.server_url, allow_none=True)

    def get_tasks(self, session_id, scenario_list = ''):
        """ Returns a dictionary of tasks for a given build_id."""
        start = 0
        limit = 100000
        stream=''
        system=''
        machines_group=''
        config=''   
        scenario=scenario_list
        tool=''
        build=''
        state=''
        project=''
        from_time=''
        to_time=''
        machines=''
        if_equal_completion_states=''
        completion_state=''
        covered=''
        test_session=str(session_id)

        # Get the list of tasks from the Berta server
        # It returns a tuple of the list of tasks and the number of tasks in the list
        tasks, num_tasks = self.proxy.show_tasks(start, limit, stream, system, machines_group, config, scenario, tool, build, state, project, from_time, to_time, machines, if_equal_completion_states, completion_state, covered, test_session)
        return tasks

    def get_test_cases(self, task_id):
        """ Returns a dictionary of test cases for a given task_id """
        return {}

    def get_builds(self, product_id):
        """ Returns an array of builds """        
        start = 0
        limit = 100
        
        builds, num_builds = self.proxy.show_builds(product_id, start, limit)

        return builds

    def get_testing_plans_configs(self, stream):
        """Returns list of testing_plan_config ids linked with given stream"""
        tps = self.proxy.show_testing_plans_configs_streams(str(stream))
        
        return tps

    def show_test_session_by_build(self, stream_id, build_id):
        """Returns test sessions from selected stream"""
        test_session = self.proxy.show_test_session_by_build(stream_id, build_id)

        return test_session
    
    def show_test_sessions(self,stream_id):
        """Returns test sessions from selected stream"""
        sessions, num_session = self.proxy.show_test_sessions(stream_id, 0, 1000)
        
        return sessions 
    
    def run_tasks_plans(self, stream_id, build_id, testing_plan_configs_ids, user_name = 'IGC_Jenkins'):
        
        session_id = self.proxy.run_tasks_plans(stream_id, build_id, testing_plan_configs_ids,'','',user_name)
        
        return session_id

    def show_products(self, product_name):

        products = self.proxy.show_products(product_name)

        return products

    def show_streams(self, stream_names):
        """ Get stream details from a comma separated list of streams names
        Example: ci-main,ufo-main,smoke 
        Returns: A list of dictionaries for each stream name in the list.  Each
        dictionary has details about the stream """
        streams = self.proxy.show_streams("all", stream_names)

        return streams

    def show_build_by_comment(self, product_id, build_comment):

        build = self.proxy.show_build_by_comment(product_id, build_comment)

        return build

    def add_build_from_quickbuild(self, files, build_comment, product_name, build_owner, build_info_url, correlation_build, build_note):
        """ Add build to Berta using 'add_build_from_quickbuild' Berta API.  This is now a LEGACY API """
        try:
            build_id = self.proxy.add_build_from_quickbuild(files, build_comment, product_name, build_owner, build_info_url, correlation_build, None, build_note)
        except xmlrpclib.Fault as err:
            if re.search('exceptions\.TypeError(.*)argument(.*)given', err.faultString):
                build_id = self.proxy.add_build_from_quickbuild(files, build_comment, product_name, build_owner, build_info_url, correlation_build, None, build_note)
            else:
                raise

        return build_id

    def add_build_from_store(self, buildstore, storage_directory, build_product, build_comment, build_owner, build_info_url, files, correlation_build, build_date, build_succeeded, build_note):
        """ Add build to Berta using 'add_build_from_store' Berta API. """
        if correlation_build == "":
            correlation_build = None
        try:
            build_id = self.proxy.add_build_from_store(buildstore, storage_directory, build_product, build_comment,build_owner, build_info_url, build_date, None, None, None, build_note, files, correlation_build, build_succeeded)
        except xmlrpclib.Fault as err:
            if re.search('exceptions\.TypeError(.*)argument(.*)given', err.faultString):
                build_id = self.proxy.add_build_from_store(buildstore, storage_directory, build_product, build_comment,build_owner, build_info_url, build_date, None, None, None, build_note, files, correlation_build, build_succeeded)
            else:
                raise
        return build_id

    def show_testing_plans_configs_streams(self, streams):

        testing_plans_configs = self.proxy.show_testing_plans_configs_streams(streams)

        return testing_plans_configs

    def show_build_test_sessions(self, build_id):

        test_sessions = self.proxy.show_build_test_sessions(build_id)

        return test_sessions


    def get_api_versions(self):

        api_version = self.proxy.get_api_version()

        return api_version

def save_json(file, data):
    """ Saves an object to a local JSON file """
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except:
        print('ERROR: save_json')

if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description='Berta Command Line Tool')
    parser.add_argument('-s', '--server', help='server url, e.g. http://bertax-3dv.igk.intel.com', required=True)
    parser.add_argument('--streams', help='stream names comma separated, e.g. UNIFIED-SMOKE,UNIFIED-REGRESSION')
    parser.add_argument('--stream_id', help='stream id, e.g. 33')
    parser.add_argument('--session_id', help='session id, e.g. 40204', type=int)
    parser.add_argument('--product_id', help='session id, e.g. 148')
    args = parser.parse_args()

    if "https_proxy" in os.environ:
        del os.environ["https_proxy"]
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if "USERDOMAIN" in os.environ:
        domain = os.environ['USERDOMAIN']

    username = getpass.getuser()
    #passwd = getpass.getpass("\nEnter your password: ")
    passwd = '' # Berta doesn't need a password right now

    berta = BertaApi(args.server, username, passwd)
    
    # TEST: Verify we can download berta tasks.  Saves data as json files
    if args.session_id:
        save_json( 'berta_test_session_%s.json' %(args.session_id), berta.get_tasks(args.session_id) )
    
    # TEST: Verify we can download berta build information.  Save builds as json files
    if args.product_id:
        save_json( 'berta_builds_%s.json' %(args.product_id), berta.get_builds(args.product_id) )

    # TEST: Verify we can download test sessions for a given stream, 
    #       walk the sessions and find a matching session_id.
    if args.stream_id and args.session_id:
        sessions = berta.show_test_sessions(args.stream_id)
        for session in sessions:
            if session['id'] == args.session_id:
                if session['still_testing']:
                    print('Still testing %s' %(session['id']) )
                else:
                    print('Session %s is Done' %(args.session_id))

    # TEST: Download stream details
    if args.streams:
        streams_details = berta.show_streams(args.streams)
        save_json( 'berta_stream_%s.json' %(args.streams), streams_details)
        print( streams_details)