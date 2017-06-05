import urlparse, urllib, urllib2, os.path, socket, sys

class Api(object):
    def __init__(self, url, ltokenPath):
        self.url = url

        scheme, server, path, query, anchor = urlparse.urlsplit(self.url)
        host, portstr = server.split(':')
        port = int(portstr)

        found = 0
        if os.path.exists(ltokenPath):
            with open(ltokenPath, 'r') as ltokenFile:
                for line in ltokenFile :
                    rd = line.strip().split(';')
                    self.token = rd[3]
                    self.user = rd[2]
                    found = 1
                    break
            if found == 0:
                print "ERROR - Cannot locate token entry in the ltoken file. Please run kwauth to reauthenticate with the desired server."
                return None
        else:
            print "ERROR - Cannot locate ltoken file. Please run kwauth to generate the ltoken file."
            return None
                

    def send_request(self, options):
        options["user"] = self.user
        if self.token is not None :
            options["ltoken"] = self.token

        data = urllib.urlencode(options)
        req = urllib2.Request(self.url, data)
        return urllib2.urlopen(req)