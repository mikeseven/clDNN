#!/usr/bin/env python2


import json
import logging
import logging.handlers
import re
import urllib2
import urlparse
import weakref


def escapeTeamCityMsg(message):
    """ Escapes TeamCity server messages.

    :param message: Message to escape.
    :type message: string
    :return: Escaped string that can be used in service messages sent to TeamCity.
    :rtype: unicode
    """

    if not isinstance(message, (str, unicode)) or message == u'':
        return u''
    return re.sub(ur'''['|\[\]]''', ur'|\g<0>', unicode(message)).replace(u'\n', u'|n').replace(u'\r', u'|r') \
        .replace(u'\u0085', '|x').replace(u'\u2028', '|l').replace(u'\u2029', '|p')


def updateTcParameter(paramName, value):
    """ Sends log service message to TeamCity which updates/adds build configuration parameter.

    :param paramName: Name of TeamCity parameter to add or update.
    :type paramName: string
    :param value: New value of the parameter.
    :type value: string
    """

    pName = escapeTeamCityMsg(paramName)
    msg   = escapeTeamCityMsg(value)
    print u"""##teamcity[setParameter name='{0}' value='{1}']""".format(pName, msg)


def updateTcBuildNumber(number):
    """ Sends log service message to TeamCity which updates build number for current build.

    :param number: New number string that will update current build number.
    :type number: string
    """

    bNumber = escapeTeamCityMsg(number)
    print u"""##teamcity[buildNumber '{0}']""".format(bNumber)


########################################################################################################################


def reportTcTestSuiteStart(suiteName):
    """ Sends log service message to TeamCity which reports start of test suite.

    :param suiteName: Name of test suite which is starting.
    :type suiteName: string
    """

    class TestReporter:
        """ Reporter of test suite execution for TeamCity. """

        def __init__(self, testSuiteReporter, name, captureStdOutput = False):
            """ Creates new instance of object which reports to TeamCity test results.

            :param testSuiteReporter: Reporter of test suite containing current test.
            :type testSuiteReporter: TestSuiteReporter
            :param name: Test name.
            :type name: string
            :param captureStdOutput: Indicates that any standard output after reporter construction and before
                                     reporter finish is test output.
            :type captureStdOutput: bool
            """
            escapedTestName = escapeTeamCityMsg(name)
            print u"""##teamcity[testStarted name='{0}' captureStandardOutput='{1}']""" \
                .format(escapedTestName, repr(bool(captureStdOutput)).lower())

            self.name   = escapedTestName
            self.parent = testSuiteReporter

            self.failOrIgnoreReported = False
            self.stdOutReported       = False
            self.stdErrReported       = False

        def __del__(self):
            self.finish()

        def reportIgnored(self, ignoreComment):
            """ Sends log service message to TeamCity which reports that test was ignored.

            :param ignoreComment: Comment detailing why test was ignored.
            :type ignoreComment: string
            """

            if self.name is None:
                raise RuntimeError('Test is already reported as finished. No more test output can be added.')
            if self.failOrIgnoreReported:
                return self

            escapedComment = escapeTeamCityMsg(ignoreComment)
            print u"""##teamcity[testIgnored name='{0}' message='{1}']""".format(self.name, escapedComment)
            self.failOrIgnoreReported = True

            return self

        def reportStdOut(self, out):
            """ Sends log service message to TeamCity which reports standard output stream from test.

            :param out: Standard output stream content from test.
            :type out: string
            """

            if self.name is None:
                raise RuntimeError('Test is already reported as finished. No more test output can be added.')
            if self.stdOutReported:
                raise RuntimeError('Standard output stream was already reported in current test.')

            escapedOut = escapeTeamCityMsg(out)
            print u"""##teamcity[testStdOut name='{0}' out='{1}']""".format(self.name, escapedOut)
            self.stdOutReported = True

            return self

        def reportStdErr(self, out):
            """ Sends log service message to TeamCity which reports standard error stream from test.

            :param out: Standard error stream content from test.
            :type out: string
            """

            if self.name is None:
                raise RuntimeError('Test is already reported as finished. No more test output can be added.')
            if self.stdErrReported:
                raise RuntimeError('Standard error stream was already reported in current test.')

            escapedOut = escapeTeamCityMsg(out)
            print u"""##teamcity[testStdErr name='{0}' out='{1}']""".format(self.name, escapedOut)
            self.stdErrReported = True

            return self

        def reportFailed(self, message, details = None, expected = None, actual = None):
            """ Sends log service message to TeamCity which reports that test failed.

            :param message: Message describing cause of the failure.
            :type message: string
            :param details: Details of the failure.
            :type details: string
            :param expected: Expected value in test (in case of comparison failure).
            :type expected: Any
            :param actual: Actual value in test (in case of comparison failure).
            :type actual: Any
            """

            if self.name is None:
                raise RuntimeError('Test is already reported as finished. No more test output can be added.')
            if self.failOrIgnoreReported:
                return self

            escapedMessage = escapeTeamCityMsg(message)
            escapedDetails = escapeTeamCityMsg(details)
            if expected is None or actual is None:
                print u"""##teamcity[testFailed name='{0}' message='{1}' details='{2}']""" \
                    .format(self.name, escapedMessage, escapedDetails)
            else:
                escapedExpected = escapeTeamCityMsg(repr(expected))
                escapedActual   = escapeTeamCityMsg(repr(actual))
                print u"""##teamcity[testFailed type='comparisonFailure' name='{0}' message='{1}' details='{2}' expected='{3}' actual='{4}']""" \
                    .format(self.name, escapedMessage, escapedDetails, escapedExpected, escapedActual)
            self.failOrIgnoreReported = True

            return self

        def finish(self, durationMs = None):
            """ Sends log service message to TeamCity which reports finish of current test.

            :param durationMs: Test duration in milliseconds.
            :type durationMs: int
            :return: Parent test suite reporter (method chaining).
            :rtype: TestSuiteReporter
            """
            if self.name is not None:
                if durationMs is not None:
                    print u"""##teamcity[testFinished name='{0}' duration='{1:d}']""".format(self.name, int(durationMs))
                else:
                    print u"""##teamcity[testFinished name='{0}']""".format(self.name)
                self.name = None

            return self.parent

    class TestSuiteReporter:
        """ Reporter of test suite execution for TeamCity. """

        def __init__(self, name):
            """ Creates new instance of object which reports to TeamCity test suite results.

            :param name: Suite name.
            :type name: string
            """
            escapedSuiteName = escapeTeamCityMsg(name)
            print u"""##teamcity[testSuiteStarted name='{0}']""".format(escapedSuiteName)

            self.name             = escapedSuiteName
            self.lastTestReporter = None

        def __del__(self):
            self.finish()

        def reportTestStart(self, testName, captureStdOutput = False):
            """ Sends log service message to TeamCity which reports start of test inside current test suite.

            :param testName: Name of test which is starting.
            :type testName: string
            :param captureStdOutput: Indicates that any standard output after current method and before reporter finish
                                     is test output.
            :type captureStdOutput: bool
            """

            if self.name is None:
                raise RuntimeError('Test suite is already reported as finished. No more tests cannot be added.')

            self.__finishTestReporter()
            newTestReporter = TestReporter(self, testName, captureStdOutput)
            self.lastTestReporter = weakref.ref(newTestReporter)
            return newTestReporter

        def __finishTestReporter(self):
            if self.lastTestReporter is not None:
                oldTestReporter = self.lastTestReporter()
                if oldTestReporter is not None:
                    oldTestReporter.finish()
                self.lastTestReporter = None

        def finish(self):
            """ Sends log service message to TeamCity which reports finish of current test suite. """

            if self.name is not None:
                self.__finishTestReporter()
                print u"""##teamcity[testSuiteFinished name='{0}']""".format(self.name)
                self.name = None

    return TestSuiteReporter(suiteName)


########################################################################################################################


def prepareTcRestConnection(teamCityUrl, agentUser, agentPass):
    """ Prepares read-only (verb: GET) REST connection to TeamCity server.

    :param teamCityUrl: URL to TeamCity server.
    :type teamCityUrl: str
    :param agentUser: TeamCity server credentials (user name that will be used to log on).
    :type agentUser: str
    :param agentPass: TeamCity server credentials (user's password that will be used to log on).
    :type agentPass: str
    :return: Delegate function that is able to request data from TeamCity REST end-point.
    :rtype: (string, dict[str, string]) -> Any
    """

    if agentUser is not None and agentUser != '':
        passMgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        passMgr.add_password(None, teamCityUrl, agentUser, agentPass)
        authHandler = urllib2.HTTPBasicAuthHandler(passMgr)
        restOpener = urllib2.build_opener(authHandler)
        urllib2.install_opener(restOpener)

    def prepareGetRequest(restGetRequest, **requestArgs):
        """ Prepares and invokes GET request to REST end-point in TeamCity.

        :param restGetRequest: GET request with optional str.format() placeholders. Request is automatically prefixed
                               with TeamCity server URL and REST API root.
        :type restGetRequest: string
        :param requestArgs: Arguments for GET request in form key = value. Values are automatically quoted/escaped.
        :type requestArgs: dict[str, string]
        :return: Parsed response (from returned JSON).
        """

        getRequestPart = unicode(restGetRequest).format(
            **{k: urllib2.quote(v, safe = '') for (k, v) in requestArgs.iteritems()})
        getRequestFragment = u'/httpAuth/app/rest/{0}'.format(getRequestPart)
        getRequestUrl = urlparse.urljoin(teamCityUrl, getRequestFragment)

        getRequest = urllib2.Request(getRequestUrl, headers = {'Accept': 'application/json'})
        return json.loads(urllib2.urlopen(getRequest).read())

    return prepareGetRequest


########################################################################################################################


def initLogger(loggerName = None, logFileName = None):
    """ Initializes logging capabilities.

    It should not be use more than once on the same loggerName.

    :param loggerName: Name of logger. If not specified, root logger is used.
    :type loggerName: string
    :param logFileName: Path to log file name. If it not specified, only console output will handle log output.
    :type logFileName: string
    :return: Logger object.
    """
    logger          = logging.getLogger(loggerName)
    loggerFormatter = logging.Formatter("%(asctime)-22s %(name)12s: %(levelname)8s:  %(message)s",
                                        datefmt = '[%Y-%m-%d  %H:%M:%S]')
    logger.setLevel(logging.DEBUG)
    if logFileName is not None and logFileName != '':
        logFileHandler = logging.handlers.RotatingFileHandler(logFileName, maxBytes = 5 * 1024 * 1024, backupCount = 2)
        logFileHandler.setFormatter(loggerFormatter)
        logger.addHandler(logFileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(loggerFormatter)
    logger.addHandler(consoleHandler)

    intendedLen = 120
    formatPrefixLen = 48
    restLen = intendedLen - formatPrefixLen
    logger.info('=' * restLen)
    logger.info(str.format('{0:^' + repr(restLen) + 's}', 'START'))
    logger.info('=' * restLen)

    return logger


########################################################################################################################
