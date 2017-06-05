#########################################################################################
#
#    Python script to populate the xml variable in quickbuild to generate html reports
#
import sys
import argparse
import json
import os.path
import logging
import logging.handlers
import textwrap
import webapi_new
from datetime import datetime

log = None  # Global log handle


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


def init_logging(fname='kw_html_results.log'):
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


def fetchIssueCount(url, ltoken, project, view, searchQuery):
    """ Do a HTTP request to get data from the Klocwork server """
    req = {"project": project, "action": "search", "view": view, "query": searchQuery}
    api = webapi_new.Api(url, ltoken)
    try:
        response = api.send_request(req)
        if response:
            count = 0
            for record in response:
                count += 1
            return count
    except:
        log.exception("fetchIssueCount(), failed to get data")

    return -1


def main(args):
    """ For each component in the JSON file, get the Klocwork statistics for the build and generate HTML table of the results """
    status = 0
    init_logging()

    # Load the projects JSON data
    projectData = load_json(args.jsonFilePath)

    # Create a summary object to hold details about total critical issues and
    # the components with critical issues and new critical issues
    summaryData = {}

    # Create a totals object to hold the number of issues in each "pattern"
    issueTypeTotals = {}
    issueTypeTotals['x32'] = {}
    issueTypeTotals['x64'] = {}

    # Generate a report for each architecture requested
    for architecture in args.reportArchitecture:
        # Loop through each issue type that is stored in the 'patterns' list in the JSON and set the value to 0
        # E.g. "New Critical Issues", "Not a Problem", "Ignore", ...    
        for issueType in projectData["patterns"]:
            issueTypeTotals[architecture][issueType["name"]] = 0

        # Initialize the summary data
        summaryData['new_critical_components_list_' + architecture] = []
        summaryData['critical_components_list_' + architecture] = []
        summaryData['total_critical_issues_' + architecture] = 0

        # Create the header for the HTML
        html = textwrap.dedent("""\
            <link rel='stylesheet' type='text/css' href='/styles/package.css' />
            <script type='text/javascript' src='/package.js'></script>
            <h1>Klocwork Report Windows</h1>
            <h3>Build Date: {1}</h3>
            <h3>Build Version: {2}</h3>
            <div class='panel'>
            <table class='oddeven' style='width:75%'>
            <thead><tr bgcolor='#C0C0C0'><td>Project/View</td><td>New Critical Issues</td><td>Total Critical Issues</td><td>Fix In Next/Later Release</td><td>Not a Problem</td><td>Ignore</td><td>Banned Functions</td></tr></thead>
            """.format(args.klocworkApiUrl, args.buildDate, args.buildVersion))

        # Loop through the projects and add each project and sub-component
        for project in projectData["projects"]:
            # Fetch data for the current architecture (32-bit or 64-bit) for the report
            if project["architecture"] == architecture:

                # project["name"] will be 'main_igfxdev_lh32_kmd' or 'main_igfxdev_lh32_dx10', ...
                log.info("Fetching data for Klocwork project " + project["name"])

                # Add in the HTML for each project
                html += "<thead><tr><td colspan='7'>" + project["name"] + "</td></thead>\n"

                # For each sub-component, fetch the scan result details
                for component in project["views"]:
                    # Example component:
                    # { "display": "kmd-render", "id": "4", "issues": [ ... ] }
                    log.info("    Sub-component: " + component["name"])

                    # Create a list to store the issues and their counts
                    component["issues"] = []

                    # Loop through all the "patterns" which are just issue names and their search strings
                    # Fetch the number of issues in each "pattern" and store them for the component.
                    for issueType in projectData["patterns"]:
                        # Example Issue
                        # { "count": 0, "name": "New Critical Issues", "searchstring": "severity:Critical status:Analyze,+Fix state:New" }

                        # Fetch the count.  Retry 3 times if it fails (-1 count means it failed)
                        retryCount = 0
                        count = -1
                        while count < 0 and retryCount < 3:
                            count = fetchIssueCount(args.klocworkApiUrl, args.klocworkApiToken, project["name"],
                                                    component["name"], issueType["searchstring"])
                            retryCount += 1

                        # Store the count and name details in the dictionary object
                        issueDetails = {}
                        issueDetails["name"] = issueType["name"]
                        issueDetails["searchstring"] = issueType["searchstring"]
                        issueDetails["count"] = count

                        # Only add the count to the total if it's > 0.  Count will be -1 if there was an error fetching the count.
                        if count > 0:
                            if 'New Critical Issues' or 'Total Critical Issues' in issueType["name"]:
                                status = -3
                            issueTypeTotals[architecture][issueType["name"]] += count

                            # Also log summary details
                            if issueType["name"] == "New Critical Issues":
                                # This means there are new critical issues for a component.  If this component
                                # is not already in the list of components with new critical issues, then
                                # add it to the list.
                                if not component['display'] in summaryData[
                                            'new_critical_components_list_' + architecture]:
                                    summaryData['new_critical_components_list_' + architecture].append(
                                        component['display'])
                            elif issueType["name"] == "Total Critical Issues":
                                # Add up all the critical issues
                                summaryData['total_critical_issues_' + architecture] += count

                                # Also keep track of all the components which have critical issues.  If this component
                                # is not already in the list of components with critical issues, then
                                # add it to the list.
                                if not component['display'] in summaryData['critical_components_list_' + architecture]:
                                    summaryData['critical_components_list_' + architecture].append(component['display'])

                        log.info("        " + issueType["name"] + ": " + str(count))

                        component["issues"].append(issueDetails)
                        # At this point, component should look like this:
                        # {
                        #     "id": "4",
                        #     "name": "kmd-render",
                        #     "issues": [
                        #         {
                        #             "name": "New Critical Issues",
                        #             "count": 4,
                        #         },
                        #         {
                        #             "name": "Total Critical Issues",
                        #             "count": 8,
                        #         },
                        #         ...
                        #     ]
                        # },

                    html += "<tr><td>" + component["name"] + "</td>"

                    for issueType in component["issues"]:
                        # Write out the HTML table cell for this result

                        # Construct the URL.  First include the project name, e.g. 'main_igfxdev_lh32_kmd'
                        # NOTE: There might be a different name used to contruct the URL to the issues
                        # from the name used to fetch data from the Klocwork API.
                        if "internal_name" in project:
                            href = args.klocworkBaseUrl + "#issuelist_goto:project=" + project["internal_name"].replace(" ", "_")
                        else:
                            href = args.klocworkBaseUrl + "#issuelist_goto:project=" + project["name"].replace(" ", "_")
                        # Next add in the search query part of the URL.  Replace ':' with encoding of the character.
                        # Also replace white space with '+'
                        # Also replace ',+' with the encoding of those characters
                        href += ",searchquery=" + issueType["searchstring"].replace(":", "%253A").replace(" ",
                                                                                                          "+").replace(
                            ",+", "%252C%252B")
                        # Next add in the column sorting
                        href += ",sortcolumn=id,sortDirection=ASC,start=0"
                        # Next add in the view ID
                        href += ",view_id=" + component["id"]

                        if issueType["count"] < 0:
                            html += "<td>ERROR</td>"
                            status = -1
                        else:
                            html += "<td><a href=" + href + " target=\"_new\">" + str(issueType["count"]) + "</a></td>"
                    # Close out the row
                    html += "</tr>\n"

        # Add a final Totals row
        html += "<thead><tr bgcolor='#C0C0C0'><td>Totals</td>"
        for issueType in projectData["patterns"]:
            html += "<td>" + str(issueTypeTotals[architecture][issueType["name"]]) + "</td>"
        html += "</thead>\n"

        # Close out the table
        html += "</table>"

        # Write the HTML to file
        with open(args.outputHtmlFileName + '_' + architecture + '.html', 'w') as f:
            f.write(html)

        # If the summary data has empty lists, append the string 'none'
        if summaryData['new_critical_components_list_' + architecture] == []:
            summaryData['new_critical_components_list_' + architecture].append('none')
        if summaryData['critical_components_list_' + architecture] == []:
            summaryData['critical_components_list_' + architecture].append('none')

    # Write the updated JSON data with counts to file
    save_json(args.outputDetailsJsonFile, projectData["projects"])

    # Write the summary JSON data to file
    save_json(args.outputSummaryJsonFile, summaryData)

    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonFilePath", type=str, help="path to the json file with the component mappings",
                        default="kw_win_projects.json", required=False)
    parser.add_argument("--klocworkApiUrl", type=str,
                        help="klocwork web server api url, e.g. https://klocwork-igk1.devtools.intel.com:8075/review/api",
                        default="https://klocwork-igk1.devtools.intel.com:8075/review/api", required=False)
    parser.add_argument("--klocworkApiToken", type=str, help="klocwork ltoken file path",
                        default="C:\Users\sys_gpudnna1\.klocwork\ltoken", required=False)
    parser.add_argument("--outputHtmlFileName", type=str, help="output HTML file", required=True)
    parser.add_argument("--reportArchitecture", type=str, nargs='*',
                        help="Specify the reports that you want to generate", required=True)
    parser.add_argument("--buildVersion", type=str, help="%my.build.name%_%build.counter%", required=True)
    parser.add_argument("--buildDate", type=str, help="date the build was started, e.g. 08/29/15",
                        default=datetime.today(), required=False)
    parser.add_argument("--outputDetailsJsonFile", type=str, help="output json file with table data",
                        default="projectData.json", required=False)
    parser.add_argument("--outputSummaryJsonFile", type=str, help="output json file with summary data",
                        default="summaryData.json", required=False)
    parser.add_argument("--klocworkBaseUrl", type=str, help="output file",
                        default="https://klocwork-igk1.devtools.intel.com:8075/review/insight-review.html", required=False)

    args = parser.parse_args()

    success = main(args)
    if success == -3:
        print('\nExiting with status ' + str(success) + ' !!!CRITICALS FOUND!!!')
    else:
        print('\nExiting with status ' + str(success))
    sys.exit(success)
