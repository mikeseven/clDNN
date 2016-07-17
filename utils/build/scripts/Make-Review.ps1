param([string] $InputServerUri,
      [string] $InputServerUser,
      [string] $InputServerPassword,
      [string] $InputTriggeredBy,
      [string] $InputBranch,
      [string] $InputMergeBranch,
      [string] $InputReviewEnabled,
      [string] $InputReviewId = "",
      [string] $InputServerUserMapJson = "");

function escapeTeamCityMsg([string] $Message) {
    if ([string]::IsNullOrEmpty($Message)) { return ''; }
    return ((((($Message -replace '[''|\[\]]', '|$0') -replace '\n', '|n') -replace '\r', '|r') `
                -replace '\u0085', '|x') -replace '\u2028', '|l') -replace '\u2029', '|p';
}

# ------------------------------------------ Constants ---------------------------------------------

$ServiceRelUri = '/services/json/v1';

# ------------------------------------------ Parameters --------------------------------------------

try {
<#
    [string] $InputServerUri = @'
        %my.build.review.uri%
'@.Trim();
    [string] $InputServerUser = @'
        %my.build.review.logon.user%
'@.Trim();
    [string] $InputServerPassword = @'
        %my.build.review.logon.password%
'@.Trim();
    [string] $InputTriggeredBy = @'
        %teamcity.build.triggeredBy.username%
'@.Trim();
    [string] $InputBranch = @'
        %my.build.review.branch%
'@.Trim();
    [string] $InputMergeBranch = @'
        %my.build.merge.target.branch%
'@.Trim();
    [string] $InputReviewEnabled = @'
        %my.build.review.is_enabled%
'@.Trim();
    [string] $InputReviewId = @'
        %my.build.review.id%
'@.Trim();
    [string] $InputServerUserMapJson = @'
        %my.build.review.useMapJson%
'@.Trim();
#>

    try { [Uri] $ServerUri = New-Object 'Uri' @($InputServerUri, [UriKind]::Absolute); }
    catch { throw 'Invalid SmartBear Collaborator server URL.'; }

    [string] $ServerUser     = $InputServerUser;
    [string] $ServerPassword = $InputServerPassword;
    [string] $TriggeredBy    = $InputTriggeredBy;

    [string] $Branch = git check-ref-format --branch $InputBranch 2>&1;
    if (!$?) { throw 'Invalid branch name (review branch).'; }

    [string] $MergeBranch = git check-ref-format --branch $InputMergeBranch 2>&1;
    if (!$?) { throw 'Invalid branch name (merge base branch).'; }

    [bool] $ReviewEnabled = $InputReviewEnabled -ne '0';

    [long] $ReviewId = -1;
    if (![string]::IsNullOrEmpty($InputReviewId)) {
        if (![long]::TryParse($InputReviewId, [ref] $ReviewId) -or ($ReviewId -lt 0)) {
            throw 'Invalid review ID.';
        }
    }

    [Hashtable] $ServerUserMap = New-Object hashtable; #case-sensitive
    if (![string]::IsNullOrEmpty($InputServerUserMapJson)) {
        try {
            [System.Web.Script.Serialization.JavaScriptSerializer] $Deserializer = `
                New-Object System.Web.Script.Serialization.JavaScriptSerializer;

            $Deserializer.DeserializeObject($InputServerUserMapJson).GetEnumerator() | % {
                    $ServerUserMap["$($_.Key)"] = "$($_.Value)";
                }
        }
        catch { throw 'Invalid user login map (JSON malformed).'; }
    }

# --------------------------------------------------------------------------------------------------

    function getCCollabServiceUri([Uri] $ServerUri) {
        try { return New-Object Uri $ServerUri, $ServiceRelUri; }
        catch { throw 'Cannot create SmartBear Collaborator service end-point.'; }
    }

    function invokeCCollabService([Uri]      $ServiceUri,
                                  [object[]] $InvokeArgs,
                                  [int]      $Timeout = 30) {
        try {
            $ArgsBody = ConvertTo-Json $InvokeArgs -Compress;
            $RequestBody = [System.Text.Encoding]::UTF8.GetBytes($ArgsBody);

            $Response = Invoke-WebRequest $ServiceUri.AbsoluteUri -Body $RequestBody `
                -ContentType 'application/json;charset=utf-8' -Method Post `
                -TimeoutSec $Timeout;
        }
        catch {
            throw 'Communication error with Collaborator service (invalid request).';
        }

        try {
            if ($Response.Headers.ContainsKey("Content-Type")) {
                if($Response.Headers["Content-Type"] -cnotmatch '^[^;/]+/\s*json\s*(;.*)?$') {
                    throw 'Communication error with Collaborator service (invalid response).';
                }
            }
            return ConvertFrom-Json $Response.Content;
        }
        catch {
            throw 'Communication error with Collaborator service (malformed response).';
        }
    }

    function loginCCollab([Uri]    $ServerUri,
                          [string] $User,
                          [string] $Password,
                          [int]    $Timeout = 30) {
        $ServiceUri = getCCollabServiceUri $ServerUri;
        $LoginCommand = @{command = "SessionService.getLoginTicket";
                          args    = @{login    = $User;
                                      password = $Password}};
        $Commands = @($LoginCommand);

        $Response = invokeCCollabService $ServiceUri $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $Commands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        $LoginTicket = $Result[0].loginTicket;

        return New-Object PSObject -Property @{
            ServiceUri  = $ServiceUri;
            AuthCommands = @(@{command = "SessionService.authenticate";
                               args    = @{login  = $User;
                                           ticket = $LoginTicket}},
                             @{command = "SessionService.setMetadata";
                               args    = @{clientName = "TeamCity Interaction Script";
                                           expectedServerVersion = "9.2.9200"}});
        };
    }

    function findUser([PSObject] $LoginTicket,
                      [string]   $UserSsid,
                      [int]      $Timeout = 30) {
        if ($ServerUserMap.ContainsKey($UserSsid)) { $UserSsid = $ServerUserMap[$UserSsid]; }

        $FindUserCommand = @{command = "UserService.findUserByLogin";
                             args    = @{login = $UserSsid}};
        $Commands = @($LoginTicket.AuthCommands; $FindUserCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            return $null;
        }

        return $Result[$Commands.Count - 1];
    }

    function findReview([PSObject] $LoginTicket,
                        [long]     $ReviewId,
                        [int]      $Timeout = 30) {
        $FindReviewCommand = @{command = "ReviewService.findReviewById";
                               args    = @{reviewId = $ReviewId}};
        $Commands = @($LoginTicket.AuthCommands; $FindReviewCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            return $null;
        }

        return $Result[$Commands.Count - 1];
    }

    function createBuildId() {
        return [Guid]::NewGuid() -replace '-', '';
    }

    function createReview([PSObject] $LoginTicket,
                          [PSObject] $Author,
                          [string]   $Title = '',
                          [int]      $Timeout = 30) {
        $FindReviewCommand = @{command = "ReviewService.findReviewById";
                               args    = @{reviewId = $ReviewId}};
        $Commands = @($LoginTicket.AuthCommands; $FindReviewCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            return $null;
        }

        return $Result[$Commands.Count - 1];
    }



    if (!$ReviewEnabled) {
        Write-Host "##teamcity[buildStatus status='SUCCESS' text='Review creation disabled. Build skipped.']";
        return;
    }

    $CCollabTicket = loginCCollab $ServerUri $ServerUser $ServerPassword;
    $CCollabAuthor = findUser $CCollabTicket $TriggeredBy;
    if ($CCollabAuthor -eq $null) {
        throw 'Cannot create / update review. User who trigger review build is not registered in SmartBear Collaborator.';
    }

    if ($ReviewId -ge 0) {
        $CCollabReview = findReview $CCollabTicket $ReviewId;
        if ($CCollabReview -eq $null) {
            throw 'Cannot update review. Review with specified review ID cannot be found.';
        }
    }
}
catch {
    [string] $ErrMsg = escapeTeamCityMsg $_.Exception.Message;

    Write-Host ("##teamcity[buildStatus status='FAILURE' text='{0}']" -f $ErrMsg);
    throw;
}