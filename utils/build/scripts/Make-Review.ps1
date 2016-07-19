param([string] $InputServerUri,
      [string] $InputServerUser,
      [string] $InputServerPassword,
      [string] $InputTriggeredBy,
      [string] $InputBranchRef,
      [string] $InputMergeBranch,
      [string] $InputDevProcessMode,
      [string] $InputReviewId = "",
      [string] $InputReviewTitle = "",
      [string] $InputServerUserMapJson = "",
      [string] $InputBuildNo = "",
      [string] $InputTeamCityUri = "https://teamcity01-igk.devtools.intel.com");

$CPC = [char] 0x25;

function escapeTeamCityMsg([string] $Message) {
    if ([string]::IsNullOrEmpty($Message)) { return ''; }
    return ((((($Message -replace '[''|\[\]]', '|$0') -replace '\n', '|n') -replace '\r', '|r') `
                -replace '\u0085', '|x') -replace '\u2028', '|l') -replace '\u2029', '|p';
}

function writeTcSuccessStatus([string] $Message) {
    [string] $Msg = escapeTeamCityMsg $Message;
    Write-Host ("##teamcity[buildStatus status='SUCCESS' text='{0}']" -f $Msg);
}

function writeTcFailureStatus([string] $Message) {
    [string] $Msg = escapeTeamCityMsg $Message;
    Write-Host ("##teamcity[buildStatus status='FAILURE' text='{0}']" -f $Msg);
}

function updateTcParameter([string] $ParamName, [string] $Message) {
    [string] $PName = escapeTeamCityMsg $ParamName;
    [string] $Msg   = escapeTeamCityMsg $Message;
    Write-Host ("##teamcity[setParameter name='{0}' value='{1}']" -f @($PName, $Msg));
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
    [string] $InputBranchRef = @'
        %my.build.review.branch_ref%
'@.Trim();
    [string] $InputMergeBranch = @'
        %my.build.merge.branch%
'@.Trim();
    [string] $InputDevProcessMode = @'
        %my.build.dev_process.mode%
'@.Trim();
    [string] $InputReviewId = @'
        %my.build.review.id%
'@.Trim();
    [string] $InputReviewTitle = @'
        %my.build.review.title%
'@.Trim();
    [string] $InputServerUserMapJson = @'
        %my.build.review.userMapJson%
'@.Trim();
    [string] $InputBuildNo = @'
        %build.number%
'@.Trim();
    [string] $InputTeamCityUri = @'
        %teamcity.serverUrl%
'@.Trim();
#>

    try { [Uri] $ServerUri = New-Object 'Uri' @($InputServerUri, [UriKind]::Absolute); }
    catch { throw 'Invalid SmartBear Collaborator server URL.'; }

    [string] $ServerUser     = $InputServerUser;
    [string] $ServerPassword = $InputServerPassword;
    [string] $TriggeredBy    = $InputTriggeredBy;

    [string] $Branch = git check-ref-format --branch $InputBranchRef 2>&1;
    if (!$?) { throw 'Invalid branch name (review branch).'; }

    [string] $MergeBranch = git check-ref-format --branch $InputMergeBranch 2>&1;
    if (!$?) { throw 'Invalid branch name (merge base branch).'; }

    [string] $DevProcessMode = $InputDevProcessMode;
    if ($DevProcessMode -cnotmatch '^(RCoU|BTR|BTO)$') {
        throw 'Invalid development process / review mode.';
    }

    [long] $ReviewId = -1;
    if (![string]::IsNullOrEmpty($InputReviewId)) {
        if (![long]::TryParse($InputReviewId, 'Integer', [CultureInfo]::InvariantCulture, [ref] $ReviewId) -or ($ReviewId -lt 0)) {
            throw 'Invalid review ID.';
        }
    }
    switch -casesensitive ($DevProcessMode) {
        'RCoU' {}                                                                         # Do nothing if mode is: Create / Update Review
        'BTR'  { if ($ReviewId -lt 0) { throw 'Review ID is mandatory in this mode.'; } } # Mandatory review ID if mode is: Re-run Build and Tests
        'BTO'  { $ReviewId = -1; }                                                        # Ignore review ID if mode is: Run Build and Tests w/o Review
    }

    [string] $ReviewTitle = $InputReviewTitle;

    [Hashtable] $ServerUserMap = New-Object Hashtable; #case-sensitive
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

    [string] $BuildNo = $InputBuildNo;

    try { [Uri] $TeamCityUri = New-Object 'Uri' @($InputTeamCityUri, [UriKind]::Absolute); }
    catch { throw 'Invalid TeamCity server URL.'; }

# --------------------------------------------------------------------------------------------------

    function getFailMsgStart([string] $DPMode,
                             [long]   $ReviewId) {
        switch -casesensitive ($DPMode) {
            'RCoU' { if ($ReviewId -lt 0) { return 'Cannot create review.'; }
                     else                 { return 'Cannot update review.'; } }
            'BTR'  { return 'Cannot re-run build and tests.'; }
            'BTO'  { return 'Cannot run build and tests.'; }
        }
    }

    function getCCollabServiceUri([Uri] $ServerUri) {
        try { return New-Object Uri $ServerUri, $ServiceRelUri; }
        catch { throw 'Cannot create SmartBear Collaborator service end-point.'; }
    }

    function invokeCCollabService([Uri]      $ServiceUri,
                                  [object[]] $InvokeArgs,
                                  [int]      $Timeout = 30) {
        try {
            $ArgsBody = ConvertTo-Json $InvokeArgs -Compress -Depth 8;
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

    function findCurrentUser([PSObject] $LoginTicket,
                             [int]      $Timeout = 30) {
        $FindUserCommand = @{command = "UserService.getSelfUser"};
        $Commands = @($LoginTicket.AuthCommands; $FindUserCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            throw 'Communication error with Collaborator service (timeout).';
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
                          [PSObject] $Creator,
                          [string]   $Title = '',
                          [int]      $Timeout = 30) {
        $CreateReviewCommand = @{command = "ReviewService.createReview";
                                 args    = @{creator = $Creator.login}};
        if (![string]::IsNullOrEmpty($Title)) {
            $CreateReviewCommand['args']['title'] = $Title;
        }

        $Commands = @($LoginTicket.AuthCommands; $CreateReviewCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            throw 'Communication error with Collaborator service (create review failed).';
        }
        return [long] $Result[$Commands.Count - 1].reviewId;
    }

    function assignReviewAuthor([PSObject] $LoginTicket,
                                [long]     $ReviewId,
                                [PSObject] $Author,
                                [int]      $Timeout = 30) {
        if ($Author.login -eq $Creator.login) {
            return;
        }

        $AssignCommand = @{command = "ReviewService.setAssignments";
                           args    = @{reviewId    = $ReviewId;
                                       assignments = @(
                                           @{user = $Author.login;
                                             role = 'AUTHOR'}
                                       )}};
        $Commands = @($LoginTicket.AuthCommands; $AssignCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;
        ConvertTo-Json $Response -Depth 8;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        return ($Result.Count -eq $Commands.Count);
    }

    function unassignReviewCreator([PSObject] $LoginTicket,
                                   [long]     $ReviewId,
                                   [PSObject] $Creator,
                                   [PSObject] $Author,
                                   [int]      $Timeout = 30) {
        if ($Author.login -eq $Creator.login) {
            return;
        }

        $UnassignCommand = @{command = "ReviewService.removeAssignments";
                             args    = @{reviewId    = $ReviewId;
                                         assignments = @(
                                             @{user = $Creator.login;
                                               role = 'AUTHOR'}
                                         )}};
        $Commands = @($LoginTicket.AuthCommands; $UnassignCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;
        ConvertTo-Json $Response -Depth 8;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        return ($Result.Count -lt $Commands.Count);
    }

    function addReviewComment([PSObject] $LoginTicket,
                              [long]     $ReviewId,
                              [string]   $Comment,
                              [int]      $Timeout = 30) {
        $AssignCommand = @{command = "ReviewService.createReviewComment";
                           args    = @{reviewId = $ReviewId;
                                       comment  = $Comment }};
        $Commands = @($LoginTicket.AuthCommands; $AssignCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            Write-Warning 'Communication error with Collaborator service (comment add failed).';
            return [long] -1;
        }
        return [long] $Result[$Commands.Count - 1].commentId;
    }

    function findReviewComments([PSObject] $LoginTicket,
                                [long]     $ReviewId,
                                [PSObject] $Creator,
                                [int]      $Timeout = 30) {
        $FindCommentsCommand = @{command = "ReviewService.getComments";
                                 args    = @{reviewId = $ReviewId;
                                             user     = $Creator.login;
                                             type     = 'USER' 
                                             }};
        $Commands = @($LoginTicket.AuthCommands; $FindCommentsCommand);

        $Response = invokeCCollabService $LoginTicket.ServiceUri `
                          $Commands $Timeout;

        $Result = @($Response.result | ? { ![object]::Equals($_, $null) });
        if ($Result.Count -lt $LoginTicket.AuthCommands.Count) {
            throw 'Communication error with Collaborator service (login failed).';
        }
        if ($Result.Count -lt $Commands.Count) {
            return @();
        }
        return $Result[$Commands.Count - 1].comments;
    }

    function addReviewGitDiffFiles([Uri]    $ServerUri,
                                   [string] $User,
                                   [string] $Password,
                                   [long]   $ReviewId,
                                   [string] $MergeTargetBranch,
                                   [string] $ReviewBranchRef = 'HEAD',
                                   [string] $UploadComment = '') {
        $DiffSpecification = "origin/$MergeTargetBranch...$ReviewBranchRef";

        if ([string]::IsNullOrEmpty($UploadComment)) {
            $UComment = git show -s --format="${CPC}s" 2>&1;
            if (!$?) {
                $UComment = $DiffSpecification;
            }
        } else { $UComment = $UploadComment; }

        ccollab --url $ServerUri --user $User --password $Password --non-interactive --no-browser addgitdiffs --upload-comment $UComment $ReviewId $DiffSpecification 2>&1 | ? { $_ -is [string] };
        if ($LASTEXITCODE -ne 0) {
            throw 'Creating / updating review failed. File upload failed.'
        }
    }

# --------------------------------------------------------------------------------------------------

    # [BTO] No review path...
    if ($DevProcessMode -ceq 'BTO') {
        writeTcSuccessStatus 'Review creation disabled. Build skipped.';
        return;
    }

    $CCollabTicket = loginCCollab $ServerUri $ServerUser $ServerPassword;
    $CCollabDemon  = findCurrentUser $CCollabTicket;
    $CCollabAuthor = findUser $CCollabTicket $TriggeredBy;
    if ($CCollabAuthor -eq $null) {
        $MsgStart = getFailMsgStart $DevProcessMode $ReviewId;
        throw ('{0} User who triggered review build had not been registered in SmartBear Collaborator.' -f $MsgStart);
    }

    # [RCoU, BTR] Update review path...
    if ($ReviewId -ge 0) {
        $CCollabReview = findReview $CCollabTicket $ReviewId;
        if ($CCollabReview -eq $null) {
            $MsgStart = getFailMsgStart $DevProcessMode $ReviewId;
            throw ('{0} Review with specified review ID cannot be found.' -f $MsgStart);
        }
        $BuildComments = findReviewComments $CCollabTicket $ReviewId $CCollabDemon;
        $IsInitCommentAdded = @($BuildComments.text | ? { $_ -cmatch '\n\s*-\s*Build\s+system\s+link\s*:' }).Count -gt 0;
    }
    # [RCoU] Create review path...
    else {
        $ReviewId = createReview $CCollabTicket $CCollabDemon $ReviewTitle;
        if (!(assignReviewAuthor $CCollabTicket $ReviewId $CCollabAuthor)) {
            throw 'Communication error with Collaborator service (author assign failed).';
        }
        $IsInitCommentAdded = $false;
    }

    if (!$IsInitCommentAdded) {
        addReviewComment $CCollabTicket $ReviewId "[Build Agent]`n - Build system link: $($TeamCityUri.AbsoluteUri)" | Out-Null;
    }

    # [RCoU] Update review files...
    if ($DevProcessMode -ceq 'RCoU') {
        addReviewGitDiffFiles $ServerUri $ServerUser $ServerPassword $ReviewId $MergeBranch;
        unassignReviewCreator $CCollabTicket $ReviewId $CCollabDemon $CCollabAuthor | Out-Null;
    }

    updateTcParameter 'my.build.review.id' $ReviewId.ToString([CultureInfo]::InvariantCulture);
}
catch {
    writeTcFailureStatus $_.Exception.Message;
    throw;
}