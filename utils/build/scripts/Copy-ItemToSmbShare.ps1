<#

    .SYNOPSIS
        Copies items to Windows/Samba share.

    .DESCRIPTION
        Copies items from source directory to destination directory in specified Windows/Samba share.

        The script allows to specify plain credentials for share.

        The script supports TeamCity console.

    .PARAMETER Path
        List of items to copy. Wildards are allowed.

    .PARAMETER LiteralPath
        List of items to copy. All paths are treated literally. No wildcards resolution is done.

    .PARAMETER SharePath
        Path to Windows/Samba share that is root of access (it will be temporary mapped to drive).

    .PARAMETER DestinationPath
        Destination path which can be file or directory. If the path ends with backslash (or slash)
        the destination is automatically recognized as directory. Otherwise, destination is treated
        as file.

        The destination is relative to SharePath (which is treated as root). All directories that
        do not exist in destination path will be created.

        You can always treat destination as directory with -Container switch.

    .PARAMETER Container
        Treats destination as directory, even without backslash/slash suffix.

    .PARAMETER User
        User name used to access share.

    .PARAMETER Password
        Password used to access share.

    .PARAMETER Force
        Forces copying, even if destination files exist and there are read-only (see Copy-Item -Force switch).

    .PARAMETER Recurse
        Does recursive copy (see Copy-Item -Recurse switch).

    .LINK
        Copy-Item
#>

[CmdletBinding(DefaultParameterSetName = 'wildcards')]
param(
    [Parameter(Mandatory = $true, Position = 0, ParameterSetName = 'wildcards',
               ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true,
               HelpMessage = 'Specify source items to copy (wildcards allowed).')]
    [Alias('SourcePath', 'SrcPath')]
    [AllowEmptyCollection()]
    [string[]] $Path,

    [Parameter(Mandatory = $true, ParameterSetName = 'literal',
               ValueFromPipelineByPropertyName = $true,
               HelpMessage = 'Specify source items to copy (literal paths).')]
    [AllowEmptyCollection()]
    [string[]] $LiteralPath,

    [Parameter(Mandatory = $true, Position = 1,
               ValueFromPipelineByPropertyName = $true,
               HelpMessage = 'Specify path to destination Windows/Samba share.')]
    [Alias('SambaSharePath', 'SmbSharePath', 'SambaPath', 'SmbPath')]
    [string] $SharePath,

    [Parameter(Mandatory = $false, Position = 2,
               ValueFromPipelineByPropertyName = $true)]
    [Alias('DstPath')]
    [string] $DestinationPath = '',

    [Parameter(Mandatory = $false)]
    [switch] $Container,

    [Parameter(Mandatory = $false,
               ValueFromPipelineByPropertyName = $true)]
    [string] $User = $null,

    [Parameter(Mandatory = $false,
               ValueFromPipelineByPropertyName = $true)]
    [string] $Password = $null,

    [Parameter(Mandatory = $false)]
    [switch] $Force,

    [Parameter(Mandatory = $false)]
    [switch] $Recurse
);
begin {
    # Resizes Powershell console to at least 500 characters.
    function resizeTcPsConsole() {
        if (Test-Path 'env:TEAMCITY_VERSION') {
            try {
                $RawUI = (Get-Host).UI.RawUI
                $CurrentUIWidth = $RawUI.MaxPhysicalWindowSize.Width
                $RawUI.BufferSize = New-Object 'Management.Automation.Host.Size' @([Math]::Max($CurrentUIWidth, 500), $RawUI.BufferSize.Height)
                $RawUI.WindowSize = New-Object 'Management.Automation.Host.Size' @($CurrentUIWidth, $RawUI.WindowSize.Height)
            }
            catch {}
        }
    }

    resizeTcPsConsole;

    $ScriptVerbose = [bool] $PSBoundParameters['Verbose'].IsPresent;

    Write-Verbose 'Starting upload script.';
}
process {
    # Constructing credential objects (if credentials are specified).
    Write-Verbose 'Preparing credentials.';
    if (![string]::IsNullOrEmpty($Password)) { $SecPassword = ConvertTo-SecureString $Password -AsPlainText -Force; }
    else { $SecPassword = New-Object SecureString; }

    if (![string]::IsNullOrEmpty($User)) { $ShareCredential = New-Object PSCredential $User, $SecPassword; }
    else { $ShareCredential = $null; }


    # Preparing parameters for creating temporary drive.
    $NewDriveParams = @{};
    if ($ShareCredential -ne $null) { $NewDriveParams['Credential'] = $ShareCredential; }


    # Preparing parameters for copy operation.
    $CopyParams = @{
        'Force'   = $Force.IsPresent;
        'Recurse' = $Recurse.IsPresent;
    };

    if ($PSCmdlet.ParameterSetName -eq 'literal') { $CopyParams['LiteralPath'] = $LiteralPath; }
    else { $CopyParams['Path'] = $Path; }


    # Create temporary drive to share.
    Write-Verbose 'Creating temporary drive for share.';
    $TmpDrive     = $null;
    $TmpDriveName = 'tmpShare1';
    try {
        $TmpDrive = New-PSDrive $TmpDriveName FileSystem $SharePath @NewDriveParams -Verbose:$ScriptVerbose -ErrorAction Stop;

        $TmpDstPath = Join-Path "${TmpDriveName}:" $DestinationPath;
        if ($Container.IsPresent -or $TmpDstPath.EndsWith([System.IO.Path]::DirectorySeparatorChar))  {
            $TmpDstDir = $TmpDstPath;
        }
        else {
            $TmpDstDir = Split-Path $TmpDstPath;
        }
        $CopyParams['Destination'] = $TmpDstPath;

        Write-Verbose 'Checking and creating destination directories.';
        try { mkdir $TmpDstDir -Force -Verbose:$ScriptVerbose -ErrorAction Stop | Out-Null; } catch {}
        Write-Verbose 'Copying files to destination.';
        if ($Path.Count -gt 0) {
            Copy-Item @CopyParams -Verbose:$ScriptVerbose -ErrorAction Stop;
        }
    }
    catch {
        Write-Verbose 'Terminating with error.';
        Write-Error -Exception $_;
        $PSCmdlet.ThrowTerminatingError($_);
    }
    finally {
        if ($TmpDrive -ne $null) { Remove-PSDrive $TmpDrive; }
    }
}
end {
    Write-Verbose 'Completed upload script.';
}
