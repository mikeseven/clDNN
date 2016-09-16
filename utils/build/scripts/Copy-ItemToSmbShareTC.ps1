# INTEL CONFIDENTIAL
# Copyright 2016 Intel Corporation
#
# The source code contained or described herein and all documents related to the source code ("Material") are owned by
# Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual property right is granted to
# or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel
# in writing.
#
#
# For details about script please contact following people:
#  * [Version: 1.0] Walkowiak, Marcin <marcin.walkowiak@intel.com>


<#
    .SYNOPSIS
        Copies items to Windows/Samba share (TeamCity wrapper for script).

    .DESCRIPTION
        Copies items from source directory to destination directory in specified Windows/Samba share.

        The script allows to specify plain credentials for share.

        This is TeamCity wrapper for Copy-ItemToSmbShare.ps1 script. The script provides the following:
         * Better support for TeamCity build console.
         * Disabled pipelining (workaround for TeamCity 9.x bug where process section is not called).

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
               HelpMessage = 'Specify source items to copy (wildcards allowed).')]
    [Alias('SourcePath', 'SrcPath')]
    [AllowEmptyCollection()]
    [string[]] $Path,

    [Parameter(Mandatory = $true, ParameterSetName = 'literal',
               HelpMessage = 'Specify source items to copy (literal paths).')]
    [AllowEmptyCollection()]
    [string[]] $LiteralPath,

    [Parameter(Mandatory = $true, Position = 1,
               HelpMessage = 'Specify path to destination Windows/Samba share.')]
    [Alias('SambaSharePath', 'SmbSharePath', 'SambaPath', 'SmbPath')]
    [string] $SharePath,

    [Parameter(Mandatory = $false, Position = 2)]
    [Alias('DstPath')]
    [string] $DestinationPath = '',

    [Parameter(Mandatory = $false)]
    [switch] $Container,

    [Parameter(Mandatory = $false)]
    [string] $User = $null,

    [Parameter(Mandatory = $false)]
    [string] $Password = $null,

    [Parameter(Mandatory = $false)]
    [switch] $Force,

    [Parameter(Mandatory = $false)]
    [switch] $Recurse
);


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

Write-Verbose ('=' * 111);
Write-Verbose 'Resizing Teamcity console (powershell runner console) to at least 500 characters per line.';

# Get scripts location.
if (![string]::IsNullOrEmpty($PSCmdlet.MyInvocation.MyCommand.Path)) {
    $ScriptDir = Split-Path $PSCmdlet.MyInvocation.MyCommand.Path;
}
else {
    $ScriptDir = (Get-Location).Path;
}
Write-Verbose ('Locating script directory: "{0}"' -f $ScriptDir);

# Get scipt path.
$ScriptFile = 'Copy-ItemToSmbShare.ps1';

if (Split-Path -IsAbsolute $ScriptFile) {
    $ScriptFilePath = (Resolve-Path $ScriptFile -ErrorAction Stop).Path
}
else {
    $ScriptFilePath = Join-Path $ScriptDir $ScriptFile -Resolve -ErrorAction Stop;
}
Write-Verbose ('Executing script file: "{0}"' -f $ScriptFilePath);
Write-Verbose ('=' * 111);

& $ScriptFilePath @PSBoundParameters;

Write-Verbose ('=' * 111);