<#
    .SYNOPSIS
        Synchronize current sources repository with Open Source repository.

    .DESCRIPTION
        Synchronizes files between from one repository to another. Source repository is usually
        internal code repository; destination repository - Open Source repository, where
        drop should be synchronized.

        Synchronization is done according to rules specified in configuration file.

        Configuration file has JSON-like syntax. It starts with member "sync" (no encapsulating
        object is allowed). "sync" member is an object that may contain following members:
         -- "copy"
         -- "update_version"
         -- "process_file"
         -- "format_file"
         -- "chmod_staged" (change mode of files in VCS - stages files if necessary)
        Any other member is ignored.

        See sample Sync-WithOpenSource.config.template for details which sections are allowed
        in each member. 

    .PARAMETER SrcPath
        Path to source repository (usually internal code repository). It should point to
        repository root.

        If it is not specified or empty string is specified, the default source directory
        is calculated (fixed to the location of script file).
        
    .PARAMETER DstPath
        Path to destination repository (usually Open Source repository). It should point to
        repository root.
        
        If it is not specified or empty string is specified, the destination path is
        set to source path (repository is converted in-place; useful when you want to
        prepare branches/streams/junctions on internal repository).

    .PARAMETER CfgPath
        Path to configuration file. It should point to the special JSON-like file with
        description of synchronization process.

        If it is not specified or empty string is specified, the default configuration file
        is used (named as script with .config extension, located in the same directory as
        script).

    .PARAMETER UpdateSrcVersion
        Updates version files at source repository as well. It is implied when
        source and destination are the same.

    .PARAMETER VcsServer
        URL/URI to server/port/entrypoint of version control system.

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

    .PARAMETER VcsCredential
        Credentials for accessing server/port/entrypoint of version control system.

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

        When this parameter is specified and non-null, the -PromptVcsCredential is ignored.

    .PARAMETER PromptVcsCredential
        Prompts for credentials for accessing server/port/entrypoint of version
        control system. Usually it is in the form of dialog with default user name for VCS
        specified.

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

        When VcsCredential parameter is specified and non-null, current parameter is ignored.

    .PARAMETER VcsWorkspace
        Expected workspace in VCS. If it is specified, the destination location is
        checked to see whether it is contained in specified workspace. If location
        is outside workspace or workspace is absent in VCS, the confirmation dialog
        will be shown (whether to continue).

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

    .PARAMETER VcsBranch
        Expected branch/stream in VCS. If it is specified, the destination location is
        checked to see whether it is switch to selected branch/stream. If location
        is not set to use specified branch/stream or branch/stream is absent in VCS,
        the confirmation dialog will be shown (whether to continue).

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

    .PARAMETER VcsChangelist
        Changelist where to stage files in VCS.

        If 'New' is selected, the new changelist will be created for staged files.
        If 'Default' is selected, any files will be staged in default changelist.
        If number is specified, the files will be staged on selected changelist.

        Allowed values: 'New', 'Default', <any non-negative integral number>

        This parameter is used only by "chmod_staged" section in configuration; otherwise,
        it is ignored.

        This parameter is only used by specific types of VCS and ignored by the rest.

    .PARAMETER PassThru
        Return list of files to synchronize. Each item has at least following members:
        -- 'SyncWosMode'           - how the file should be/was synchronized.
        -- 'SyncWosDstRootRelPath' - path relative to destination (of item target).
        -- 'SyncWosRootRelPath'    - path relative to source (of source item).
        -- 'PSIsContainer'         - indicates that item is container.

    .PARAMETER Force
        Force overwrite/removal of destination items (even if they are read-only, hidden
        or write-protected).
        
        Also forces file staging even if selected workspace/branch is different than selected
        (for "chmod_staged" sections).
                
    .PARAMETER Silent
        Suppresses any confirmation. Implies -Force.

    .OUTPUTS
        None

        If -PassThru, list of items to synchronize (extended [System.Io.DirectoryInfo], [System.Io.FileInfo], etc.).
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
[OutputType([System.Io.DirectoryInfo], [System.Io.FileInfo])]
param(
    [Parameter(Position = 0, Mandatory = $false)]
    [Alias('SourcePath')]
    [string] $SrcPath = '',
    [Parameter(Position = 1, Mandatory = $false)]
    [Alias('DestPath', 'DestinationPath')]
    [string] $DstPath = '',
    [Parameter(Position = 2, Mandatory = $false)]
    [Alias('ConfigPath', 'ConfigurationPath', 'ConfigurationFilePath')]
    [string] $CfgPath = '',
    [Parameter(Mandatory = $false)]
    [switch] $UpdateSrcVersion,

    [Parameter(Mandatory = $false)]
    [Alias('Server')]
    [string] $VcsServer = '',
    [Parameter(Mandatory = $false)]
    [Alias('Credential')]
    [System.Management.Automation.PSCredential] $VcsCredential = $null,
    [Parameter(Mandatory = $false)]
    [Alias('PromptCredential')]
    [switch] $PromptVcsCredential,
    [Parameter(Mandatory = $false)]
    [Alias('Workspace')]
    [string] $VcsWorkspace = '',
    [Parameter(Mandatory = $false)]
    [Alias('Branch', 'VcsStream', 'Stream')]
    [string] $VcsBranch = '',
    [Parameter(Mandatory = $false)]
    [ValidatePattern('^(?:[0-9]+|Default|New)$')]
    [Alias('Changelist', 'VcsChange', 'Change')]
    [string] $VcsChangelist = 'New',

    [Parameter(Mandatory = $false)]
    [switch] $PassThru,
    [Parameter(Mandatory = $false)]
    [switch] $Force,
    [Parameter(Mandatory = $false)]
    [Alias('Quiet')]
    [switch] $Silent
);
begin
{
    # Position of default root source path in relation to script directory.
    $__commons__defaultSrcPathRelPosition = '..\..\..';


    # Check for minimum version of PowerShell.
    if ($Host.Version.Major -lt 3)
    {
        throw ('Script requires at least PowerShell version 3. Current version ({0}) is too low.' -f $Host.Version);
    }


    ### --- UTILS ---

    # done
    function _get-ScriptPath
    {
        <#
            .SYNOPSIS
                Returns absolute path to invoked script file.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param();

        return $PSCmdlet.MyInvocation.ScriptName;
    }

    # done
    function _get-ScriptDir
    {
        <#
            .SYNOPSIS
                Returns absolute path to directory of invoked script file.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param();

        return $PSCmdlet.MyInvocation.PSScriptRoot;
    }

    # done
    function _get-ScriptConfigPath
    {
        <#
            .SYNOPSIS
                Returns absolute path to configuration file for invoked script.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param();

        return Join-Path (_get-ScriptDir) ([System.IO.Path]::ChangeExtension((Split-Path -Leaf (_get-ScriptPath)), '.config'));
    }

    # done
    function _get-DefaultSrcPath
    {
        <#
            .SYNOPSIS
                Returns absolute path to default location of repository with project code.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param();

        return (Resolve-Path -LiteralPath (Join-Path (_get-ScriptDir) $__commons__defaultSrcPathRelPosition)).Path;
    }

    ### --- CONFIGURATION ---

    # done
    function _load-ScriptConfig
    {
        <#
            .SYNOPSIS
                Loads configuration from configuration file.

            .DESCRIPTION
                Loads JSON-like confuration file and deserializes configuration into object(s).

            .PARAMETER CfgPath
                Path to configuration file.

            .OUTPUTS
                [PSObject] One or more "sync" objects deserialized as PSObject.

                If error occured, none is returned.
        #>

        [CmdletBinding()]
        [OutputType([PSObject])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [Alias('Path')]
            [string] $CfgPath
        );
        begin {}
        process
        {
            # Check for configuration file.
            $CfgPath = _get-ScriptConfigPath;
            if (!(Test-Path -LiteralPath $CfgPath -PathType Leaf))
            {
                Write-Error ('Configuration file for script cannot be found: "{0}"' -f $CfgPath);
                return;
            }

            # Parse JSON in configuration file.
            try
            {
                $_cfgContent = ConvertFrom-Json ('{' + (Get-Content -LiteralPath $CfgPath -Encoding UTF8 -Raw -Force -ErrorAction Stop) + '}') -ErrorAction Stop;
            }
            catch
            {
                Write-Error -ErrorRecord $_;
                Write-Error ('Content of configuration file is invalid (parsing failed): "{0}"' -f $CfgPath);
                return;
            }
        
            # Select primary configuration member.
            $_cfgSync = @($_cfgContent.sync | ? { $_ -ne $null });
            if ($_cfgSync.Count -le 0)
            {
                Write-Error ('Content of configuration file is invalid (no "sync" member): "{0}"' -f $CfgPath);
                return;
            }

            $_cfgSync | Write-Output;
        }
        end {}
    }

    ### --- PATH UTILS ---

    # done
    # Standard directory separators:
    $__commons__dirSeps       = @([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar, '\', '/');
    # Standard path patterns:
    $__commons__dirSepsString = @($__commons__dirSeps | Sort-Object -Unique | % { [regex]::Escape($_) }) -join '';
    $__commons__dirSepPattern = '[{0}]' -f $__commons__dirSepsString;
    # Two star exception handler (replace patterns). Allows '/**/' patterns to match single '/' or '/**/**/' to match '/sample/' (collapsing of single '/').
    $_convert_AntLikeWildcard_starStarPattern = '.*';
    $_convert_AntLikeWildcard_twoStarExceptPattern = New-Object regex ('{0}{1}{0}((?:{1}{0})*)' -f [regex]::Escape($__commons__dirSepPattern), [regex]::Escape($_convert_AntLikeWildcard_starStarPattern)), 'IgnoreCase, Compiled';
    $_convert_AntLikeWildcard_twoStarExceptReplace = '{0}(?:{1}{0})?$1' -f $__commons__dirSepPattern, $_convert_AntLikeWildcard_starStarPattern;
    function _convert-AntLikeWildcard
    {
        <#
            .SYNOPSIS
                Converts Ant-like wildcards to regular expression patterns.

            .DESCRIPTION
                Converts Ant-like wildcards to regular expression patterns that can be matched by normal regex engine.

            .PARAMETER Wildcard
                Ant-like wildcard to convert.

            .OUTPUTS
                [string] - Regex pattern that matches the same paths as input wildcard. This pattern should only be
                used with simplified/normalized relative paths (.\-prefixed).
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyString()]
            [string] $Wildcard
        );
        begin
        {
            # Standard patterns:
            $_dirSepsString   = $__commons__dirSepsString;
            $_dirSepPattern   = $__commons__dirSepPattern;
            $_questionPattern = '[^{0}]' -f $_dirSepsString;
            $_starPattern     = '{0}*' -f $_questionPattern;
            $_starStarPattern = $_convert_AntLikeWildcard_starStarPattern;

            # Exception pattern/replace (post-convert correction):
            $_twoStarExceptPattern = $_convert_AntLikeWildcard_twoStarExceptPattern;
            $_twoStarExceptReplace = $_convert_AntLikeWildcard_twoStarExceptReplace;
        }
        process
        {
            # Remove redundant elements from wildcard:
            $_wildcard = $Wildcard -replace '[\\/]+', '\';
            $_wildcard = $_wildcard -replace '(?:\\\.)+(?=\\)', '';
            $_wildcard = $_wildcard -replace '^\.(?:\\|$)', '';
            $_wildcard = $_wildcard -replace '(?:\*\*){2,}', '**';


            # Very simple token parsing and subsitution (4 state).
            [bool] $_inStar  = $false;
            [bool] $_inGroup = $false;
            $_wildcardPattern = (@($_wildcard.GetEnumerator() | % {
                switch -Exact ($_)
                {
                    '\' { if($_inGroup) { return $_dirSepsString; }        else { if($_inStar) { Write-Output $_starPattern; $_inStar = $false; } return $_dirSepPattern; } }
                    '?' { if($_inGroup) { return [regex]::Escape($_); }    else { if($_inStar) { Write-Output $_starPattern; $_inStar = $false; } return $_questionPattern; } }
                    '*' { if($_inGroup) { return [regex]::Escape($_); }    else { if($_inStar) { $_inStar = $false; return $_starStarPattern; }   $_inStar = $true; } }
                    '[' { if($_inGroup) { return [regex]::Escape($_); }    else { if($_inStar) { Write-Output $_starPattern; $_inStar = $false; } $_inGroup = $true; return $_; } }
                    ']' { if($_inGroup) { $_inGroup = $false; return $_; } else { if($_inStar) { Write-Output $_starPattern; $_inStar = $false; } return [regex]::Escape($_); } }
                    default { if($_inStar) { Write-Output $_starPattern; $_inStar = $false; } return [regex]::Escape($_); }
                }
            }) -join '');

            if ($_inGroup)
            {
                $_wildcardPattern += ']';
            }
            if ($_inStar)
            {
                $_wildcardPattern += $_starPattern;
            }
            if ([string]::IsNullOrEmpty($_wildcardPattern))
            {
                $_wildcardPattern = $_starStarPattern;
            }

            $_wildcardPattern = '^\.' + $_dirSepPattern + $_wildcardPattern + '$'; # add anchors and common prefix.
            return $_twoStarExceptPattern.Replace($_wildcardPattern, $_twoStarExceptReplace);
        }
        end {}
    }

    # done
    # Two dot (..) simplifying pattern for relative paths:
    [regex] $_simplify_RelPath_twoDotPattern = New-Object regex '(?:\\|^)(?:[^\.]|\.[^\.]|\.\.[^\\])[^\\]*\\\.\.(?=\\|$)', 'IgnoreCase, Compiled';
    function _simplify-RelPath
    {
        <#
            .SYNOPSIS
                Simplify relative path.

            .DESCRIPTION
                Simplify relative path by removing redundant directory separators, single and two dot patterns.

                Simplified path does not need to point to existing item.

            .PARAMETER Path
                Path to simplify.

            .OUTPUTS
                [string] - Simplified relative path.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [string] $Path
        );
        begin
        {
            # Standard patterns:
            $_dirSepPattern = $__commons__dirSepPattern;

            $_twoDotPattern = $_simplify_RelPath_twoDotPattern;
        }
        process
        {
            $_path = $Path -replace "${_dirSepPattern}+", '\'; # Remove redundant separators.
            $_path = $_path -replace '(?:\\\.)+(?=\\)', '';    # Remove unnecessary dots.
            $_path = $_path -replace '^\.(?:\\|$)', '';
            while ($true)
            {
                $_reducedPath = $_twoDotPattern.Replace($_path, '');
                if ($_reducedPath.Length -eq $_path.Length)
                {
                    break;
                }
                $_path = $_reducedPath;
            }

            if ([string]::IsNullOrEmpty($_path))
            {
                return '.';
            }
            if (Split-Path $_path -IsAbsolute) { return $_path; } # Fall-back for absolute paths.
            return Join-Path '.' $_path; # Normalize.
        }
        end {}
    }

    ### --- LIST ITEMS BY CONFIG ---

    # done
    # The invalid relative path pattern (contains two dot (..) special directories).
    $_list_AllCopyItems_invalidPathPattern = New-Object regex ('(?:^|{0})\.\.(?:{0}|$)' -f $__commons__dirSepPattern), 'IgnoreCase, Compiled';
    function _list-AllCopyItems
    {
        <#
            .SYNOPSIS
                List all items that need to be synchronized with information how the item needs to be synchronized.

            .DESCRIPTION
                Based on source path, destination path, optional prefixes and filters, the function returns list
                of item (file/directory) information that meets createria.
                Each item returned is extended with following properties:
                 -- [string] 'SyncWosMode'           - Mode of synchronization that should be applied to the item.
                                                       Currently it is returning following modes:
                                                        -- 'Copy'      - The item should be copied from source to destination.
                                                        -- 'None'      - The item does not need to be copied
                                                                         (source is the same as destination).
                                                        -- 'Remove'    - The item should be removed from destination
                                                                         (possibly from source, if source is the same
                                                                         as destination).
                                                        -- 'RemoveDst' - The item should be removed from destination
                                                                         (will not affect source).
                 -- [string] 'SyncWosRootRelPath'    - Relative path to selected source path (source-rooted). 
                 -- [string] 'SyncWosDstRootRelPath' - Relative path to selected destination path (destination-rooted).
                Rest of item information is taken from source except for 'RemoveDst' items which are taken from destination.

                The filter can define which items to include, which to exclude (will be marked to remove if detected) and which
                to ignore (remove completely from list of items to analyze).
                Filters are applied in following order for items from source:
                 1. Ignore filters remove all items that should be ignored (if container is ignored all child items are
                    ignored as well.
                 2. Include filters are applied for container-like items from point #1 - items are marked with 'Copy' or 'None'.
                 3. Exclude filters are applied for container-like items from point #1 - items are marked with 'Remove'.
                 4. Include filters are applied for all items from point #1 - items are marked with 'Copy' or 'None'.
                    Also all child items of containers marked 'Copy'/'None' in point #2 are marked with 'Copy'/'None'.
                 5. Exclude filters are applied for all items from point #1 - items are marked with 'Remove'.
                    Also all child items of containers marked 'Remove' in point #3 are marked with 'Remove'.
                Items from destination are filtered only by ignore filters. Remaining items are marked by 'RemoveDst'.

                Items from pipelined runs are merged (duplicates with the same relative source and destination paths removed)
                and mode corrected if necessary (the mode is ugraded from 'RemoveDst' -> 'Remove' -> 'Copy'/'None'). 

            .PARAMETER LiteralPath
                Path to source root path.

                Wildcards are NOT allowed. Path must exist and point to container (directory).

            .PARAMETER Qualifier
                Qualifier which is applied to source root. It allows to preselect files to copy.

                Files not selected are marked to remove.

                Wildcards are allowed. Qualifier cannot be rooted (absolute).

            .PARAMETER DstLiteralPath
                Path to destination root path.

                Wildcards are NOT allowed. Path can point to non-existent location.

            .PARAMETER SrcPrefix
                Prefix path for source items (suffix for source root path) that allows to rebase files
                during copy/removal. When mapping files from source to destination the SrcPrefix is switched
                to DstPrefix.
                
            .PARAMETER DstPrefix
                Prefix path for destination items (suffix for destination root path) that allows to rebase files
                during copy/removal. When mapping files from source to destination the SrcPrefix is switched
                to DstPrefix.

            .PARAMETER IncludeFilter
                Ant-like list of include filters (Ant-like wildcards). If none or empty collection is specified in both
                IncludeFilter and ReIncludeFilter, all non-ignored files from source are included.
                
            .PARAMETER ReIncludeFilter
                Regular expression list of include filters. If none or empty collection is specified in both
                IncludeFilter and ReIncludeFilter, all non-ignored files from source are included.

            .PARAMETER ExcludeFilter
                Ant-like list of exclude filters (Ant-like wildcards).
                
            .PARAMETER ReExcludeFilter
                Regular expression list of exclude filters.

            .PARAMETER IgnoreFilter
                Ant-like list of ignore filters (Ant-like wildcards).
                
            .PARAMETER ReIgnoreFilter
                Regular expression list of ignore filters.

            .OUTPUTS
                [System.Io.DirectoryInfo], [System.Io.FileInfo] or other item information extended with:
                 -- [string] 'SyncWosMode'           - Mode of synchronization that should be applied to the item.
                                                       Currently it is returning following modes:
                                                        -- 'Copy'      - The item should be copied from source to destination.
                                                        -- 'None'      - The item does not need to be copied
                                                                         (source is the same as destination).
                                                        -- 'Remove'    - The item should be removed from destination
                                                                         (possibly from source, if source is the same
                                                                         as destination).
                                                        -- 'RemoveDst' - The item should be removed from destination
                                                                         (will not affect source).
                 -- [string] 'SyncWosRootRelPath'    - Relative path to selected source path (source-rooted). 
                 -- [string] 'SyncWosDstRootRelPath' - Relative path to selected destination path (destination-rooted).
                Rest of item information is taken from source except for 'RemoveDst' items which are taken from destination.

                Returned items are returned in sorted by item type (first directories/containers, next files/leafs),
                SyncWosRootRelPath (ascending) and SyncWosMode (ascending).
        #>

        [CmdletBinding()]
        [OutputType([System.Io.DirectoryInfo], [System.Io.FileInfo])]
        param(
            [Parameter(Position = 0, Mandatory = $true)]
            [Alias('SrcPath')]
            [string] $LiteralPath,
            [Parameter(Position = 1, Mandatory = $false,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [string] $Qualifier = '.',
            [Parameter(Position = 2, Mandatory = $false)]
            [Alias('DstPath')]
            [string] $DstLiteralPath = '',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateNotNullOrEmpty()]
            [string] $SrcPrefix = '.',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateNotNullOrEmpty()]
            [string] $DstPrefix = '.',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $IncludeFilter = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $ReIncludeFilter = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $ExcludeFilter = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $ReExcludeFilter = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $IgnoreFilter = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $ReIgnoreFilter = @()
        );
        begin
        {
            $_pipeMergedContainerItems = @{};
            $_pipeMergedLeafItems      = @{};
        }
        process
        {
            # Initial checks.
            if (!(Test-Path -LiteralPath $LiteralPath -PathType Container))
            {
                Write-Warning ('LiteralPath: Path to source is invalid, is not container (directory) or cannot be found: "{0}"' -f $LiteralPath);
                return;
            }
            $_srcQualPath = Join-Path $LiteralPath $Qualifier;
            if (!(Test-Path -Path $_srcQualPath))
            {
                Write-Warning ('Qualifier: Source path qualifier seems to be invalid or point to place with no items: "{0}"' -f $Qualifier);
                return;
            }

            $_srcPrefixPath = Join-Path $LiteralPath $SrcPrefix;
            if (!(Test-Path -LiteralPath $_srcPrefixPath -PathType Container))
            {
                Write-Warning ('SrcPrefix: Path to selected source prefix is invalid, is not container (directory) or cannot be found: "{0}"' -f $SrcPrefix);
                return;
            }
            if (!(Test-Path -LiteralPath $DstPrefix -PathType Container -IsValid) -or (Split-Path $DstPrefix -IsAbsolute))
            {
                Write-Warning ('DstPrefix: Destination path prefix is invalid or is not relative: "{0}"' -f $DstPrefix);
                return;
            }

            if ([string]::IsNullOrEmpty($DstLiteralPath)) { $_dstPath = $LiteralPath; } else { $_dstPath = $DstLiteralPath; }
            if (!(Test-Path -LiteralPath $_dstPath -PathType Container -IsValid))
            {
                Write-Warning ('DstLiteralPath: Path to destination is invalid: "{0}"' -f $_dstPath);
                return;
            }
            $_dstPrefixPath = Join-Path $_dstPath $DstPrefix;
            [bool] $_dstNeedsCleanup = $false;
            [bool] $_srcDstSame      = $false;
            if (Test-Path -LiteralPath $_dstPrefixPath)
            {
                if (Test-Path -LiteralPath $_dstPrefixPath -PathType Container)
                {
                    $_srcAbsPath = (Resolve-Path -LiteralPath $_srcPrefixPath).Path;
                    $_dstAbsPath = (Resolve-Path -LiteralPath $_dstPrefixPath).Path;

                    $_srcRootAbsPath = (Resolve-Path -LiteralPath $LiteralPath).Path;
                    $_dstRootAbsPath = (Resolve-Path -LiteralPath $DstLiteralPath).Path;

                    if ($_srcAbsPath -ne $_dstAbsPath) { $_dstNeedsCleanup = $true; }
                    if (($_srcAbsPath -eq $_dstAbsPath) -and ($_srcRootAbsPath -eq $_dstRootAbsPath)) { $_srcDstSame = $true; }
                }
                else { Write-Warning ('DstLiteralPath, DstPrefix: Calculated path to destination is invalid or is not container (directory): "{0}"' -f $_dstPrefixPath); }
            }


            # List of qualified items to copy/remove (from source).
            Write-Verbose ('Gathering all candidate items from source qualified path for copy/remove: "{0}"' -f $_srcQualPath);
            [string] $_progActivity = 'Preparing list of items to copy/remove';
            [string] $_progStatus   = 'Gathering all candidates for copy/remove';
            [int]    $_progPercent  = 0;
            [int]    $_progDelta    = 5;
            [int]    $_progItems    = 0;
            [int]    $_progItemsPP  = 1;
            [int]    $_progItemsCP  = $_progPercent;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            $_allItems    = @(Get-ChildItem -Path $_srcQualPath -Recurse -Force | Add-Member 'SyncWosMode' 'Remove' -PassThru | Add-Member 'SyncWosTasks' ([System.Collections.Generic.HashSet`1[[string]]] @('Copy')) -PassThru);
            Write-Verbose ('Gathered all source candidate items ({1} items): "{0}"' -f $_srcQualPath, $_allItems.Count);
            if ($_dstNeedsCleanup)
            {
                Write-Verbose ('Gathering all candidate items from effective destination path for remove: "{0}"' -f $_dstPrefixPath);
                $_allDstItems = @(Get-ChildItem -LiteralPath $_dstPrefixPath -Recurse -Force | Add-Member 'SyncWosMode' 'RemoveDst' -PassThru | Add-Member 'SyncWosTasks' ([System.Collections.Generic.HashSet`1[[string]]] @('Copy')) -PassThru);
                Write-Verbose ('Gathered all destination candidate items ({1} items): "{0}"' -f $_dstPrefixPath, $_allDstItems.Count);
            }
            else { $_allDstItems = @(); }


            Write-Verbose ("Gathering information about items (source-path-rooted relative paths): `"{0}`"`n    Source path: `"{1}`"." -f $_srcQualPath, $LiteralPath);
            $_progStatus  = 'Gathering information (source-path-rooted relative paths)';
            $_progPercent += $_progDelta;
            $_progDelta   = 25;
            $_progItems   = 0;
            $_progItemsPP = [Math]::Ceiling(($_allItems.Count + $_allDstItems.Count) / $_progDelta);
            $_progItemsCP = $_progPercent;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            [bool] $_locationPushed = $false;
            try
            {
                Push-Location -LiteralPath $LiteralPath;
                $_locationPushed = $?;

                # Get rooted path information.
                $_allItems = @($_allItems | % {
                        Add-Member 'SyncWosRootRelPath' (_simplify-RelPath (Resolve-Path -Relative -LiteralPath $_.PSPath)) -InputObject $_ -PassThru;
                        ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; }
                    });
            }
            finally
            {
                if ($_locationPushed)
                {
                    Pop-Location;
                    $_locationPushed = $false;
                }
            }
            Write-Verbose ('Gathered information about items (source-path-rooted relative paths; {1} items): "{0}"' -f $_srcQualPath, $_allItems.Count);

            if ($_dstNeedsCleanup)
            {
                Write-Verbose ("Gathering information about destination items (effective source-/destination-path-rooted relative paths): `"{0}`"`n    Effective destination path: `"{1}`"." -f $_dstPath, $_dstPrefixPath);
                $_progStatus  = 'Gathering information (effective source-/destination-path-rooted relative paths)';
                $_locationPushed = $false;
                try
                {
                    Push-Location -LiteralPath $_dstPrefixPath;
                    $_locationPushed = $?;

                    # Get rooted path information.
                    $_allDstItems = @($_allDstItems | % {
                            $_relPath = _simplify-RelPath (Resolve-Path -Relative -LiteralPath $_.PSPath);
                            Add-Member 'SyncWosRootRelPath' (_simplify-RelPath (Join-Path $SrcPrefix $_relPath)) -InputObject $_ -PassThru |
                            Add-Member 'SyncWosDstRootRelPath' (_simplify-RelPath (Join-Path $DstPrefix $_relPath)) -PassThru;
                            ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; }
                        });
                }
                finally
                {
                    if ($_locationPushed)
                    {
                        Pop-Location;
                        $_locationPushed = $false;
                    }
                }
                Write-Verbose ('Gathered information about destination items (effective source-/destination-path-rooted relative paths; {1} items): "{0}"' -f $_dstPath, $_allDstItems.Count);
            }


            Write-Verbose ("Gathering information about items (calculated destination relative paths): `"{0}`"`n    Source path:        `"{1}`"`n    Source prefix:      `"{2}`"`n    Destination prefix: `"{3}`"." -f $_srcQualPath, $LiteralPath, $SrcPrefix, $DstPrefix);
            $_progStatus  = 'Gathering information (calculated destination relative paths)';
            $_progPercent += $_progDelta;
            $_progDelta   = 25;
            $_progItems   = 0;
            $_progItemsPP = [Math]::Ceiling($_allItems.Count / $_progDelta);
            $_progItemsCP = $_progPercent;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            $_locationPushed = $false;
            try
            {
                Push-Location -LiteralPath $_srcPrefixPath;
                $_locationPushed = $?;

                # Get relative destination path information (calculated).
                $_allItems = @($_allItems | % {
                        Add-Member 'SyncWosDstRootRelPath' (_simplify-RelPath (Join-Path $DstPrefix (Resolve-Path -Relative -LiteralPath $_.PSPath))) -InputObject $_ -PassThru;
                        ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; }
                    });
            }
            finally
            {
                if ($_locationPushed)
                {
                    Pop-Location;
                    $_locationPushed = $false;
                }
            }
            Write-Verbose ('Gathered information about items (calculated destination relative paths; {1} items): "{0}"' -f $_srcQualPath, $_allItems.Count);


            # Gather all paterns for include/exclude/ignore items.
            $_invalidPathPattern = $_list_AllCopyItems_invalidPathPattern;
            $_includePatterns = @(@($IncludeFilter | _convert-AntLikeWildcard; $ReIncludeFilter | ? { ![string]::IsNullOrEmpty($_) }) | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_excludePatterns = @(@($ExcludeFilter | _convert-AntLikeWildcard; $ReExcludeFilter | ? { ![string]::IsNullOrEmpty($_) }) | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_ignorePatterns  = @(@($IgnoreFilter  | _convert-AntLikeWildcard; $ReIgnoreFilter  | ? { ![string]::IsNullOrEmpty($_) }) | % { New-Object regex $_, 'IgnoreCase, Compiled' });

            # Remove ignored items.
            Write-Verbose ('Filtering out ignored items ({1} items): "{0}"' -f $_srcQualPath, ($_allItems.Count + $_allDstItems.Count));
            $_progStatus  = 'Filtering out ignored items';
            $_progPercent += $_progDelta;
            $_progDelta   = 10;
            $_progItems   = 0;
            $_progItemsPP = [Math]::Ceiling(($_allItems.Count + $_allDstItems.Count) / $_progDelta);
            $_progItemsCP = $_progPercent;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            $_ignContainerPaths    = @($_allItems | ? { $_.PSIsContainer } |
                ? { $_item = $_; ($_ignorePatterns.Count -gt 0) -and (@($_ignorePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                % { Join-Path $_.SyncWosRootRelPath '*' });
            $_ignDstContainerPaths = @($_allDstItems | ? { $_.PSIsContainer } |
                ? { $_item = $_; ($_ignorePatterns.Count -gt 0) -and (@($_ignorePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                % { Join-Path $_.SyncWosRootRelPath '*' });

            $_allItems    = @($_allItems |
                    ? {
                        $_srcValid = !$_invalidPathPattern.IsMatch($_.SyncWosRootRelPath);
                        $_dstValid = !$_invalidPathPattern.IsMatch($_.SyncWosDstRootRelPath);

                        if (!$_srcValid) { Write-Warning ("Calculated source path for item is probably outside source directory. It will be ignored: `"{0}`"`n    -- Invalid path: `"{1}`"."      -f $_.PSPath, $_.SyncWosRootRelPath); }
                        if (!$_dstValid) { Write-Warning ("Calculated destination path for item is probably outside source directory. It will be ignored: `"{0}`"`n    -- Invalid path: `"{1}`"." -f $_.PSPath, $_.SyncWosDstRootRelPath); }

                        $_srcValid -and $_dstValid

                        ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; }
                    } |
                    ? { $_item = $_; (($_ignorePatterns.Count -eq 0) -or (@($_ignorePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -le 0)) -and `
                                     (($_ignContainerPaths.Count -eq 0) -or (@($_ignContainerPaths | ? { $_item.SyncWosRootRelPath -like $_ }).Count -le 0)) }
                );
            $_allDstItems = @($_allDstItems |
                    ? {
                        $_srcValid = !$_invalidPathPattern.IsMatch($_.SyncWosRootRelPath);
                        $_dstValid = !$_invalidPathPattern.IsMatch($_.SyncWosDstRootRelPath);

                        if (!$_srcValid) { Write-Warning ("Calculated source path for item is probably outside source directory. It will be ignored: `"{0}`"`n    -- Invalid path: `"{1}`"."      -f $_.PSPath, $_.SyncWosRootRelPath); }
                        if (!$_dstValid) { Write-Warning ("Calculated destination path for item is probably outside source directory. It will be ignored: `"{0}`"`n    -- Invalid path: `"{1}`"." -f $_.PSPath, $_.SyncWosDstRootRelPath); }

                        $_srcValid -and $_dstValid

                        ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; }
                    } |
                    ? { $_item = $_; (($_ignorePatterns.Count -eq 0) -or (@($_ignorePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -le 0)) -and `
                                     (($_ignDstContainerPaths.Count -eq 0) -or (@($_ignDstContainerPaths | ? { $_item.SyncWosRootRelPath -like $_ }).Count -le 0)) }
                );
            Write-Verbose ('Filtered out ignored items ({1} items remained): "{0}"' -f $_srcQualPath, ($_allItems.Count + $_allDstItems.Count));

            # Split items on containers (directories, etc.) and leafs (files, etc.).
            $_allContainerItems    = @($_allItems    | ? { $_.PSIsContainer });
            $_allLeafItems         = @($_allItems    | ? { !$_.PSIsContainer });
            $_allDstContainerItems = @($_allDstItems | ? { $_.PSIsContainer });
            $_allDstLeafItems      = @($_allDstItems | ? { !$_.PSIsContainer });

            # Provide container filtering (if container is included, all items from this container are included; if container is excluded, all items from this container are excluded;
            # include filter is applied first (after ignore filters filtered out items which should not be taken into consideration), next exclude filter).
            Write-Verbose ('Filtering containers (directories) by includes/excludes ({1} items): "{0}"' -f $_srcQualPath, $_allContainerItems.Count);
            $_progStatus  = 'Filtering containers (directories) by includes/excludes';
            $_progPercent += $_progDelta;
            $_progDelta   = 10;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            $_incContainerPaths = @($_allContainerItems |
                ? { $_item = $_; ($_includePatterns.Count -eq 0) -or (@($_includePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                % { $_.SyncWosMode = 'Copy'; Join-Path $_.SyncWosRootRelPath '*' });
            $_excContainerPaths = @($_allContainerItems |
                ? { $_item = $_; ($_excludePatterns.Count -gt 0) -and (@($_excludePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                % { $_.SyncWosMode = 'Remove'; Join-Path $_.SyncWosRootRelPath '*' });
            Write-Verbose ('Filtered containers (directories) by includes/excludes ({1} items): "{0}"' -f $_srcQualPath, $_allContainerItems.Count);

            # Filter leaf/containers items (by filters and container filtering).
            Write-Verbose ('Filtering items (directories and files) by includes/excludes and calculated container filters ({1} items): "{0}"' -f $_srcQualPath, ($_allContainerItems.Count + $_allLeafItems.Count));
            $_progStatus  = 'Filtering items (directories and files) by includes/excludes and calculated container filters';
            $_progPercent += $_progDelta;
            $_progDelta   = 20;
            $_progItems   = 0;
            $_progItemsPP = [Math]::Ceiling(2 * ($_allContainerItems.Count + $_allLeafItems.Count) / $_progDelta);
            $_progItemsCP = $_progPercent;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            @($_allContainerItems; $_allLeafItems) |
                ? { $_item = $_; ($_includePatterns.Count -eq 0) -or (@($_includePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) -or `
                                 (@($_incContainerPaths | ? { $_item.SyncWosRootRelPath -like $_ }).Count -gt 0)
                    ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; } } |
                % { $_.SyncWosMode = 'Copy' };
            @($_allContainerItems; $_allLeafItems) |
                ? { $_item = $_; (($_excludePatterns.Count -gt 0) -and (@($_excludePatterns | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0)) -or `
                                 (($_excContainerPaths.Count -gt 0) -and (@($_excContainerPaths | ? { $_item.SyncWosRootRelPath -like $_ }).Count -gt 0))
                    ++$_progItems; if ($_progItems % $_progItemsPP -eq 0) { Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP; ++$_progItemsCP; } } |
                % { $_.SyncWosMode = 'Remove' };
            Write-Verbose ('Filtered items (directories and files) by includes/excludes and calculated container filters ({1} items): "{0}"' -f $_srcQualPath, ($_allContainerItems.Count + $_allLeafItems.Count));


            # Pipeline merging of item collections.
            Write-Verbose ('Merging items into list ({1} items): "{0}"' -f $_srcQualPath, ($_allContainerItems.Count + $_allDstContainerItems.Count + $_allLeafItems.Count + $_allDstLeafItems.Count));
            $_progStatus  = 'Merging items into list';
            $_progPercent += $_progDelta;
            $_progDelta   = 5;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
            @($_allDstContainerItems; $_allContainerItems) | % {
                    $_key = '{0}|{1}' -f $_.SyncWosRootRelPath, $_.SyncWosDstRootRelPath;
                    if($_pipeMergedContainerItems.ContainsKey($_key))
                    {
                        $_pipeMergedItem = $_pipeMergedContainerItems[$_key];
                        if (($_.SyncWosMode -eq 'Copy') -and (($_pipeMergedItem.SyncWosMode -ne 'None') -or ($_pipeMergedItem.SyncWosMode -ne 'Copy')))
                        {
                            if ($_srcDstSame) { $_.SyncWosMode = 'None'; }
                            $_pipeMergedContainerItems[$_key] = $_;
                        }
                        elseif (($_.SyncWosMode -eq 'Remove') -and ($_pipeMergedItem.SyncWosMode -eq 'RemoveDst'))
                        {
                            $_pipeMergedContainerItems[$_key] = $_;
                        }
                    }
                    else
                    {
                        if ($_srcDstSame -and ($_.SyncWosMode -eq 'Copy')) { $_.SyncWosMode = 'None'; }
                        $_pipeMergedContainerItems[$_key] = $_;
                    }
                };
            @($_allDstLeafItems; $_allLeafItems) | % {
                    $_key = '{0}|{1}' -f $_.SyncWosRootRelPath, $_.SyncWosDstRootRelPath;
                    if($_pipeMergedLeafItems.ContainsKey($_key))
                    {
                        $_pipeMergedItem = $_pipeMergedLeafItems[$_key];
                        if (($_.SyncWosMode -eq 'Copy') -and (($_pipeMergedItem.SyncWosMode -ne 'None') -or ($_pipeMergedItem.SyncWosMode -ne 'Copy')))
                        {
                            if ($_srcDstSame) { $_.SyncWosMode = 'None'; }
                            $_pipeMergedLeafItems[$_key] = $_;
                        }
                        elseif (($_.SyncWosMode -eq 'Remove') -and ($_pipeMergedItem.SyncWosMode -eq 'RemoveDst'))
                        {
                            $_pipeMergedLeafItems[$_key] = $_;
                        }
                    }
                    else
                    {
                        if ($_srcDstSame -and ($_.SyncWosMode -eq 'Copy')) { $_.SyncWosMode = 'None'; }
                        $_pipeMergedLeafItems[$_key] = $_;
                    }
                };
            Write-Verbose ('Merged items into list ({1} items): "{0}"' -f $_srcQualPath, ($_allContainerItems.Count + $_allDstContainerItems.Count + $_allLeafItems.Count + $_allDstLeafItems.Count));
            $_progStatus  = 'Completed';
            $_progPercent += $_progDelta;
            Write-Progress -Id 2 -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent -Completed;
        }
        end
        {
            $_pipeMergedContainerItems.Values | sort SyncWosRootRelPath, SyncWosMode;
            $_pipeMergedLeafItems.Values      | sort SyncWosRootRelPath, SyncWosMode;
        }
    }

    ### --- PROCESS ---

    [bool] $_owDstYesToAll = $false;
    [bool] $_owDstNoToAll  = $false;
    [bool] $_rmDstYesToAll = $false;
    [bool] $_rmDstNoToAll  = $false;

    # done
    $_process_SectionsItem_anyNewLine = New-Object regex '\r\n|[\v\r\f\n\x85\p{Zl}\p{Zp}]', 'Compiled';
    function _process-SectionsItem
    {
        <#
            .SYNOPSIS
                Process sections-aware items by filtering out selected sections (by excluding specified sections).

            .DESCRIPTION
                Process selected source item (sections-aware file) and removes/uncomments/indents/back-indents
                sections found inside. Result is written to selected destination items.
                
                Source item can be at the same time destination item (updating file in place is allowed).

                Processing sections are placed in single line, BEGIN in the form (case-senstive):
                    '## PROCESS <process_key> BEGIN (<section_name> [, <section_name> [, ...]])
                    e.g.   '## PROCESS PPC BEGIN (Sec1, Sec2, Sec3)'
                and END in the form (case-sensitive):
                    '## PROCESS <process_key> END'
                    e.g.   '## PROCESS PPC END'
                where:
                    <process_key>  - token to identify group of sections to process (see ProcessKey parameter).
                    <section_name> - name(s) of section.

                The section can be excluded by specifying ExcludeSections with all names specified in section
                you want to exclude.

                There is also sepecial ELSE part of section which can be placed inside section which will
                indent/indent back or uncomment code:
                    '## PROCESS <process_key> ELSE INDENT <levels>'
                    '## PROCESS <process_key> ELSE INDENT BACK <levels>'
                    '## PROCESS <process_key> ELSE UNCOMMENT'
                where:
                    <process_key> - token to identify group of sections to process (see ProcessKey parameter).
                    <levels>      - positive integral number with number of levels to indent or indent back.
                If section is excluded and contains ELSE, the lines between BEGIN and ELSE are excluded,
                and lines between ELSE and END are either indented, intended back or special comments are
                uncommented.

                UTF8 content is assumed. Written items are written using UTF8 encoding (BOM is not written).

            .PARAMETER SrcPath
                Path to source item. Path must be valid and must point to (non-container) item which exists.

                Wildcards are NOT allowed.

            .PARAMETER DstPath
                Paths to destination items (non-container). If the item exists, it will be overwritten with updated content
                from source item. If it does not exists, it will be created.

                Written items are written using UTF8 encoding (BOM is not written).

                Wildcards are NOT allowed. Please use -Force switch, if you want to overwrite read-only files.

            .PARAMETER ProcessKey
                Identifier used to identify processing sections processed by this cmd-let.

                Please note that this cmd-let will only process sections/conditional code comments marked
                with selected key (other will be treated as normal comments).

            .PARAMETER ExcludeSections
                Names of sections to exclude. Section is excluded if all names for this section
                are excluded.

                For example, if ExcludeSections = @(A, B) the sections started with:
                 -- '## PROCESS PPC BEGIN (A)'
                 -- '## PROCESS PPC BEGIN (B)'
                 -- '## PROCESS PPC BEGIN (A,B)'
                will be excluded, but e.g.:
                 -- '## PROCESS PPC BEGIN (C)'
                 -- '## PROCESS PPC BEGIN (A,C)'
                will be not.

            .PARAMETER CommentType
                Type of processing comments (type of file in which special comments reside).

                All special comments (defining conditional code comments or sections) should start at new line
                (indent is allowed) be prefixed with:
                  -- '##'    for 'CMake'
                  -- '// ##' for 'Cxx' (spacing between '//' and '##' is optional)

                Allowed values: 'CMake', 'Cxx'.

            .PARAMETER TabSize
                Number of spaces used as one tab (for indenting/back-indenting).

                Allowed range: 1 - 1024.

            .PARAMETER EolConvention
                End-of-line convention used to write/overwrite destination items (processed content).

                New lines will be written as:
                 -- <LF>     for 'Unix'
                 -- <CR><LF> for 'Windows'
                 -- <CR>     for 'Mac'

                Allowed values: 'Unix', 'Windows', 'Mac'.

            .PARAMETER UseTab
                Use tabs for indenting.

            .PARAMETER Force
                Force overwrite destination items (even if they are read-only).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([void])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [string] $SrcPath,
            [Parameter(Position = 1, Mandatory = $true,
                       ValueFromPipelineByPropertyName = $true)]
            [string[]] $DstPath,
            [Parameter(Position = 2, Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateNotNullOrEmpty()]
            [Alias('Key')]
            [string] $ProcessKey = 'PPC',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [AllowEmptyCollection()]
            [string[]] $ExcludeSections = @(),
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('CMake', 'Cxx')]
            [string] $CommentType = 'CMake',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateRange(1, 1024)]
            [int] $TabSize = 2,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('Unix', 'Windows', 'Mac')]
            [string] $EolConvention = 'Unix',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [switch] $UseTab,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [switch] $Force,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [Alias('Quiet')]
            [switch] $Silent
        );
        begin
        {
            $_anyNewLine = $_process_SectionsItem_anyNewLine;

            $_commentStartSectionT = '^\s*{0}\s*PROCESS\s+{1}\s+BEGIN\s*\(\s*([^\s,]+(?:\s*,\s*[^\s,]+)*)\s*\)\s*$';
            $_commentElseSectionT  = '^\s*{0}\s*PROCESS\s+{1}\s+ELSE\s+(?:(UNCOMMENT)|(INDENT)(?:\s+(BACK))?\s+([1-9][0-9]{{0,5}}))\s*$';
            $_commentEndSectionT   = '^\s*{0}\s*PROCESS\s+{1}\s+END\s*$';

            $_commentCCodeT        = '^\s*{0}\s*{1}\s(.*)$';

            $_secSplit             = '\s*,\s*';

            $_emptyLine            = '^\s*$';
            $_indentBackCharT      = '(?:[ ]{{1,{0}}}|\t)';
        }
        process
        {
            # Selecting key.
            $_key = [regex]::Escape($ProcessKey);

            # Selecting special comment prefix.
            switch -Exact ($CommentType)
            {
                'CMake' { $_commentPrefix = '##'; }
                'Cxx'   { $_commentPrefix = '//\s*##'; }
            }

            # Selecting EOL.
            switch -Exact ($EolConvention)
            {
                'Unix'    { $_eolChar = "`n";   }
                'Windows' { $_eolChar = "`r`n"; }
                'Mac'     { $_eolChar = "`r";   }
            }

            # Preparing patterns.
            $_commentStartSection = $_commentStartSectionT -f $_commentPrefix, $_key;
            $_commentElseSection  = $_commentElseSectionT  -f $_commentPrefix, $_key;
            $_commentEndSection   = $_commentEndSectionT   -f $_commentPrefix, $_key;

            $_commentCCode        = $_commentCCodeT -f $_commentPrefix, $_key;

            $_indentBackChar      = $_indentBackCharT -f $TabSize;

            if ($UseTab.IsPresent) { $_indentChar = "`t"; } else { $_indentChar = ' ' * $TabSize; }


            # Checking for source file.
            if (!(Test-Path -LiteralPath $SrcPath -PathType Leaf))
            {
                Write-Error ('Cannot find the file for processing: "{0}"' -f $SrcPath);
                return; 
            }

            [System.Collections.Generic.HashSet`1[[string]]] $_excSections = @($ExcludeSections | % { $_.Trim() } | Sort-Object -Unique);
            [System.Collections.Generic.LinkedList`1[[object]]] $_processState = @();
            [int] $_line = 0;

            Write-Verbose ("Starting processing section-aware file: `"{0}`"`n -- exclusions: {1}." -f $SrcPath, (@($_excSections | Sort-Object | % { '"{0}"' -f $_ }) -join ', '));
            $_processedContent = @($_anyNewLine.Split((Get-Content -Encoding UTF8 -LiteralPath $SrcPath -Force -Raw)) | % {
                    ++$_line;
                    switch -RegEx -CaseSensitive ($_)
                    {
                        $_commentStartSection { 
                                [System.Collections.Generic.HashSet`1[[string]]] $_sec = @(($_ -creplace $_commentStartSection, '$1') -csplit $_secSplit | % { $_.Trim() } | Sort-Object -Unique);
                                $_sec.ExceptWith($_excSections);

                                $_state = New-Object PSObject -Property @{
                                    Line     = $_line;
                                    Active   = ($_sec.Count -gt 0); # true -> section stays (then block) or it is indented/uncommented (else block).
                                    InElse   = $false;
                                    Indent   = 0;
                                    ElseMode = 'Uncomment';
                                };

                                $_processState.AddLast($_state) | Out-Null;
                                break;
                            }
                        $_commentElseSection  {
                                if (($_processState.Count -le 0) -or $_processState.Last.Value.InElse) { Write-Warning ('Encountered section ELSE without beginning: "{0}":{1}' -f $SrcPath, $_line); }
                                else
                                {
                                    $_processState.Last.Value.Active = !$_processState.Last.Value.Active;
                                    $_processState.Last.Value.InElse = $true;

                                    $_hasUncommentTk = ![string]::IsNullOrEmpty(($_ -replace $_commentElseSection, '$1'));
                                    $_hasIndentTk    = ![string]::IsNullOrEmpty(($_ -replace $_commentElseSection, '$2'));

                                    if ($_hasUncommentTk)  { $_processState.Last.Value.ElseMode = 'Uncomment'; }
                                    elseif ($_hasIndentTk)
                                    { 
                                        $_processState.Last.Value.ElseMode = 'Indent';

                                        if (![string]::IsNullOrEmpty(($_ -replace $_commentElseSection, '$3'))) { $_indentSize = -1 } else { $_indentSize = 1; }
                                        $_indentSize *= [int]::Parse(($_ -replace $_commentElseSection, '$4'), [cultureinfo]::InvariantCulture);
                                        $_processState.Last.Value.Indent = $_indentSize;
                                    }
                                    break;
                                }
                            }
                        $_commentEndSection   {
                                if ($_processState.Count -le 0) { Write-Warning ('Encountered section END without beginning: "{0}":{1}' -f $SrcPath, $_line); }
                                else
                                {
                                    $_processState.RemoveLast();
                                    break;
                                }
                            }
                        '^.*$'                {
                                $_uncommentLine = $false;
                                $_indentSize    = 0;      # 0 -> do not apply indent.
                                $_excludeLine   = $false;

                                $_processState | % {
                                    $_state = $_;
                                    if ($_state.InElse)
                                    {
                                        if ($_state.Active)
                                        {
                                            switch -Exact ($_state.ElseMode)
                                            {
                                                'Uncomment' { $_uncommentLine = $true; }        # line in any active nested ELSE block with "Uncomment" mode -> apply uncomment for line.
                                                'Indent'    { $_indentSize += $_state.Indent; } # line in any active nested ELSE block with "Indent" mode -> apply indent summed up from each active ELSE block with "Indent" mode.
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if (!$_state.Active) { $_excludeLine = $true; }
                                    }
                                }

                                if ($_excludeLine) { return; } # '-- '

                                $_text = $_;
                                if ($_uncommentLine) { $_text = $_text -creplace $_commentCCode, '$1'; }
                                if ($_text -cmatch $_emptyLine) { $_text = ''; }
                                elseif ($_indentSize -gt 0) { $_text = ($_indentChar * $_indentSize) + $_text; }
                                elseif ($_indentSize -lt 0) { $_text = $_text -creplace "^${_indentBackChar}{1,$(-$_indentSize)}", ''; }

                                # $_text = ' ' + $_text;
                                # if (!$_uncommentLine -and ($_indentSize -eq 0)) { $_text = '++' + $_text; }
                                # if ($_uncommentLine) { $_text = '^' + $_text; } elseif ($_indentSize -ne 0) { $_text = ' ' + $_text; }
                                # if ($_indentSize -gt 0)     { $_text = '>' + $_text; }
                                # elseif ($_indentSize -lt 0) { $_text = '<' + $_text; }
                                # elseif ($_uncommentLine) { $_text = ' ' + $_text; }

                                return $_text;
                            }
                    }
                } -End {
                    if ($_processState.Count -gt 0)
                    {
                        $_processState | % { Write-Warning ('Encountered section START without ending: "{0}":{1}' -f $SrcPath, $_.Line); }
                        $_processState.Clear();
                    }
                    Write-Verbose ("Processed section-aware file: `"{0}`"`n -- processed {1} lines`n -- exclusions: {2}." -f $SrcPath, $_line, (@($_excSections | Sort-Object | % { '"{0}"' -f $_ }) -join ', '));
                });


            # Checking and writing output.
            $DstPath | ? { [string]::IsNullOrEmpty($_) -or !(Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -or (Test-Path -LiteralPath $_ -PathType Container) } | % {
                Write-Error ('Destination path where updated sections-aware file should be written is invalid: "{0}"' -f $_);
            }
            [string[]] $_dstPath = @($DstPath | ? { ![string]::IsNullOrEmpty($_) -and (Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -and !(Test-Path -LiteralPath $_ -PathType Container) } | Sort-Object -Unique);
            if ($_dstPath.Count -le 0)
            {
                Write-Error ('There is no valid destination path for processed sections-aware file: "{0}"' -f $SrcPath);
                return;
            }

            if ($Silent.IsPresent -or $PSCmdlet.ShouldProcess(($_dstPath  -join ', '),  'Create/Update section-aware files'))
            {
                $_force = $Silent.IsPresent -or $Force.IsPresent;
                $_dstPath = @($_dstPath | % {
                        if ($_force -or !(Test-Path -LiteralPath $_ -PathType Leaf) -or $PSCmdlet.ShouldContinue(('Overwrite existing file: "{0}"' -f $_), 'Overwrite section-aware file', [ref] $_owDstYesToAll, [ref] $_owDstNoToAll)) { return $_; }
                    });

                if ($_dstPath.Count -gt 0)
                {
                    $_dstPath | % { $_dstDir = Split-Path -Parent $_; if (![string]::IsNullOrEmpty($_dstDir) -and !(Test-Path -LiteralPath $_dstDir)) { mkdir $_dstDir -Force:$_force -Confirm:$false | Out-Null; } };

                    Write-Verbose ("Creating/Updating sections-aware files:`n{0}" -f (@($_dstPath | % { ' -- "{0}"' -f $_ }) -join "`n"));
                    # Saved to variable to enable writing to the same file.
                    Set-Content ([System.Text.Encoding]::UTF8.GetBytes($_processedContent -join $_eolChar)) -LiteralPath $_dstPath -Encoding Byte -NoNewline -Force:$_force -Confirm:$false;
                }
            }
        }
        end {}
    }

    # done
    $_process_VersionItem_versionObjPattern = New-Object regex '"version"\s*:\s*\{\s*(.*)\s*\}', 'Singleline, Compiled';
    function _process-VersionItem
    {
        <#
            .SYNOPSIS
                Process version items by updating one of version field inside.

            .DESCRIPTION
                Process selected source item (version file) and updates selected field by adding step to its current value.
                Result is written to selected destination items.
                
                Source item can be at the same time destination item (updating file in place is allowed).

            .PARAMETER SrcPath
                Path to source item. Path must be valid and must point to (non-container) item which exists.

                Wildcards are NOT allowed.

            .PARAMETER DstPath
                Paths to destination items (non-container). If the item exists, it will be overwritten with updated content
                from source item. If it does not exists, it will be created.

                Wildcards are NOT allowed. Please use -Force switch, if you want to overwrite read-only files.

            .PARAMETER Position
                Field to update in version file.

                Allowed values: 'Major', 'Minor', 'Build'.

            .PARAMETER Step
                Value that will be added to selected field in version.

                Allowed range: (-16M) - 16M.

            .PARAMETER Force
                Force overwrite destination items (even if they are read-only).

            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([void])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [string] $SrcPath,
            [Parameter(Position = 1, Mandatory = $true,
                       ValueFromPipelineByPropertyName = $true)]
            [string[]] $DstPath,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('Major', 'Minor', 'Build')]
            [string] $Position = 'Build',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateRange(-16*1024*1024, 16*1024*1024)]
            [int] $Step = 1,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [switch] $Force,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [Alias('Quiet')]
            [switch] $Silent
        );
        begin {
            $_versionObjPattern = $_process_VersionItem_versionObjPattern;

            $_positionPatternT = '"{0}"\s*:\s*([-+]?[0-9]+)';
        }
        process
        {
            # Preparing patterns.
            $_positionPattern = New-Object regex ($_positionPatternT -f $Position.ToLowerInvariant()), 'Singleline';

            # Checking for source file.
            if (!(Test-Path -LiteralPath $SrcPath -PathType Leaf))
            {
                Write-Error ('Cannot find the file for processing: "{0}"' -f $SrcPath);
                return; 
            }

            $_versionText = Get-Content -Encoding UTF8 -LiteralPath $SrcPath -Force -Raw;

            $_versionObjMatch = $_versionObjPattern.Match($_versionText);
            if (!$_versionObjMatch.Success)
            {
                Write-Error ('Cannot find "version" JSON entry in file: "{0}"' -f $SrcPath);
                return;
            }

            $_positionMatch = $_positionPattern.Match($_versionObjMatch.Groups[1].Value);
            if (!$_positionMatch.Success)
            {
                Write-Error ('Cannot find "{1}" field in "version" JSON entry in file: "{0}"' -f $SrcPath, $Position.ToLowerInvariant());
                return;
            }

            $_positionIndex  = $_versionObjMatch.Groups[1].Index + $_positionMatch.Groups[1].Index;
            $_positionLength = $_positionMatch.Groups[1].Length;
            $_positionValue  = [int]::Parse($_positionMatch.Groups[1].Value, [CultureInfo]::InvariantCulture);

            $_positionValue += $Step;

            Write-Verbose ('Processed version file. Updating version of "{0}" to {1}: "{2}"' -f $Position.ToLowerInvariant(), $_positionValue, $SrcPath);
            $_updVersionText = $_versionText.Substring(0, $_positionIndex) + $_positionValue.ToString([CultureInfo]::InvariantCulture) + $_versionText.Substring($_positionIndex + $_positionLength);


            # Checking and writing output.
            $DstPath | ? { [string]::IsNullOrEmpty($_) -or !(Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -or (Test-Path -LiteralPath $_ -PathType Container) } | % {
                Write-Error ('Destination path where updated version file should be written is invalid: "{0}"' -f $_);
            }
            [string[]] $_dstPath = @($DstPath | ? { ![string]::IsNullOrEmpty($_) -and (Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -and !(Test-Path -LiteralPath $_ -PathType Container) } | Sort-Object -Unique);
            if ($_dstPath.Count -le 0)
            {
                Write-Error ('There is no valid destination path for updated version file: "{0}"' -f $SrcPath);
                return;
            }

            if ($Silent.IsPresent -or $PSCmdlet.ShouldProcess(($_dstPath  -join ', '),  'Create/Update version files'))
            {
                $_force = $Silent.IsPresent -or $Force.IsPresent;
                $_dstPath = @($_dstPath | % {
                        if ($_force -or !(Test-Path -LiteralPath $_ -PathType Leaf) -or $PSCmdlet.ShouldContinue(('Overwrite existing file: "{0}"' -f $_), 'Overwrite version file', [ref] $_owDstYesToAll, [ref] $_owDstNoToAll)) { return $_; }
                    });

                if ($_dstPath.Count -gt 0)
                {
                    $_dstPath | % { $_dstDir = Split-Path -Parent $_; if (![string]::IsNullOrEmpty($_dstDir) -and !(Test-Path -LiteralPath $_dstDir)) { mkdir $_dstDir -Force:$_force -Confirm:$false | Out-Null; } };

                    Write-Verbose ("Creating/Updating version files:`n{0}" -f (@($_dstPath | % { ' -- "{0}"' -f $_ }) -join "`n"));
                    Set-Content ([System.Text.Encoding]::UTF8.GetBytes($_updVersionText)) -LiteralPath $_dstPath -Encoding Byte -NoNewline -Force:$_force -Confirm:$false;
                }
            }
        }
        end {}
    }

    # done
    $_process_FormatItem_anyNewLine = New-Object regex '\r\n|[\v\r\f\n\x85\p{Zl}\p{Zp}]', 'Compiled';
    # Trim replace patterns.
    $_process_FormatItem_trimLinePattern_CMake = New-Object regex '(?<!(?:^|[^\\])(?:\\\\)*\\)\s+$', 'Compiled';
    $_process_FormatItem_trimLineReplace_CMake = '';
    $_process_FormatItem_trimLinePattern_Cxx   = $_process_FormatItem_trimLinePattern_CMake;
    $_process_FormatItem_trimLineReplace_Cxx   = $_process_FormatItem_trimLineReplace_CMake;
    function _process-FormatItem
    {
        <#
            .SYNOPSIS
                Process text-like items by formatting them line by line.

            .DESCRIPTION
                Process selected source item (usually text file) and formats it line by line and reformats
                them sligtly by:
                 -- removing trailing white-spaces from each line (if needed)
                 -- unifying/normalizing indent
                 -- changing and unifying EOL characters

                 UTF8 content is assumed. Written items are written using UTF8 encoding (BOM is not written).

            .PARAMETER SrcPath
                Path to source item. Path must be valid and must point to (non-container) text item which exists.

                Wildcards are NOT allowed.

            .PARAMETER DstPath
                Paths to destination items (non-container). If the item exists, it will be overwritten with updated content
                from source item. If it does not exists, it will be created.

                Written items are written using UTF8 encoding (BOM is not written).

                Wildcards are NOT allowed. Please use -Force switch, if you want to overwrite read-only files.

            .PARAMETER ItemType
                Type of processing item (type of file which will be formatted).

                Allowed values: 'CMake', 'Cxx'.

            .PARAMETER TrimWhitespace
                Removes excess/unnecessary whitespace from the end of each line.

            .PARAMETER NormalizeIndent
                Normalizes indent by converting them to spaces or tabulations.

                If 'Preserve' is selected, an indent is not normalized. If 'AlignedSpaces' or 'AlignedTabs'
                is selected, an indent is realigned to mutiply of tab size (for spaces) or only using tabulations
                (for tabs). Otherwise, if line is misaligned, an indent misalignment will be preserved (may
                result in intermingled tabs and spaces for 'Tabs').

                Allowed values: 'AlignedSpaces', 'AlignedTabs', 'Spaces', 'Tabs', 'Preserve'.

            .PARAMETER TabSize
                Number of spaces used as one tab (for replacing tab with spaces on indentations).

                Allowed range: 1 - 1024.

            .PARAMETER EolConvention
                End-of-line convention used to write/overwrite destination items (formatted content).

                New lines will be written as:
                 -- <LF>     for 'Unix'
                 -- <CR><LF> for 'Windows'
                 -- <CR>     for 'Mac'

                Allowed values: 'Unix', 'Windows', 'Mac'.

            .PARAMETER Force
                Force overwrite destination items (even if they are read-only).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([void])]
        param(
            [Parameter(Position = 0, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [string] $SrcPath,
            [Parameter(Position = 1, Mandatory = $true,
                       ValueFromPipelineByPropertyName = $true)]
            [string[]] $DstPath,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('CMake', 'Cxx')]
            [Alias('Type')]
            [string] $ItemType = 'Cxx',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [Alias('TrimWs', 'Trim')]
            [switch] $TrimWhitespace,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('AlignedSpaces', 'AlignedTabs', 'Spaces', 'Tabs', 'Preserve')]
            [string] $NormalizeIndent = 'Preserve',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateRange(1, 1024)]
            [int] $TabSize = 2,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('Unix', 'Windows', 'Mac')]
            [string] $EolConvention = 'Unix',
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [switch] $Force,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [Alias('Quiet')]
            [switch] $Silent
        );
        begin
        {
            $_anyNewLine = $_process_FormatItem_anyNewLine;

            $_trimLinePattern_CMake = $_process_FormatItem_trimLinePattern_CMake;
            $_trimLineReplace_CMake = $_process_FormatItem_trimLineReplace_CMake;
            $_trimLinePattern_Cxx   = $_process_FormatItem_trimLinePattern_Cxx;
            $_trimLineReplace_Cxx   = $_process_FormatItem_trimLineReplace_Cxx;

            $_indentAlignedPatternT = '(?:^|\G)(?:[ ]{{1,{0}}}|\t)'; # realign everything to proper multiply of spaces (result is always aligned to TabSize or using only tabs).
            $_indentMixedPatternT   = '(?<=^\s*)(?:[ ]{{{0}}}|\t)';  # only realign matching multiply of spaces (so e.g. after leading tabs, there can be spaces not aligned to TabSize or tabs are
                                                                     # intermingled with spaces).
        }
        process
        {
            # Selecting special comment prefix.
            switch -Exact ($ItemType)
            {
                'CMake' { $_trimLinePattern = $_trimLinePattern_CMake; $_trimLineReplace = $_trimLineReplace_CMake; }
                'Cxx'   { $_trimLinePattern = $_trimLinePattern_Cxx;   $_trimLineReplace = $_trimLineReplace_Cxx;   }
            }

            # Selecting indent normalization..
            switch -Exact ($NormalizeIndent)
            {
                'AlignedSpaces' { $_normalizeIndent = $true;  $_indentPatternS = $_indentAlignedPatternT -f $TabSize; $_indentReplace = ' ' * $TabSize; }
                'AlignedTabs'   { $_normalizeIndent = $true;  $_indentPatternS = $_indentAlignedPatternT -f $TabSize; $_indentReplace = "`t"; }
                'Spaces'        { $_normalizeIndent = $true;  $_indentPatternS = $_indentMixedPatternT   -f $TabSize; $_indentReplace = ' ' * $TabSize; }
                'Tabs'          { $_normalizeIndent = $true;  $_indentPatternS = $_indentMixedPatternT   -f $TabSize; $_indentReplace = "`t"; }
                'Preserve'      { $_normalizeIndent = $false; $_indentPatternS = '';                                  $_indentReplace = '';   }
            }
            $_indentPattern = New-Object regex $_indentPatternS, 'Compiled';

            # Selecting EOL.
            switch -Exact ($EolConvention)
            {
                'Unix'    { $_eolChar = "`n";   }
                'Windows' { $_eolChar = "`r`n"; }
                'Mac'     { $_eolChar = "`r";   }
            }

            # Checking for source file.
            if (!(Test-Path -LiteralPath $SrcPath -PathType Leaf))
            {
                Write-Error ('Cannot find the file for formatting: "{0}"' -f $SrcPath);
                return; 
            }

            Write-Verbose ('Starting formatting content of file: "{0}".' -f $SrcPath);
            $_processedContent = @($_anyNewLine.Split((Get-Content -Encoding UTF8 -LiteralPath $SrcPath -Force -Raw)) | % {
                    ++$_line;
                    $_processedLine = $_;
                    if ($TrimWhitespace.IsPresent) { $_processedLine = $_trimLinePattern.Replace($_processedLine, $_trimLineReplace); }
                    if ($_normalizeIndent)         { $_processedLine = $_indentPattern.Replace($_processedLine, $_indentReplace); }
                    return $_processedLine;
                } -End {
                    Write-Verbose ("Formatted file: `"{0}`"`n -- processed {1} lines." -f $SrcPath, $_line);
                });


            # Checking and writing output.
            $DstPath | ? { [string]::IsNullOrEmpty($_) -or !(Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -or (Test-Path -LiteralPath $_ -PathType Container) } | % {
                Write-Error ('Destination path where formatted file should be written is invalid: "{0}"' -f $_);
            }
            [string[]] $_dstPath = @($DstPath | ? { ![string]::IsNullOrEmpty($_) -and (Test-Path -LiteralPath $_ -PathType Leaf -IsValid) -and !(Test-Path -LiteralPath $_ -PathType Container) } | Sort-Object -Unique);
            if ($_dstPath.Count -le 0)
            {
                Write-Error ('There is no valid destination path for formatted file: "{0}"' -f $SrcPath);
                return;
            }

            if ($Silent.IsPresent -or $PSCmdlet.ShouldProcess(($_dstPath  -join ', '),  'Format content of files'))
            {
                $_force = $Silent.IsPresent -or $Force.IsPresent;
                $_dstPath = @($_dstPath | % {
                        if ($_force -or !(Test-Path -LiteralPath $_ -PathType Leaf) -or $PSCmdlet.ShouldContinue(('Overwrite existing file: "{0}"' -f $_), 'Overwrite file', [ref] $_owDstYesToAll, [ref] $_owDstNoToAll)) { return $_; }
                    });

                if ($_dstPath.Count -gt 0)
                {
                    $_dstPath | % { $_dstDir = Split-Path -Parent $_; if (![string]::IsNullOrEmpty($_dstDir) -and !(Test-Path -LiteralPath $_dstDir)) { mkdir $_dstDir -Force:$_force -Confirm:$false | Out-Null; } };

                    Write-Verbose ("Formatting content of files:`n{0}" -f (@($_dstPath | % { ' -- "{0}"' -f $_ }) -join "`n"));
                    # Saved to variable to enable writing to the same file.
                    Set-Content ([System.Text.Encoding]::UTF8.GetBytes($_processedContent -join $_eolChar)) -LiteralPath $_dstPath -Encoding Byte -NoNewline -Force:$_force -Confirm:$false;
                }
            }
        }
        end {}
    }

    # done
    <#
        .SYNOPSIS
            Extracts user name from fully qualified domain name (FQDN).

        .PARAMETER DomainQualName
            Fully qualified domain name. Both standard forms are supported:
             -- <dom>\<username>
             -- <username>@dom

            If name is not qualified, the returned name is the same as input name.
    #>
    function _extract-UserName([string] $DomainQualName) { return $DomainQualName.Trim() -replace '^.*\\([^\\]+)$|^([^@]+)@.*$', '$1$2'; }

    # done
    function _get-VcsDefaultUser
    {
        <#
            .SYNOPSIS
                Returns name of default user for selected version control system.

            .PARAMETER VcsType
                Type of version control system used.

                Allowed values: 'Git', 'Perforce'.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param(
            [Parameter(Position = 0, Mandatory = $false)]
            [ValidateSet('Git', 'Perforce')]
            [Alias('Type', 'Vcs')]
            [string] $VcsType = 'Git'
        );

        switch -Exact ($VcsType)
        {
            'Perforce' { if (![string]::IsNullOrWhiteSpace($env:P4USER)) { return $env:P4USER.Trim(); } }
        }

        try
        {
            $_domainQualName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name;
            return _extract-UserName $_domainQualName;
        }
        catch
        {
            return $env:USERNAME.Trim();
        }
    }

    # done
    function _get-VcsDefaultServer
    {
        <#
            .SYNOPSIS
                Returns name of default server (port) for selected version control system.

            .PARAMETER VcsType
                Type of version control system used.

                Allowed values: 'Git', 'Perforce'.
        #>

        [CmdletBinding()]
        [OutputType([string])]
        param(
            [Parameter(Position = 0, Mandatory = $false)]
            [ValidateSet('Git', 'Perforce')]
            [Alias('Type', 'Vcs')]
            [string] $VcsType = 'Git'
        );

        switch -Exact ($VcsType)
        {
            'Perforce' { if (![string]::IsNullOrWhiteSpace($env:P4PORT)) { return $env:P4PORT.Trim(); } }
        }
        return '';
    }

    # done
    function _login-Vcs
    {
        <#
            .SYNOPSIS
                Log into selected version control system.

            .DESCRIPTION
                Tries to log into selected (local) VCS. Uses server URL and credentials, if VCS
                supports these parameters.

            .PARAMETER VcsType
                Type of version control system used.

                Allowed values: 'Git', 'Perforce'.

            .PARAMETER VcsServer
                URL/URI to server/port/entrypoint of version control system.

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsCredential
                Credentials for accessing server/port/entrypoint of version control system.

                This parameter is only used by specific types of VCS and ignored by the rest.

                When this parameter is specified and non-null, the -PromptVcsCredential is ignored.

            .PARAMETER PromptVcsCredential
                Prompts for credentials for accessing server/port/entrypoint of version
                control system. Usually it is in the form of dialog with default user name for VCS
                specified.

                This parameter is only used by specific types of VCS and ignored by the rest.

                When VcsCredential parameter is specified and non-null, current parameter is ignored.

            .PARAMETER VcsWorkspace
                Expected workspace in VCS. If it is specified, the destination location is
                checked to see whether it is contained in specified workspace. If location
                is outside workspace or workspace is absent in VCS, the confirmation dialog
                will be shown (whether to continue).

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsBranch
                Expected branch/stream in VCS. If it is specified, the destination location is
                checked to see whether it is switch to selected branch/stream. If location
                is not set to use specified branch/stream or branch/stream is absent in VCS,
                the confirmation dialog will be shown (whether to continue).

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsChangelist
                Changelist where to stage files in VCS.

                If 'New' is selected, the new changelist will be created for staged files.
                If 'Default' is selected, any files will be staged in default changelist.
                If number is specified, the files will be staged on selected changelist.

                Allowed values: 'New', 'Default', <any non-negative integral number>

                This parameter is only used by specific types of VCS and ignored by the rest.

            .OUTPUTS
                [PSObject] - VCS token (login token to version control system).

                Token contains information whether operation failed or not. Also it contains
                required properties and currently selected configuration.
        #>

        [CmdletBinding()]
        [OutputType([PSObject])]
        param(
            [Parameter(Position = 0, Mandatory = $false)]
            [ValidateSet('Git', 'Perforce')]
            [Alias('Type', 'Vcs')]
            [string] $VcsType = 'Git',

            [Parameter(Mandatory = $false)]
            [Alias('Server')]
            [string] $VcsServer = '',
            [Parameter(Mandatory = $false)]
            [Alias('Credential')]
            [System.Management.Automation.PSCredential] $VcsCredential = $null,
            [Parameter(Mandatory = $false)]
            [Alias('PromptCredential')]
            [switch] $PromptVcsCredential,
            [Parameter(Mandatory = $false)]
            [Alias('Workspace')]
            [string] $VcsWorkspace = '',
            [Parameter(Mandatory = $false)]
            [Alias('Branch', 'VcsStream', 'Stream')]
            [string] $VcsBranch = '',
            [Parameter(Mandatory = $false)]
            [ValidatePattern('^(?:[0-9]+|Default|New)$')]
            [Alias('Changelist', 'VcsChange', 'Change')]
            [string] $VcsChangelist = 'New'
        );

        $_success    = $false;
        $_aborted    = $false;
        $_state      = New-Object 'System.Collections.Generic.HashSet`1[[string]]';
        $_toolParams = @();

        $_propsType       = @{'Git' = 'Git'; 'Perforce' = 'Perforce'}[$VcsType]; # Cannonize.
        $_propsTool       = @{'Git' = 'git'; 'Perforce' = 'p4'}[$_propsType];
        $_propsServer     = $VcsServer.Trim();
        $_propsUser       = _get-VcsDefaultUser $_propsType;
        $_propsCredential = $VcsCredential;
        $_propsWorkspace  = $VcsWorkspace.Trim();
        $_propsBranch     = $VcsBranch.Trim();
        $_propsChange     = $VcsChangelist.Trim().ToLowerInvariant();

        if ([string]::IsNullOrEmpty($_propsServer))
        {
            $_propsServer = _get-VcsDefaultServer $_propsType;
        }

        if (($_propsCredential -eq $null) -and $PromptVcsCredential.IsPresent)
        {
            $_propsCredential = Get-Credential -UserName $_propsUser -Message ('Provide login/password for {0} VCS: "{1}"' -f $_propsType, $_propsServer);
        }
        if ($_propsCredential -eq $null)
        {
            $_propsCredential = New-object 'System.Management.Automation.PSCredential' $_propsUser, (New-Object 'System.Security.SecureString');
        }

        $_propsUser = _extract-UserName $_propsCredential.UserName;


        $_toolInfo = @(Get-Command $_propsTool -ErrorAction Ignore);
        if ($_toolInfo.Count -gt 0)
        {
            $_success = $true;
        }
        else
        {
            Write-Error ('VCS: {0}: Command-line tool for {0} VCS is not available. Please make it available in PATH: "{1}".' -f $_propsType, $_propsTool);
        }

        $_props = New-Object PSObject |
            Add-Member 'Type'       $_propsType       -PassThru |
            Add-Member 'Tool'       $_propsTool       -PassThru |
            Add-Member 'Server'     $_propsServer     -PassThru |
            Add-Member 'User'       $_propsUser       -PassThru |
            Add-Member 'Credential' $_propsCredential -PassThru |
            Add-Member 'Workspace'  $_propsWorkspace  -PassThru |
            Add-Member 'Branch'     $_propsBranch     -PassThru |
            Add-Member 'Change'     $_propsChange     -PassThru;

        switch -Exact ($_props.Type)
        {
            'Git'      {} # No login required.
            'Perforce' {
                    if (![string]::IsNullOrEmpty($_props.Server)) { $_cmdServer = @('-p', $_props.Server); } else { $_cmdServer = @(); }
                    if (![string]::IsNullOrEmpty($_props.User))   { $_cmdUser   = @('-u', $_props.User); }   else { $_cmdUser = @(); }

                    $_cmdAll = @($_cmdServer; $_cmdUser);

                    if ($_success)
                    {
                        Write-Verbose ('VCS: {0}: Logging into "{1}" as "{2}".' -f $_props.Type, $_props.Server, $_props.User); 
                        # Command: p4 <opts> login
                        $_props.Credential.GetNetworkCredential().Password | & $_props.Tool $_cmdAll 'login' 2>&1 | Out-Null;
                        $_success = $? -and ($LASTEXITCODE -eq 0);
                    }

                    if ($_success) { $_toolParams = $_cmdAll; }
                    else
                    {
                        Write-Error ('VCS: {0}: Failed to log into selected VCS: "{1}".' -f $_props.Type, $_props.Server);
                    }
                }
        }

        $_state.Add('Login') | Out-Null;
        return New-Object PSObject |
            Add-Member 'Success'    $_success    -PassThru |
            Add-Member 'Aborted'    $_aborted    -PassThru |
            Add-Member 'State'      $_state      -PassThru |
            Add-Member 'ToolParams' $_toolParams -PassThru |
            Add-Member 'Props'      $_props      -PassThru;
    }

    # done
    function _set-VcsWorkspace
    {
        <#
            .SYNOPSIS
                Select workspace in version control system.

            .DESCRIPTION
                If workspace is specified (during logging in) and VCS supports workspace selection,
                the VCS is checked for workspace existence at currrent directory. If it not specified or
                VCS does not support workspace, current directory is checked whether it is mapped in VCS.

                If workspace is specified during loging and it does not exist in VCS, the failed token
                is returned. If workspace exists, the current directory is checked whether it is mapped
                to selected workspace. If it not, continue confirmation is presented. Depending on it
                successfull or failed token is returned.

                If workspace is not specified, the workspace is selected based on
                current directory mappings (current directory is checked whether it is mapped into VCS
                and into which workspace - first is selected). If workspace cannot be selected, the
                failed token is returned.

                If workspace is not supported by VCS, only current directory is checked whether it
                is mapped into VCS. If current directory is not mapped, the failed token is returned.
            
            .PARAMETER VcsToken
                Token from _login-Vcs.

            .PARAMETER Force
                Force operation (even if the expected workspace does not match mapped workspace for
                current directory).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.

            .OUTPUTS
                [PSObject] - Updated VCS token.

                Token contains information whether operation failed or not. Also it contains
                required properties and currently selected configuration.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([PSObject])]
        param(
            [Parameter(Position = 0, Mandatory = $true)]
            [Alias('Token')]
            [PSObject] $VcsToken,
            [Parameter(Mandatory = $false)]
            [switch] $Force,
            [Parameter(Mandatory = $false)]
            [Alias('Quiet')]
            [switch] $Silent
        );

        if (($VcsToken -eq $null) -or !$VcsToken.Success)
        {
            Write-Error 'VCS: Failed to select workspace in VCS. VCS token is invalid.';
            return $VcsToken;
        }
        if (!$VcsToken.State.Contains('Login'))
        {
            Write-Error 'VCS: Failed to select workspace in VCS. VCS token is invalid (not logged in).';
            return $VcsToken;
        }
        if ($VcsToken.Aborted)
        {
            return $VcsToken;
        }

        $_success    = $true;
        $_aborted    = $false;
        $_state      = $VcsToken.State;
        $_toolParams = $VcsToken.ToolParams;
        $_props      = $VcsToken.Props;

        switch -Exact ($_props.Type)
        {
            'Git'      {
                    $_currentDir = (Get-Location).Path;
                    $_cmdAll = @($_toolParams);

                    Write-Verbose ('VCS: {0}: Checking current directory mapping in VCS: "{1}".' -f $_props.Type, $_currentDir); 
                    # Command: git <opts> status
                    & $_props.Tool $_cmdAll 'status' 2>&1 | Out-Null;
                    $_success = $? -and ($LASTEXITCODE -eq 0);

                    if (!$_success)
                    {
                        Write-Error ('VCS: {0}: Current directory is not mapped in VCS: "{1}".' -f $_props.Type, $_currentDir);
                    }
                }
            'Perforce' {
                    if (![string]::IsNullOrEmpty($_props.User)) { $_propsUser = $_props.User; } else { $_propsUser = _get-VcsDefaultUser $_props.Type; }
                    $_cmdUser = @('-u', $_propsUser);

                    if (![string]::IsNullOrEmpty($_props.Workspace)) { $_cmdWorkspace = @('-c', $_props.Workspace); } else { $_cmdWorkspace = @(); }

                    # Enumerating workspaces.
                    $_cmdAll      = @($_toolParams);
                    $_cmdSpecific = @($_cmdUser);

                    Write-Verbose ('VCS: {0}: Checking for available workspaces in "{1}" (user: "{2}").' -f $_props.Type, $_props.Server, $_propsUser);
                    # Command: p4 <opts> clients -u <user>
                    $_workspaces = @(& $_props.Tool $_cmdAll 'clients' $_cmdSpecific 2>&1 | ? { ($_ -is [string]) -and ($_ -match '^Client\s+\S+.*$') } | % { $_ -replace '^Client\s+(\S+).*$', '$1' });
                    $_success = $? -and ($LASTEXITCODE -eq 0);

                    # Checking workspace availability.
                    if ($_success)
                    {
                        Write-Verbose ('VCS: {0}: Found {1} workspace(s): {2}.' -f $_props.Type, $_workspaces.Count, (@($_workspaces | % { "`"$_`"" }) -join ', '));

                        if (![string]::IsNullOrEmpty($_props.Workspace) -and ($_props.Workspace -cnotin $_workspaces))
                        {
                            Write-Error ('VCS: {0}: Selected workspace is not available in VCS: "{1}".' -f $_props.Type, $_props.Workspace);
                            $_success = $false;
                        }
                    }

                    # Checking current directory for VCS workspace mapping.
                    if ($_success)
                    {
                        $_currentDir = (Get-Location).Path;

                        Write-Verbose ('VCS: {0}: Checking current directory mapping in VCS: "{1}" (server: "{2}", user: "{3}").' -f $_props.Type, $_currentDir, $_props.Server, $_propsUser); 
                        $_mappedWorkspaces = @($_workspaces | ? {
                                $_cmdAll = @($_toolParams; @('-c', $_));

                                Write-Verbose ('VCS: {0}: Checking current directory mapping in VCS: "{1}" (workspace: "{2}").' -f $_props.Type, $_currentDir, $_);
                                # Command: p4 <opts> -c <workspace> where ...
                                & $_props.Tool $_cmdAll 'where' '...' 2>&1 | Out-Null;
                                return $? -and ($LASTEXITCODE -eq 0);
                            });
                        
                        if ($_mappedWorkspaces.Count -le 0)
                        {
                            Write-Error ('VCS: {0}: Current directory is not mapped in VCS: "{1}".' -f $_props.Type, $_currentDir);
                            $_success = $false;
                        }
                    }

                    if ($_success)
                    {
                        $_force = $Silent.IsPresent -or $Force.IsPresent;
                        Write-Verbose ('VCS: {0}: Current directory is mapped in {1} workspace(s): {2}.' -f $_props.Type, $_mappedWorkspaces.Count, (@($_mappedWorkspaces | % { "`"$_`"" }) -join ', '));

                        if (![string]::IsNullOrEmpty($_props.Workspace) -and ($_props.Workspace -cnotin $_mappedWorkspaces))
                        {
                            if ($_force -or $PSCmdlet.ShouldContinue(('Current directory "{0}" is not mapped in selected workspace: "{1}". It will be switched to "{2}". Do you want to continue?' -f $_currentDir, $_props.Workspace, $_mappedWorkspaces[0]), 'Workspace mapping'))
                            {
                                $_props.Workspace = $_mappedWorkspaces[0];
                                $_cmdWorkspace = @('-c', $_props.Workspace);
                            }
                            else
                            {
                                Write-Verbose ('VCS: {0}: User aborted operation.' -f $_props.Type);
                                $_aborted = $true;
                                $_cmdWorkspace = @();
                            }
                        }
                        elseif ([string]::IsNullOrEmpty($_props.Workspace))
                        {
                            $_props.Workspace = $_mappedWorkspaces[0];
                            $_cmdWorkspace = @('-c', $_mappedWorkspaces[0]);
                        }

                        # Compile new tool parameters.
                        if ($_state.Contains('SelectWorkspace'))
                        {
                            $_removeParam = $false;
                            $_toolParams = @($_toolParams | ? { if ($_ -ceq '-c') { $_removeParam = $true; return $false; } elseif ($_removeParam) { $_removeParam = $false; return $false; } else { return $true; } });
                        }
                        $_toolParams = @($_toolParams; $_cmdWorkspace);
                    }
                }
        }

        $_state.Add('SelectWorkspace') | Out-Null;
        $VcsToken.Success    = $_success;
        $VcsToken.Aborted    = $_aborted;
        $VcsToken.State      = $_state;
        $VcsToken.ToolParams = $_toolParams;
        $VcsToken.Props      = $_props;

        return $VcsToken;
    }

    # done
    function _set-VcsBranch
    {
        <#
            .SYNOPSIS
                Select branch/stream in version control system.

            .DESCRIPTION
                If branch/stream is specified (during logging in) and VCS supports branch/stream selection,
                the VCS is checked for selected branch/stream at currrent directory. If it not specified or
                VCS does not support branch/stream, no operation is done.

                If branch/stream is specified during loging and it does not exist in VCS, the failed token
                is returned. If branch/stream exists, the current directory is checked whether it is mapped
                to selected branch/stream. If it not, continue confirmation is presented. Depending on it
                successfull or failed token is returned.

                If branch/stream is not specified, no checking/operation is done.
            
            .PARAMETER VcsToken
                Token from _set-VcsWorkspace.

            .PARAMETER Force
                Force operation (even if the expected branch/stream does not match mapped branch/stream for
                current directory).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.

            .OUTPUTS
                [PSObject] - Updated VCS token.

                Token contains information whether operation failed or not. Also it contains
                required properties and currently selected configuration.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([PSObject])]
        param(
            [Parameter(Position = 0, Mandatory = $true)]
            [Alias('Token')]
            [PSObject] $VcsToken,
            [Parameter(Mandatory = $false)]
            [switch] $Force,
            [Parameter(Mandatory = $false)]
            [Alias('Quiet')]
            [switch] $Silent
        );

        if (($VcsToken -eq $null) -or !$VcsToken.Success)
        {
            Write-Error 'VCS: Failed to select workspace in VCS. VCS token is invalid.';
            return $VcsToken;
        }
        if (!$VcsToken.State.Contains('SelectWorkspace'))
        {
            Write-Error 'VCS: Failed to select branch in VCS. VCS token is invalid (no workspace selected).';
            return $VcsToken;
        }
        if ($VcsToken.Aborted)
        {
            return $VcsToken;
        }

        $_success    = $true;
        $_aborted    = $false;
        $_state      = $VcsToken.State;
        $_toolParams = $VcsToken.ToolParams;
        $_props      = $VcsToken.Props;

        switch -Exact ($_props.Type)
        {
            'Git'      {
                    $_currentDir = (Get-Location).Path;
                    $_cmdAll = @($_toolParams);

                    Write-Verbose ('VCS: {0}: Checking for available branches at current directory in VCS: "{1}".' -f $_props.Type, $_currentDir); 
                    # Command: git <opts> branch --no-color
                    $_branches = @(& $_props.Tool $_cmdAll 'branch' '--no-color' 2>&1 | ? { ($_ -is [string]) -and ![string]::IsNullOrEmpty($_) } | % {
                            $_trackedBranch = $_ -cmatch '^\s*\*\s*.+?\s*$';
                            $_branchName    = $_ -creplace '^\s*(?:\*\s*)?(.+?)\s*$', '$1';
                            return New-Object PSObject |
                                Add-Member 'Tracked' $_trackedBranch -PassThru |
                                Add-Member 'Name'    $_branchName    -PassThru;
                        });
                    $_success = $? -and ($LASTEXITCODE -eq 0);

                    # Checking branch availability.
                    if ($_success)
                    {
                        Write-Verbose ("VCS: {0}: Found {1} branche(s):`n{2}." -f $_props.Type, $_branches.Count, (@($_branches | % { if ($_.Tracked) { $_prefix = '=>'; } else { $_prefix = '--'; } return ' {0} "{1}"' -f $_prefix, $_.Name; }) -join "`n"));

                        if (![string]::IsNullOrEmpty($_props.Branch) -and ($_props.Branch -cnotin @($_branches.Name)))
                        {
                            Write-Error ('VCS: {0}: Selected branch is not available in VCS: "{1}".' -f $_props.Type, $_props.Branch);
                            $_success = $false;
                        }
                    }

                    # Checking current directory for VCS branch mapping.
                    if ($_success)
                    {
                        $_mappedBranches = @($_branches | ? { $_.Tracked } | % { $_.Name });
   
                        if ($_mappedBranches.Count -le 0)
                        {
                            Write-Error ('VCS: {0}: Current directory is not mapped in VCS: "{1}".' -f $_props.Type, $_currentDir);
                            $_success = $false;
                        }
                    }

                    if ($_success)
                    {
                        $_force = $Silent.IsPresent -or $Force.IsPresent;
                        Write-Verbose ('VCS: {0}: Current directory is mapped in {1} branch(es): {2}.' -f $_props.Type, $_mappedBranches.Count, (@($_mappedBranches | % { "`"$_`"" }) -join ', '));

                        if (![string]::IsNullOrEmpty($_props.Branch) -and ($_props.Branch -cnotin $_mappedBranches))
                        {
                            if ($_force -or $PSCmdlet.ShouldContinue(('Current directory "{0}" is not mapped in selected branch: "{1}". It will be switched to "{2}". Do you want to continue?' -f $_currentDir, $_props.Branch, $_mappedBranches[0]), 'Branch mapping'))
                            {
                                $_props.Branch = $_mappedBranches[0];
                            }
                            else
                            {
                                Write-Verbose ('VCS: {0}: User aborted operation.' -f $_props.Type);
                                $_aborted = $true;
                            }
                        }
                        elseif ([string]::IsNullOrEmpty($_props.Branch)) { $_props.Branch = $_mappedBranches[0]; }
                    }
                }
            'Perforce' {
                    # Enumerating streams in selected workspace.
                    $_cmdAll = @($_toolParams);

                    Write-Verbose ('VCS: {0}: Checking for available streams in "{1}" (workspace: "{2}").' -f $_props.Type, $_props.Server, $_props.Workspace); 
                    # Command: p4 <opts> streams
                    $_streams = @(& $_props.Tool $_cmdAll 'streams' 2>&1 |
                            ? { ($_ -is [string]) -and ($_ -match '^Stream\s+\S+\s+\S+\s+\S+\s+''.*''(?:\s+.*)?$') } |
                            % { $_ -replace '^Stream\s+\S+\s+\S+\s+\S+\s+''(.*)''(?:\s+.*)?$', '$1' }
                        );
                    $_success = $? -and ($LASTEXITCODE -eq 0);

                    # Checking stream availability.
                    if ($_success)
                    {
                        Write-Verbose ("VCS: {0}: Found {1} stream(s):`n{2}." -f $_props.Type, $_streams.Count, (@($_streams | % { " -- `"$_`"" }) -join "`n"));

                        if (![string]::IsNullOrEmpty($_props.Branch) -and ($_props.Branch -cnotin $_streams))
                        {
                            Write-Error ('VCS: {0}: Selected stream is not available in VCS: "{1}" (workspace: "{2}").' -f $_props.Type, $_props.Branch, $_props.Workspace);
                            $_success = $false;
                        }
                    }

                    # Checking current workspace for VCS stream mapping (selected stream).
                    if ($_success)
                    {
                        Write-Verbose ('VCS: {0}: Getting information about selected stream in current workspace in VCS: "{1}".' -f $_props.Type, $_props.Workspace); 
                        # Command: p4 <opts> stream -o
                        $_mappedStreams = @(& $_props.Tool $_cmdAll 'stream' '-o' 2>&1 |
                                ? { ($_ -is [string]) -and ($_ -match '^\s*Name\s*:\s*.+?\s*$') } |
                                % { $_ -replace '^\s*Name\s*:\s*(.+?)\s*$', '$1' }
                            );
                        if (!$? -or ($LASTEXITCODE -ne 0)) { $_mappedStreams = @(); }
                        
                        if (![string]::IsNullOrEmpty($_props.Branch) -and ($_mappedStreams.Count -le 0))
                        {
                            Write-Warning ('VCS: {0}: Current workspace does not contain streams or does not support streams in VCS: "{1}".' -f $_props.Type, $_props.Workspace);
                        }
                    }

                    if ($_success)
                    {
                        $_force = $Silent.IsPresent -or $Force.IsPresent;
                        Write-Verbose ('VCS: {0}: Current workspace is set to use {1} following stream(s): {2}.' -f $_props.Type, $_mappedStreams.Count, (@($_mappedStreams | % { "`"$_`"" }) -join ', '));

                        if (![string]::IsNullOrEmpty($_props.Branch) -and ($_props.Branch -cnotin $_mappedStreams))
                        {
                            if ($_force -or $PSCmdlet.ShouldContinue(('Current workspace "{0}" is not using specified stream: "{1}". The stream named "{2}" will be used. Do you want to continue?' -f $_props.Workspace, $_props.Branch, $_mappedStreams[0]), 'Stream mapping'))
                            {
                                $_props.Branch = $_mappedStreams[0];
                            }
                            else
                            {
                                Write-Verbose ('VCS: {0}: User aborted operation.' -f $_props.Type);
                                $_aborted = $true;
                            }
                        }
                        elseif ([string]::IsNullOrEmpty($_props.Branch)) { $_props.Branch = $_mappedStreams[0]; }
                    }
                }
        }

        $_state.Add('SelectBranch') | Out-Null;
        $VcsToken.Success    = $_success;
        $VcsToken.Aborted    = $_aborted;
        $VcsToken.State      = $_state;
        $VcsToken.ToolParams = $_toolParams;
        $VcsToken.Props      = $_props;

        return $VcsToken;
    }

    # done
    function _chmod-VcsFiles
    {
        <#
            .SYNOPSIS
                Changes mode of selected files. Stages files if necessary.

            .DESCRIPTION
                For selected files (non-existing and non-file paths are filtered out), the function
                modifies their attributes (mode) in VCS. It stages/prepares changelist if necessary.
            
            .PARAMETER VcsToken
                Token from _set-VcsWorkspace or _set-VcsBranch.

            .PARAMETER DstRelFiles
                List of relative paths to files which mode (file attributes in VCS) will be modified.
                Files will be staged to be able to change their mode.
                
                For some VCS (Git) only files which are effectively modified (content or attributes) will
                be staged. For others (Perforce), all selected files will be added to specific change list. 

                List can be empty. Paths are assumed to be relative to current directory.

            .PARAMETER Mode
                File mode (attribute) to set.

                Allowed values: '-x', '+x' (clear/set executive bit; for Perforce script ignores '-x').

            .PARAMETER Force
                Force operation (even if the expected branch/stream does not match mapped branch/stream for
                current directory).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.

            .OUTPUTS
                [PSObject] - Updated VCS token.

                Token contains information whether operation failed or not. Also it contains
                required properties and currently selected configuration.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([PSObject])]
        param(
            [Parameter(Position = 0, Mandatory = $true)]
            [Alias('Token')]
            [PSObject] $VcsToken,
            [Parameter(Position = 1, Mandatory = $true)]
            [Alias('RelFiles', 'Files')]
            [AllowEmptyCollection()]
            [string[]] $DstRelFiles,
            [Parameter(Mandatory = $false)]
            [ValidatePattern('^[+-]x$', Options = 'None')]
            [Alias('Chmod')]
            [string] $Mode = '+x',
            [Parameter(Mandatory = $false)]
            [switch] $Force,
            [Parameter(Mandatory = $false)]
            [Alias('Quiet')]
            [switch] $Silent
        );

        if (($VcsToken -eq $null) -or !$VcsToken.Success)
        {
            Write-Error 'VCS: Failed to select workspace in VCS. VCS token is invalid.';
            return $VcsToken;
        }
        if (!$VcsToken.State.Contains('SelectBranch') -and !$VcsToken.State.Contains('SelectWorkspace'))
        {
            Write-Error 'VCS: Failed to change mode of files in VCS. VCS token is invalid (no workspace and/or branch selected).';
            return $VcsToken;
        }
        if ($VcsToken.Aborted)
        {
            return $VcsToken;
        }

        $_success    = $true;
        $_aborted    = $false;
        $_state      = $VcsToken.State;
        $_toolParams = $VcsToken.ToolParams;
        $_props      = $VcsToken.Props;

        $_relFiles = @($DstRelFiles | ? {
                if (![string]::IsNullOrEmpty($_) -and (Test-Path -LiteralPath $_ -PathType Leaf)) { return $true; }
                Write-Warning ('VCS: {0}: Following relative path does not point to a file: "{1}"' -f $_props.Type, $_);
            });
        $_relFileGroups = @($_relFiles | % -Begin {
                $_groupIdx  = 0;
                $_fileGroup = New-Object PSObject |
                    Add-Member 'Index'  $_groupIdx                                                       -PassThru |
                    Add-Member 'Length' 0                                                                -PassThru |
                    Add-Member 'List'   (New-Object 'System.Collections.Generic.LinkedList`1[[string]]') -PassThru;
            } -Process {
                if (($_fileGroup.List.Count -le 0) -or (($_fileGroup.Length + $_.Length + 1) -le 7168))
                {
                    $_fileGroup.Length += $_.Length + 1;
                    $_fileGroup.List.AddLast($_) | Out-Null;
                }
                if ($_fileGroup.Length -gt 7168)
                {
                    Write-Output $_fileGroup;
                    ++$_groupIdx;
                    $_fileGroup = New-Object PSObject |
                        Add-Member 'Index'  $_groupIdx                                                       -PassThru |
                        Add-Member 'Length' 0                                                                -PassThru |
                        Add-Member 'List'   (New-Object 'System.Collections.Generic.LinkedList`1[[string]]') -PassThru;
                }
            } -End {
                if ($_fileGroup.Length -gt 0) { return $_fileGroup; }
            });

        # Abort on confirmation dialog (if requested).
        $_relFilesMsg = $_relFiles[0..4] -join ",`n";
        if ($_relFiles.Count -gt 5) { $_relFilesMsg += ("`n...`n{0}" -f $_relFiles[-1]); }
        if (!$Silent.IsPresent -and !$PSCmdlet.ShouldProcess($_relFilesMsg, 'Modify file attributes in VCS (stage files)'))
        {
            $VcsToken.Aborted = $true;
            return $VcsToken;
        }

        switch -Exact ($_props.Type)
        {
            'Git'      {
                    $_currentDir = (Get-Location).Path;
                    $_cmdAll = @($_toolParams);

                    # Updating mode in index (staging files if necessary).
                    Write-Verbose ('VCS: {0}: Updating index (mode: "{1}") for all specified files ({2} items) in current directory: "{3}".' -f $_props.Type, $Mode, $_relFiles.Count, $_currentDir);
                    # Command (files in batches of around 7kB): git <opts> update-index --add --chmod=<mode> -- <files ...>
                    $_relFileGroups | % {
                        & $_props.Tool $_cmdAll 'update-index' '--add' ('--chmod={0}' -f $Mode) '--' @($_.List) 2>&1 | % { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Error -ErrorRecord $_; } else { Write-Verbose $_; } };
                        if (!$? -or ($LASTEXITCODE -ne 0)) { $_success = $false; }
                    };
                }
            'Perforce' {
                    $_currentDir = (Get-Location).Path;
                    $_cmdAll = @($_toolParams);

                    switch -Regex ($_props.Change)
                    {
                        '^new$' {
                            $_changeNumber = @(echo "Change: new`nDescription:`n" | & $_props.Tool $_cmdAll 'change' '-i' 2>&1 |
                                ? { ($_ -is [string]) -and ($_ -match '^\s*Change\s+[0-9]+') } |
                                % { $_ -replace '^\s*Change\s+([0-9]+).*$', '$1' })
                            if (!$? -or ($LASTEXITCODE -ne 0)) { $_changeNumber = @(); }

                            if ($_changeNumber.Count -gt 0)
                            {
                                $_props.Change = $_changeNumber[0];
                                Write-Verbose ('VCS: {0}: Creating new changelist successful: {1}' -f $_props.Type, $_props.Change);
                            }
                            else
                            {
                                Write-Error ('VCS: {0}: Creating new changelist failed.' -f $_props.Type);
                                $_success = $false;
                            }
                            break;
                        }
                        '^[0-9]+$' {
                            & $_props.Tool $_cmdAll 'change' '-o' $_props.Change 2>&1 | Out-Null;
                            if (!$? -or ($LASTEXITCODE -ne 0)) { $_success = $false; }

                            if (!$_success)
                            {
                                Write-Error ('VCS: {0}: Selected changelist does not exist: {1}' -f $_props.Type, $_props.Change);
                            }
                        }
                        '^default$' {}
                    }


                    if ($_success)
                    {
                        # Adding new files.
                        Write-Verbose ('VCS: {0}: Adding new files (CL: {1}; mode: "{2}"; {3} items) in current directory (if possible): "{4}".' -f $_props.Type, $_props.Change, $Mode, $_relFiles.Count, $_currentDir);
                        # Command (files in batches of around 7kB): p4 <opts> add -c <changelist> <files ...>
                        $_relFileGroups | % {
                            & $_props.Tool $_cmdAll 'add' '-c' $_props.Change @($_.List) 2>&1 | % { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Error -ErrorRecord $_; } else { Write-Verbose $_; } };
                            if (!$? -or ($LASTEXITCODE -ne 0)) { $_success = $false; }
                        };
                    
                        # Opening files for edit.
                        Write-Verbose ('VCS: {0}: Opening files for edit (CL: {1}; mode: "{2}"; {3} items) in current directory (if possible): "{4}".' -f $_props.Type, $_props.Change, $Mode, $_relFiles.Count, $_currentDir);
                        # Command (files in batches of around 7kB): p4 <opts> open -c <changelist> <files ...>
                        $_relFileGroups | % {
                            & $_props.Tool $_cmdAll 'open' '-c' $_props.Change @($_.List) 2>&1 | % { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Error -ErrorRecord $_; } else { Write-Verbose $_; } };
                            if (!$? -or ($LASTEXITCODE -ne 0)) { $_success = $false; }
                        };
                    }

                    if ($_success)
                    {
                        if ($Mode -match '^[+]') { $_cmdSpecific = @('-t', $Mode); }
                        else
                        {
                            Write-Warning ('VCS: {0}: Script does not support attribute removal for selected VCS. Mode will be ignored: "{1}".' -f $_props.Type, $Mode);
                            $_cmdSpecific = @();
                        }

                        # Re-opening files in selected changelist.
                        Write-Verbose ('VCS: {0}: Re-opening files in proper changelist (CL: {1}; mode: "{2}"; {3} items) in current directory (if possible): "{4}".' -f $_props.Type, $_props.Change, $Mode, $_relFiles.Count, $_currentDir);
                        # Command (files in batches of around 7kB): p4 <opts> reopen -c <changelist> [-t <mode>] <files ...>
                        $_relFileGroups | % {
                            & $_props.Tool $_cmdAll 'reopen' '-c' $_props.Change $_cmdSpecific @($_.List) 2>&1 | % { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Error -ErrorRecord $_; } else { Write-Verbose $_; } };
                            if (!$? -or ($LASTEXITCODE -ne 0)) { $_success = $false; }
                        };
                    }
                }
        }

        $_state.Add('ChmodStage') | Out-Null;
        $VcsToken.Success    = $_success;
        $VcsToken.Aborted    = $_aborted;
        $VcsToken.State      = $_state;
        $VcsToken.ToolParams = $_toolParams;
        $VcsToken.Props      = $_props;

        return $VcsToken;
    }

    # done
    function _process-ChmodStaged
    {
        <#
            .SYNOPSIS
                Process text-like items by formatting them line by line.

            .DESCRIPTION
                Process selected source item (usually text file) and formats it line by line and reformats
                them sligtly by:
                 -- removing trailing white-spaces from each line (if needed)
                 -- unifying/normalizing indent
                 -- changing and unifying EOL characters

                 UTF8 content is assumed. Written items are written using UTF8 encoding (BOM is not written).

            .PARAMETER DstPath
                Path to destination repository (usually Open Source repository). It should point to
                repository root (container/directory path).

                Wildcards are NOT allowed. Path must be valid, exist and point to directory (container) item.

            .PARAMETER DstRelFiles
                List of relative paths to files which mode (file attributes in VCS) will be modified.
                Files will be staged to be able to change their mode.
                
                For some VCS (Git) only files which are effectively modified (content or attributes) will
                be staged. For others (Perforce), all selected files will be added to specific change list. 

                List can be empty. Paths are assumed to be relative to DstPath.

            .PARAMETER Mode
                File mode (attribute) to set.

                Allowed values: '-x', '+x' (clear/set executive bit; for Perforce script ignores '-x').

            .PARAMETER VcsType
                Type of version control system used.

                Allowed values: 'Git', 'Perforce'.

            .PARAMETER VcsServer
                URL/URI to server/port/entrypoint of version control system.

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsCredential
                Credentials for accessing server/port/entrypoint of version control system.

                This parameter is only used by specific types of VCS and ignored by the rest.

                When this parameter is specified and non-null, the -PromptVcsCredential is ignored.

            .PARAMETER PromptVcsCredential
                Prompts for credentials for accessing server/port/entrypoint of version
                control system. Usually it is in the form of dialog with default user name for VCS
                specified.

                This parameter is only used by specific types of VCS and ignored by the rest.

                When VcsCredential parameter is specified and non-null, current parameter is ignored.

            .PARAMETER VcsWorkspace
                Expected workspace in VCS. If it is specified, the destination location is
                checked to see whether it is contained in specified workspace. If location
                is outside workspace or workspace is absent in VCS, the confirmation dialog
                will be shown (whether to continue).

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsBranch
                Expected branch/stream in VCS. If it is specified, the destination location is
                checked to see whether it is switch to selected branch/stream. If location
                is not set to use specified branch/stream or branch/stream is absent in VCS,
                the confirmation dialog will be shown (whether to continue).

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER VcsChangelist
                Changelist where to stage files in VCS.

                If 'New' is selected, the new changelist will be created for staged files.
                If 'Default' is selected, any files will be staged in default changelist.
                If number is specified, the files will be staged on selected changelist.

                Allowed values: 'New', 'Default', <any non-negative integral number>

                This parameter is only used by specific types of VCS and ignored by the rest.

            .PARAMETER Force
                Force operation (even if the expected branch/stream does not match mapped branch/stream for
                current directory).
                
            .PARAMETER Silent
                Suppresses any confirmation. Implies -Force.
        #>

        [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
        [OutputType([void])]
        param(
            [Parameter(Position = 0, Mandatory = $true)]
            [Alias('DestPath', 'DestinationPath')]
            [string] $DstPath,
            [Parameter(Position = 1, Mandatory = $true,
                       ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
            [Alias('RelFiles', 'Files')]
            [AllowEmptyCollection()]
            [string[]] $DstRelFiles,
            [Parameter(Position = 2, Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidatePattern('^[+-]x$', Options = 'None')]
            [Alias('Chmod')]
            [string] $Mode = '+x',

            [Parameter(Position = 3, Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [ValidateSet('Git', 'Perforce')]
            [Alias('Type', 'Vcs')]
            [string] $VcsType = 'Git',

            [Parameter(Mandatory = $false)]
            [Alias('Server')]
            [string] $VcsServer = '',
            [Parameter(Mandatory = $false)]
            [Alias('Credential')]
            [System.Management.Automation.PSCredential] $VcsCredential = $null,
            [Parameter(Mandatory = $false)]
            [Alias('PromptCredential')]
            [switch] $PromptVcsCredential,
            [Parameter(Mandatory = $false)]
            [Alias('Workspace')]
            [string] $VcsWorkspace = '',
            [Parameter(Mandatory = $false)]
            [Alias('Branch', 'VcsStream', 'Stream')]
            [string] $VcsBranch = '',
            [Parameter(Mandatory = $false)]
            [ValidatePattern('^(?:[0-9]+|Default|New)$')]
            [Alias('Changelist', 'VcsChange', 'Change')]
            [string] $VcsChangelist = 'New',

            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [switch] $Force,
            [Parameter(Mandatory = $false,
                       ValueFromPipelineByPropertyName = $true)]
            [Alias('Quiet')]
            [switch] $Silent
        );
        begin
        {
            $_tokens = @{}; # vcs tokens used by the pipeline.

            [bool] $_locationPushed = $false;
            Push-Location -LiteralPath $DstPath -ErrorAction Stop;
            $_locationPushed = $?;
        }
        process
        {
            # Get existing token if possible.
            $_token = $_tokens[$VcsType];

            # If token is invalid or does not exist - create/re-create it.
            if (($_token -eq $null) -or !$_token.Success)
            {
                $_token = _login-Vcs $VcsType -VcsServer $VcsServer -VcsCredential $VcsCredential -PromptVcsCredential:$PromptVcsCredential.IsPresent `
                    -VcsWorkspace $VcsWorkspace -VcsBranch $VcsBranch -VcsChangelist $VcsChangelist;
                $_token = _set-VcsWorkspace $_token -Force:$Force.IsPresent -Silent:$Silent.IsPresent;
                $_token = _set-VcsBranch $_token -Force:$Force.IsPresent -Silent:$Silent.IsPresent;
            }

            # Change mode of VCS for selected files.
            $_token = _chmod-VcsFiles $_token $DstRelFiles -Mode $Mode -Force:$Force.IsPresent -Silent:$Silent.IsPresent;

            $_tokens[$VcsType] = $_token;
        }
        end
        {
            $_tokens = @{};
            if ($_locationPushed) { Pop-Location; $_locationPushed = $false; }
        }
    }
}
process
{
    # Initial checks.
    if ([string]::IsNullOrEmpty($SrcPath))
    {
        $SrcPath = _get-DefaultSrcPath;
    }
    if ([string]::IsNullOrEmpty($DstPath))
    {
        $DstPath = $SrcPath;
    }
    if ([string]::IsNullOrEmpty($CfgPath))
    {
        $CfgPath = _get-ScriptConfigPath;
    }


    # Loading configuration.
    Write-Verbose ('Loading script configuration file: "{0}"' -f $CfgPath);
    [string] $_progActivity = 'Performing synchronization: "{0}" -> "{1}"' -f $SrcPath, $DstPath;
    [string] $_progStatus   = 'Loading script configuration file';
    [int]    $_progPercent  = 0;
    [int]    $_progDelta    = 5;
    [float]  $_progItemsCP  = $_progPercent;
    [float]  $_progItemsCPD = 0;
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_config = _load-ScriptConfig $CfgPath;

    [int] $_entriesCount     = 0;
    [int] $_postEntriesCount = 0;
    $_configCopy            = @($_config.copy             | ? { $_ -ne $null }); $_entriesCount     += $_configCopy.Count;
    $_configUpdateVersion   = @($_config.update_version   | ? { $_ -ne $null }); $_entriesCount     += $_configUpdateVersion.Count;
    $_configProcessSections = @($_config.process_sections | ? { $_ -ne $null }); $_entriesCount     += $_configProcessSections.Count;
    $_configFormatFile      = @($_config.format_file      | ? { $_ -ne $null }); $_entriesCount     += $_configFormatFile.Count;
    $_configChmodStaged     = @($_config.chmod_staged     | ? { $_ -ne $null }); $_postEntriesCount += $_configChmodStaged.Count;


    # Preparing items for "copy" action(s).
    Write-Verbose ('Performing "copy" actions (gathering items to copy/remove): "{0}"' -f $CfgPath);
    $_progStatus   = 'Performing "copy" actions (gathering items to copy/remove)';
    $_progPercent  += $_progDelta;
    $_progDelta    = 40;
    $_progItemsCP  = $_progPercent;
    $_progItemsCPD = $_progDelta / [Math]::Max($_entriesCount, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_entryCategoryIndex = 0;
    $_allSyncItems = @($_configCopy | % {
            ++$_entryCategoryIndex;

            $_qual      = @($_.qual       | ? { ![string]::IsNullOrEmpty($_) });
            $_srcPrefix = @($_.src_prefix | ? { ![string]::IsNullOrEmpty($_) });
            $_dstPrefix = @($_.dst_prefix | ? { ![string]::IsNullOrEmpty($_) });

            if ($_qual.Count -le 0) { $_qual = '.'; }
            else
            {
                if ($_qual.Count -gt 1) { Write-Warning ('Multiple qualifiers specified in single "copy" entry (#{1}). Only first will be used: "{0}"' -f $_qual[0], $_entryCategoryIndex); }
                $_qual = $_qual[0];
            }

            if ($_srcPrefix.Count -le 0) { $_srcPrefix = '.'; }
            else
            {
                if ($_srcPrefix.Count -gt 1) { Write-Warning ('Multiple source prefixes specified in single "copy" entry (#{1}). Only first will be used: "{0}"' -f $_srcPrefix[0], $_entryCategoryIndex); }
                $_srcPrefix = $_srcPrefix[0];
            }

            if ($_dstPrefix.Count -le 0) { $_dstPrefix = '.'; }
            else
            {
                if ($_dstPrefix.Count -gt 1) { Write-Warning ('Multiple destination prefixes specified in single "copy" entry (#{1}). Only first will be used: "{0}"' -f $_dstPrefix[0], $_entryCategoryIndex); }
                $_dstPrefix = $_dstPrefix[0];
            }

            $_include   = @($_.include    | ? { ![string]::IsNullOrEmpty($_) });
            $_includeRe = @($_.include_re | ? { ![string]::IsNullOrEmpty($_) });
            $_exclude   = @($_.exclude    | ? { ![string]::IsNullOrEmpty($_) });
            $_excludeRe = @($_.exclude_re | ? { ![string]::IsNullOrEmpty($_) });
            $_ignore    = @($_.ignore     | ? { ![string]::IsNullOrEmpty($_) });
            $_ignoreRe  = @($_.ignore_re  | ? { ![string]::IsNullOrEmpty($_) });

            $_copyAction = New-Object PSObject |
                Add-Member 'Qualifier' $_qual      -PassThru |
                Add-Member 'SrcPrefix' $_srcPrefix -PassThru |
                Add-Member 'DstPrefix' $_dstPrefix -PassThru |
                % { if ($_include.Count   -gt 0) { $_ | Add-Member 'IncludeFilter'   $_include   -PassThru } else { $_ } } |
                % { if ($_includeRe.Count -gt 0) { $_ | Add-Member 'ReIncludeFilter' $_includeRe -PassThru } else { $_ } } |
                % { if ($_exclude.Count   -gt 0) { $_ | Add-Member 'ExcludeFilter'   $_exclude   -PassThru } else { $_ } } |
                % { if ($_excludeRe.Count -gt 0) { $_ | Add-Member 'ReExcludeFilter' $_excludeRe -PassThru } else { $_ } } |
                % { if ($_ignore.Count    -gt 0) { $_ | Add-Member 'IgnoreFilter'    $_ignore    -PassThru } else { $_ } } |
                % { if ($_ignoreRe.Count  -gt 0) { $_ | Add-Member 'ReIgnoreFilter'  $_ignoreRe  -PassThru } else { $_ } };

            Write-Verbose ("Performing `"copy`" action:`n{0}" -f (($_copyAction | fl * | Out-String -Stream | ? { $_ -notmatch '^\s*$' } | % { ' -- {0}' -f $_ }) -join "`n"));
            $_progOperation = '"copy" entry ({0}/{1})' -f $_entryCategoryIndex, $_configCopy.Count;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;

            Write-Output $_copyAction;

            $_progItemsCP += $_progItemsCPD;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
        } | _list-AllCopyItems -LiteralPath $SrcPath -DstLiteralPath $DstPath);

    # $_allSyncItems | select SyncWos*, PSIsContainer | Export-Clixml -LiteralPath 'Sync-WithOpenSource.xml'; return;

    # $_allSyncItems = @(Import-Clixml (Join-Path (_get-ScriptDir) 'Sync-WithOpenSource.xml') | % { $_ }); # WA for bug in enumerating deserialized collections.
    # $_allSyncItems | % { $_.SyncWosTasks = [System.Collections.Generic.HashSet`1[[string]]] @($_.SyncWosTasks); };

    # Expected modes:
    # SyncWosMode = None, Copy, Remove, RemoveDst, Processed

    # Preparing items for "update_version" action(s).
    Write-Verbose ('Performing "update_version" actions (updating version info in items): "{0}"' -f $CfgPath);
    $_progStatus   = 'Performing "update_version" actions (updating version info in items)';
    $_progItemsCPD = $_progDelta / [Math]::Max($_entriesCount, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP;
    $_entryCategoryIndex = 0;
    $_configUpdateVersion | % {
            ++$_entryCategoryIndex;

            $_file     = @($_.file     | ? { ![string]::IsNullOrEmpty($_) } | _convert-AntLikeWildcard | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_position = @($_.position | ? { $_ -in @('Major', 'Minor', 'Build') });
            $_step     = @($_.step     | ? { $_ -is [int] });

            if ($_position.Count -le 0)
            { 
                $_position = 'Build';
                Write-Warning ('Specified update position is invalid in "update_version" entry (#{1}). It will be set to: "{0}"' -f $_position, $_entryCategoryIndex);
            }
            else
            {
                if ($_position.Count -gt 1) { Write-Warning ('Multiple update positions specified in single "update_version" entry (#{1}). Only first will be used: "{0}"' -f $_position[0], $_entryCategoryIndex); }
                $_position = $_position[0];
            }

            if ($_step.Count -le 0)
            { 
                $_step = 1;
                Write-Warning ('Specified update step is invalid in "update_version" entry (#{1}). It will be set to: {0}' -f $_step, $_entryCategoryIndex);
            }
            else
            {
                if ($_step.Count -gt 1) { Write-Warning ('Multiple update steps specified in single "update_version" entry (#{1}). Only first will be used: {0}' -f $_step[0], $_entryCategoryIndex); }
                $_step = $_step[0];
            }

            $_allSrcDstPairs = @($_allSyncItems |
                    ? { !$_.PSIsContainer } |
                    ? { ($_.SyncWosMode -notlike 'Remove*') -and (($_.SyncWosMode -ne 'Processed') -or !$_.SyncWosTasks.Contains('UpdateVersion')) } |
                    ? { $_item = $_; ($_file.Count -gt 0) -and (@($_file | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                    % {
                        # For processed files we need to work on file on destination.
                        if ($_.SyncWosMode -eq 'Processed') { $_srcPath = Join-Path $DstPath $_.SyncWosDstRootRelPath; } else { $_srcPath = Join-Path $SrcPath $_.SyncWosRootRelPath; }
                        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;

                        # Prevent to process twice by the same task.
                        $_.SyncWosMode = 'Processed';
                        $_.SyncWosTasks.Add('UpdateVersion') | Out-Null;

                        Write-Output (New-Object PSObject |
                            Add-Member 'SrcPath' $_srcPath -PassThru |
                            Add-Member 'DstPath' $_dstPath -PassThru);

                        if ($UpdateSrcVersion.IsPresent)
                        {
                            $_srcPath = Join-Path $SrcPath $_.SyncWosRootRelPath;
                            Write-Output (New-Object PSObject |
                                Add-Member 'SrcPath' $_srcPath -PassThru |
                                Add-Member 'DstPath' $_srcPath -PassThru);
                        }
                    } | Sort-Object -Property SrcPath, DstPath -Unique | Group-Object -Property SrcPath | % { $_item = $_.Group[0]; $_item.DstPath = @($_.Group.DstPath); return $_item; });

            if ($_allSrcDstPairs.Count -le 0)
            {
                Write-Warning ('Could not find any version file to update in "update_version" entry (#{0}).' -f $_entryCategoryIndex);
                $_progItemsCP += $_progItemsCPD;
                return;
            }

            $_progItemsCPD /= $_allSrcDstPairs.Count;
            $_entryItemIdx = 0;
            $_allSrcDstPairs | % {
                    ++$_entryItemIdx;

                    $_updateVersionAction = $_ |
                        Add-Member 'Position' $_position -PassThru |
                        Add-Member 'Step'     $_step     -PassThru;

                    Write-Verbose ("Performing `"update_version`" action:`n{0}" -f (($_updateVersionAction | fl * | Out-String -Stream | ? { $_ -notmatch '^\s*$' } | % { ' -- {0}' -f $_ }) -join "`n"));
                    $_progOperation = '"update_version" entry ({0}/{1}), item ({2}/{3})' -f $_entryCategoryIndex, $_configUpdateVersion.Count, $_entryItemIdx, $_allSrcDstPairs.Count;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;

                    Write-Output $_updateVersionAction;

                    $_progItemsCP += $_progItemsCPD;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
            };
        } | _process-VersionItem -Force:$Force.IsPresent -Silent:$Silent.IsPresent;


    # Preparing items for "process_sections" action(s).
    Write-Verbose ('Performing "process_sections" actions (processing sections in section-aware items): "{0}"' -f $CfgPath);
    $_progStatus   = 'Performing "process_sections" actions (processing sections in section-aware items)';
    $_progItemsCPD = $_progDelta / [Math]::Max($_entriesCount, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP;
    $_entryCategoryIndex = 0;
    $_configProcessSections | % {
            ++$_entryCategoryIndex;

            $_file            = @($_.file             | ? { ![string]::IsNullOrEmpty($_) } | _convert-AntLikeWildcard | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_excludeSections = @($_.exclude_sections | ? { ![string]::IsNullOrEmpty($_) } | % { $_.Trim() } | ? { ![string]::IsNullOrEmpty($_) });
            $_type            = @($_.type             | ? { $_ -in @('CMake', 'Cxx') });
            $_key             = @($_.key              | ? { ![string]::IsNullOrEmpty($_) } | % { $_.Trim() } | ? { ![string]::IsNullOrEmpty($_) });
            $_tabSize         = @($_.tab_size         | ? { ($_ -is [int]) -and ($_ -gt 0) });
            $_useTabs         = @($_.use_tabs         | ? { $_ -is [bool] });
            $_eol             = @($_.eol              | ? { $_ -in @('Unix', 'Windows', 'Mac') });

            if ($_type.Count -le 0)
            { 
                $_type = 'CMake';
                Write-Warning ('Specified type of section-aware files is invalid in "process_sections" entry (#{1}). It will be set to: "{0}"' -f $_type, $_entryCategoryIndex);
            }
            else
            {
                if ($_type.Count -gt 1) { Write-Warning ('Multiple types specified in single "process_sections" entry (#{1}). Only first will be used: "{0}"' -f $_type[0], $_entryCategoryIndex); }
                $_type = $_type[0];
            }

            if ($_key.Count -le 0)
            { 
                $_key = 'PPC';
                Write-Warning ('Specified section key is invalid in "process_sections" entry (#{1}). It will be set to: "{0}"' -f $_key, $_entryCategoryIndex);
            }
            else
            {
                if ($_key.Count -gt 1) { Write-Warning ('Multiple section keys specified in single "process_sections" entry (#{1}). Only first will be used: "{0}"' -f $_key[0], $_entryCategoryIndex); }
                $_key = $_key[0];
            }

            if ($_tabSize.Count -le 0)
            { 
                $_tabSize = 2;
                Write-Warning ('Specified tabulation size (in spaces) is invalid in "process_sections" entry (#{1}). It will be set to: {0}' -f $_tabSize, $_entryCategoryIndex);
            }
            else
            {
                if ($_tabSize.Count -gt 1) { Write-Warning ('Multiple tab sizes specified in single "process_sections" entry (#{1}). Only first will be used: {0}' -f $_tabSize[0], $_entryCategoryIndex); }
                $_tabSize = $_tabSize[0];
            }

            if ($_useTabs.Count -le 0)
            { 
                $_useTabs = $false;
                Write-Warning ('Specified tabulation usage is invalid in "process_sections" entry (#{1}). It will be set to: {0}' -f $_useTabs, $_entryCategoryIndex);
            }
            else
            {
                if ($_useTabs.Count -gt 1) { Write-Warning ('Multiple tab use options specified in single "process_sections" entry (#{1}). Only first will be used: {0}' -f $_useTabs[0], $_entryCategoryIndex); }
                $_useTabs = $_useTabs[0];
            }

            if ($_eol.Count -le 0)
            { 
                $_eol = 'Unix';
                Write-Warning ('Specified end-of-line convention is invalid in "process_sections" entry (#{1}). It will be set to: "{0}"' -f $_eol, $_entryCategoryIndex);
            }
            else
            {
                if ($_eol.Count -gt 1) { Write-Warning ('Multiple EOL conventions in single "process_sections" entry (#{1}). Only first will be used: "{0}"' -f $_eol[0], $_entryCategoryIndex); }
                $_eol = $_eol[0];
            }

            $_allSrcDstPairs = @($_allSyncItems |
                    ? { !$_.PSIsContainer } |
                    ? { ($_.SyncWosMode -notlike 'Remove*') -and (($_.SyncWosMode -ne 'Processed') -or !$_.SyncWosTasks.Contains('ProcessSections')) } |
                    ? { $_item = $_; ($_file.Count -gt 0) -and (@($_file | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                    % {
                        # For processed files we need to work on file on destination.
                        if ($_.SyncWosMode -eq 'Processed') { $_srcPath = Join-Path $DstPath $_.SyncWosDstRootRelPath; } else { $_srcPath = Join-Path $SrcPath $_.SyncWosRootRelPath; }
                        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;

                        # Prevent to process twice by the same task.
                        $_.SyncWosMode = 'Processed';
                        $_.SyncWosTasks.Add('ProcessSections') | Out-Null;

                        Write-Output (New-Object PSObject |
                            Add-Member 'SrcPath' $_srcPath -PassThru |
                            Add-Member 'DstPath' $_dstPath -PassThru);
                    } | Sort-Object -Property SrcPath, DstPath -Unique | Group-Object -Property SrcPath | % { $_item = $_.Group[0]; $_item.DstPath = @($_.Group.DstPath); return $_item; });

            if ($_allSrcDstPairs.Count -le 0)
            {
                Write-Warning ('Could not find any section-aware file to process in "process_sections" entry (#{0}).' -f $_entryCategoryIndex);
                $_progItemsCP += $_progItemsCPD;
                return;
            }

            $_progItemsCPD /= $_allSrcDstPairs.Count;
            $_entryItemIdx = 0;
            $_allSrcDstPairs | % {
                    ++$_entryItemIdx;

                    $_processSectionsAction = $_ |
                        Add-Member 'ExcludeSections' $_excludeSections -PassThru |
                        Add-Member 'CommentType'     $_type            -PassThru |
                        Add-Member 'ProcessKey'      $_key             -PassThru |
                        Add-Member 'TabSize'         $_tabSize         -PassThru |
                        Add-Member 'UseTab'          $_useTabs         -PassThru |
                        Add-Member 'EolConvention'   $_eol             -PassThru;

                    Write-Verbose ("Performing `"process_sections`" action:`n{0}" -f (($_processSectionsAction | fl * | Out-String -Stream | ? { $_ -notmatch '^\s*$' } | % { ' -- {0}' -f $_ }) -join "`n"));
                    $_progOperation = '"process_sections" entry ({0}/{1}), item ({2}/{3})' -f $_entryCategoryIndex, $_configProcessSections.Count, $_entryItemIdx, $_allSrcDstPairs.Count;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;

                    Write-Output $_processSectionsAction;

                    $_progItemsCP += $_progItemsCPD;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
            };
        } | _process-SectionsItem -Force:$Force.IsPresent -Silent:$Silent.IsPresent;


    # Preparing items for "format_file" action(s).
    Write-Verbose ('Performing "format_file" actions (formatting textual content of items): "{0}"' -f $CfgPath);
    $_progStatus   = 'Performing "format_file" actions (formatting textual content of items)';
    $_progItemsCPD = $_progDelta / [Math]::Max($_entriesCount, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP;
    $_entryCategoryIndex = 0;
    $_configFormatFile | % {
            ++$_entryCategoryIndex;

            $_file            = @($_.file             | ? { ![string]::IsNullOrEmpty($_) } | _convert-AntLikeWildcard | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_type            = @($_.type             | ? { $_ -in @('CMake', 'Cxx') });
            $_trim            = @($_.trim             | ? { $_ -is [bool] });
            $_normalizeIndent = @($_.normalize_indent | ? { $_ -in @('AlignedSpaces', 'AlignedTabs', 'Spaces', 'Tabs', 'Preserve') });
            $_tabSize         = @($_.tab_size         | ? { ($_ -is [int]) -and ($_ -gt 0) });
            $_eol             = @($_.eol              | ? { $_ -in @('Unix', 'Windows', 'Mac') });

            if ($_type.Count -le 0)
            { 
                $_type = 'Cxx';
                Write-Warning ('Specified type of formatted files is invalid in "format_file" entry (#{1}). It will be set to: "{0}"' -f $_type, $_entryCategoryIndex);
            }
            else
            {
                if ($_type.Count -gt 1) { Write-Warning ('Multiple types specified in single "format_file" entry (#{1}). Only first will be used: "{0}"' -f $_type[0], $_entryCategoryIndex); }
                $_type = $_type[0];
            }

            if ($_trim.Count -le 0)
            { 
                $_trim = $true;
                Write-Warning ('Specified line-end white-space trim indicator is invalid in "format_file" entry (#{1}). It will be set to: {0}' -f $_trim, $_entryCategoryIndex);
            }
            else
            {
                if ($_trim.Count -gt 1) { Write-Warning ('Multiple line-end whitespace trim options specified in single "format_file" entry (#{1}). Only first will be used: {0}' -f $_trim[0], $_entryCategoryIndex); }
                $_trim = $_trim[0];
            }

            if ($_normalizeIndent.Count -le 0)
            { 
                $_normalizeIndent = 'Spaces';
                Write-Warning ('Specified mode of indent normalization is invalid in "format_file" entry (#{1}). It will be set to: "{0}"' -f $_normalizeIndent, $_entryCategoryIndex);
            }
            else
            {
                if ($_normalizeIndent.Count -gt 1) { Write-Warning ('Multiple modes of indent normalization in single "format_file" entry (#{1}). Only first will be used: "{0}"' -f $_normalizeIndent[0], $_entryCategoryIndex); }
                $_normalizeIndent = $_normalizeIndent[0];
            }

            if ($_tabSize.Count -le 0)
            { 
                $_tabSize = 4;
                Write-Warning ('Specified tabulation size (in spaces) is invalid in "format_file" entry (#{1}). It will be set to: {0}' -f $_tabSize, $_entryCategoryIndex);
            }
            else
            {
                if ($_tabSize.Count -gt 1) { Write-Warning ('Multiple tab sizes specified in single "format_file" entry (#{1}). Only first will be used: {0}' -f $_tabSize[0], $_entryCategoryIndex); }
                $_tabSize = $_tabSize[0];
            }

            if ($_eol.Count -le 0)
            { 
                $_eol = 'Unix';
                Write-Warning ('Specified end-of-line convention is invalid in "format_file" entry (#{1}). It will be set to: "{0}"' -f $_eol, $_entryCategoryIndex);
            }
            else
            {
                if ($_eol.Count -gt 1) { Write-Warning ('Multiple EOL conventions in single "format_file" entry (#{1}). Only first will be used: "{0}"' -f $_eol[0], $_entryCategoryIndex); }
                $_eol = $_eol[0];
            }

            $_allSrcDstPairs = @($_allSyncItems |
                    ? { !$_.PSIsContainer } |
                    ? { ($_.SyncWosMode -notlike 'Remove*') -and (($_.SyncWosMode -ne 'Processed') -or !$_.SyncWosTasks.Contains('FormatFile')) } |
                    ? { $_item = $_; ($_file.Count -gt 0) -and (@($_file | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                    % {
                        # For processed files we need to work on file on destination.
                        if ($_.SyncWosMode -eq 'Processed') { $_srcPath = Join-Path $DstPath $_.SyncWosDstRootRelPath; } else { $_srcPath = Join-Path $SrcPath $_.SyncWosRootRelPath; }
                        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;

                        # Prevent to process twice by the same task.
                        $_.SyncWosMode = 'Processed';
                        $_.SyncWosTasks.Add('FormatFile') | Out-Null;

                        Write-Output (New-Object PSObject |
                            Add-Member 'SrcPath' $_srcPath -PassThru |
                            Add-Member 'DstPath' $_dstPath -PassThru);
                    } | Sort-Object -Property SrcPath, DstPath -Unique | Group-Object -Property SrcPath | % { $_item = $_.Group[0]; $_item.DstPath = @($_.Group.DstPath); return $_item; });

            if ($_allSrcDstPairs.Count -le 0)
            {
                Write-Warning ('Could not find any files to format in "format_file" entry (#{0}).' -f $_entryCategoryIndex);
                $_progItemsCP += $_progItemsCPD;
                return;
            }

            $_progItemsCPD /= $_allSrcDstPairs.Count;
            $_entryItemIdx = 0;
            $_allSrcDstPairs | % {
                    ++$_entryItemIdx;

                    $_formatFileAction = $_ |
                        Add-Member 'ItemType'        $_type            -PassThru |
                        Add-Member 'TrimWhitespace'  $_trim            -PassThru |
                        Add-Member 'NormalizeIndent' $_normalizeIndent -PassThru |
                        Add-Member 'TabSize'         $_tabSize         -PassThru |
                        Add-Member 'EolConvention'   $_eol             -PassThru;

                    Write-Verbose ("Performing `"format_file`" action:`n{0}" -f (($_formatFileAction | fl * | Out-String -Stream | ? { $_ -notmatch '^\s*$' } | % { ' -- {0}' -f $_ }) -join "`n"));
                    $_progOperation = '"format_file" entry ({0}/{1}), item ({2}/{3})' -f $_entryCategoryIndex, $_configFormatFile.Count, $_entryItemIdx, $_allSrcDstPairs.Count;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;

                    Write-Output $_formatFileAction;

                    $_progItemsCP += $_progItemsCPD;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
            };
        } | _process-FormatItem -Force:$Force.IsPresent -Silent:$Silent.IsPresent;


    # Expected modes:
    # SyncWosMode = None, Copy, Remove, RemoveDst, Processed

    if (!$Silent.IsPresent -and !$PSCmdlet.ShouldProcess($DstPath, 'Synchronize files to selected destination'))
    {
        Write-Verbose ('Finishing (user interrupt): "{0}"' -f $CfgPath);
        $_progStatus  = 'Completed';
        $_progPercent = 100;
        Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;

        if ($PassThru.IsPresent)
        {
            Write-Output $_allSyncItems;
        }

        Write-Progress -Activity $_progActivity -Status $_progStatus -Completed;
        return;
    }

    $_force = $Silent.IsPresent -or $Force.IsPresent;

    # Executing all "sync" copy/remove operations gathered in first step.
    Write-Verbose ('Performing "copy" actions (executing copy/remove - creating directory structure): "{0}"' -f $CfgPath);
    $_progStatus  = 'Performing "copy" actions (executing copy/remove - creating directory structure)';
    $_progPercent += $_progDelta;
    $_progDelta   = 10;
    $_progItemsCP = $_progPercent;
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;

    $_allContainerSyncItems = @($_allSyncItems | ? { $_.PSIsContainer  } | ? { ($_.SyncWosMode -ne 'None') -and ($_.SyncWosMode -ne 'Processed') });
    $_allLeafSyncItems      = @($_allSyncItems | ? { !$_.PSIsContainer } | ? { ($_.SyncWosMode -ne 'None') -and ($_.SyncWosMode -ne 'Processed') });

    # Directories (creating).
    Write-Verbose ('Performing "copy" actions (executing copy/remove - creating directory structure ({1} items)): "{0}"' -f $CfgPath, $_allContainerSyncItems.Count);
    $_progItemsCPD = $_progDelta / [Math]::Max($_allContainerSyncItems.Count, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_entryItemIdx = 0;
    $_allContainerSyncItems | % {
        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;
        ++$_entryItemIdx;

        if (![string]::IsNullOrEmpty($_dstPath) -and !(Test-Path -LiteralPath $_dstPath))
        {
            mkdir $_dstPath -Force:$_force -Confirm:$false | Out-Null;
        }

        if (($_entryItemIdx % 15) -eq 1)
        {
            $_progItemsCP += $_progItemsCPD;
            $_progOperation = 'container item ({0}/{1}): "{2}"' -f $_entryItemIdx, $_allContainerSyncItems.Count, $_dstPath;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
        }
    }

    # Files (copying).
    $_allLeafCopyItems = @($_allLeafSyncItems | ? { $_.SyncWosMode -eq 'Copy' }); 

    Write-Verbose ('Performing "copy" actions (executing copy/remove - copying items ({1} items)): "{0}"' -f $CfgPath, $_allLeafCopyItems.Count);
    $_progStatus  = 'Performing "copy" actions (executing copy/remove - copying items)';
    $_progPercent += $_progDelta;
    $_progDelta   = 20;
    $_progItemsCP = $_progPercent;
    $_progItemsCPD = $_progDelta / [Math]::Max($_allLeafCopyItems.Count, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_entryItemIdx = 0;
    $_allLeafCopyItems | % {
        $_srcPath = Join-Path $SrcPath $_.SyncWosRootRelPath;
        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;
        ++$_entryItemIdx;

        if (![string]::IsNullOrEmpty($_srcPath) -and ![string]::IsNullOrEmpty($_dstPath))
        {
            if ($_force -or !(Test-Path -LiteralPath $_dstPath -PathType Leaf) -or $PSCmdlet.ShouldContinue(('Overwrite existing file: "{0}"' -f $_dstPath), 'Overwrite file', [ref] $_owDstYesToAll, [ref] $_owDstNoToAll))
            {
                Copy-Item -LiteralPath $_srcPath $_dstPath -Force:$_force -Confirm:$false;
            }
        }

        #if (($_entryItemIdx % 15) -eq 1)
        #{
            $_progItemsCP += $_progItemsCPD;
            $_progOperation = 'item ({0}/{1}): "{2}" -> "{3}"' -f $_entryItemIdx, $_allLeafCopyItems.Count, $_srcPath, $_dstPath;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
        #}
    }

    # Files (remove).
    $_allLeafRemoveItems = @($_allLeafSyncItems | ? { $_.SyncWosMode -like 'Remove*' }); 

    Write-Verbose ('Performing "copy" actions (executing copy/remove - removing items ({1} items)): "{0}"' -f $CfgPath, $_allLeafRemoveItems.Count);
    $_progStatus   = 'Performing "copy" actions (executing copy/remove - removing items)';
    $_progPercent += $_progDelta;
    $_progDelta    = 10;
    $_progItemsCP  = $_progPercent;
    $_progItemsCPD = $_progDelta / [Math]::Max($_allLeafRemoveItems.Count, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_entryItemIdx = 0;
    $_allLeafRemoveItems | % {
        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;
        ++$_entryItemIdx;

        if (![string]::IsNullOrEmpty($_dstPath))
        {
            if ($_force -or !(Test-Path -LiteralPath $_dstPath -PathType Leaf) -or $PSCmdlet.ShouldContinue(('Remove existing file: "{0}"' -f $_dstPath), 'Remove file', [ref] $_rmDstYesToAll, [ref] $_rmDstNoToAll))
            {
                Remove-Item -LiteralPath $_dstPath -Force:$_force -Confirm:$false -ErrorAction Ignore; # this should be always successful.
            }
        }

        if (($_entryItemIdx % 15) -eq 1)
        {
            $_progItemsCP += $_progItemsCPD;
            $_progOperation = 'item ({0}/{1}): "{2}"' -f $_entryItemIdx, $_allLeafRemoveItems.Count, $_dstPath;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
        }
    }

    # Directories (removing unnecessary).
    $_allContainerRemoveItems = @($_allContainerSyncItems | ? { $_.SyncWosMode -like 'Remove*' } |
            Sort-Object -Property @{Expression = { $_.SyncWosDstRootRelPath.Length }; Descending = $true}, @{Expression = 'SyncWosDstRootRelPath'; Descending = $true} 
        );

    Write-Verbose ('Performing "copy" actions (executing copy/remove - directory clean-up ({1} items)): "{0}"' -f $CfgPath, $_allContainerRemoveItems.Count);
    $_progStatus  = 'Performing "copy" actions (executing copy/remove - directory clean-up)';
    $_progPercent += $_progDelta;
    $_progDelta   = 10;
    $_progItemsCP = $_progPercent;
    $_progItemsCPD = $_progDelta / [Math]::Max($_allContainerRemoveItems.Count, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;
    $_entryItemIdx = 0;
    $_allContainerRemoveItems | % {
        $_dstPath = Join-Path $DstPath $_.SyncWosDstRootRelPath;
        ++$_entryItemIdx;

        if (![string]::IsNullOrEmpty($_dstPath) -and (Test-Path -LiteralPath $_dstPath -PathType Container) -and (@(Get-ChildItem -LiteralPath $_dstPath -Force).Length -le 0))
        {
            Remove-Item -LiteralPath $_dstPath -Force:$_force -Confirm:$false -ErrorAction Ignore;
        }

        if (($_entryItemIdx % 15) -eq 1)
        {
            $_progItemsCP += $_progItemsCPD;
            $_progOperation = 'empty container remove ({0}/{1}): "{2}"' -f $_entryItemIdx, $_allContainerRemoveItems.Count, $_dstPath;
            Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
        }
    }


    # Preparing items for "chmod_staged" action(s).
    Write-Verbose ('Performing "chmod_staged" actions (changing mode/attributes of files in VCS): "{0}"' -f $CfgPath);
    $_progStatus   = 'Performing "chmod_staged" actions (changing mode/attributes of files in VCS)';
    $_progPercent += $_progDelta;
    $_progDelta    = 5;
    $_progItemsCP  = $_progPercent;
    $_progItemsCPD = $_progDelta / [Math]::Max($_postEntriesCount, 1);
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progItemsCP;
    $_entryCategoryIndex = 0;
    $_configChmodStaged | % {
            ++$_entryCategoryIndex;

            $_file = @($_.file | ? { ![string]::IsNullOrEmpty($_) } | _convert-AntLikeWildcard | % { New-Object regex $_, 'IgnoreCase, Compiled' });
            $_mode = @($_.mode | ? { $_ -cmatch '^[+-]x$' });
            $_vcs  = @($_.vcs  | ? { $_ -in @('Git', 'Perforce') });

            if ($_mode.Count -le 0)
            { 
                $_mode = '+x';
                Write-Warning ('Specified VCS element mode/attribute is invalid in "chmod_staged" entry (#{1}). It will be set to: "{0}"' -f $_mode, $_entryCategoryIndex);
            }
            else
            {
                if ($_mode.Count -gt 1) { Write-Warning ('Multiple VCS modes/attributes specified in single "chmod_staged" entry (#{1}). Only first will be used: "{0}"' -f $_mode[0], $_entryCategoryIndex); }
                $_mode = $_mode[0];
            }


            if ($_vcs.Count -le 0)
            { 
                $_vcs = 'Git';
                Write-Warning ('Specified type of version control system is invalid in "chmod_staged" entry (#{1}). It will be set to: "{0}"' -f $_vcs, $_entryCategoryIndex);
            }
            else
            {
                if ($_vcs.Count -gt 1) { Write-Warning ('Multiple types of version control system specified in single "chmod_staged" entry (#{1}). Only first will be used: "{0}"' -f $_vcs[0], $_entryCategoryIndex); }
                $_vcs = $_vcs[0];
            }

            $_allDstRelFiles = @($_allSyncItems |
                    ? { !$_.PSIsContainer } |
                    ? { $_.SyncWosMode -notlike 'Remove*' } |
                    ? { $_item = $_; ($_file.Count -gt 0) -and (@($_file | ? { $_.IsMatch($_item.SyncWosRootRelPath) }).Count -gt 0) } |
                    % {
                        # For post-synch actions we always work on file(s) in destination.
                        $_dstRelFile = $_.SyncWosDstRootRelPath;

                        # Mark as processed by the task (but allow to process multiple times for "chmod_staged").
                        $_.SyncWosMode = 'Processed';
                        $_.SyncWosTasks.Add(('ChmodStaged:{0}:{1}' -f $_vcs, $_entryCategoryIndex)) | Out-Null;

                        Write-Output $_dstRelFile;
                    } | Sort-Object -Unique);

            if ($_allDstRelFiles.Count -le 0)
            {
                Write-Warning ('Could not find any files to change mode in VCS in "chmod_staged" entry (#{0}).' -f $_entryCategoryIndex);
                $_progItemsCP += $_progItemsCPD;
                return;
            }

            $_allDstItems = @(New-Object PSObject | Add-Member 'DstRelFiles' $_allDstRelFiles -PassThru);

            $_progItemsCPD /= $_allDstItems.Count;
            $_entryItemIdx = 0;
            $_allDstItems | % {
                    ++$_entryItemIdx;

                    $_chmodStagedAction = $_ |
                        Add-Member 'Mode'    $_mode -PassThru |
                        Add-Member 'VcsType' $_vcs  -PassThru;

                    Write-Verbose ("Performing `"chmod_staged`" action:`n{0}" -f (($_chmodStagedAction | fl * | Out-String -Stream | ? { $_ -notmatch '^\s*$' } | % { ' -- {0}' -f $_ }) -join "`n"));
                    $_progOperation = '"chmod_staged" entry ({0}/{1}), item ({2}/{3})' -f $_entryCategoryIndex, $_configChmodStaged.Count, $_entryItemIdx, $_allDstItems.Count;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;

                    Write-Output $_chmodStagedAction;

                    $_progItemsCP += $_progItemsCPD;
                    Write-Progress -Activity $_progActivity -Status $_progStatus -CurrentOperation $_progOperation -PercentComplete $_progItemsCP;
            };
        } | _process-ChmodStaged -DstPath $DstPath -VcsServer $VcsServer -VcsCredential $VcsCredential -PromptVcsCredential:$PromptVcsCredential.IsPresent `
                    -VcsWorkspace $VcsWorkspace -VcsBranch $VcsBranch -VcsChangelist $VcsChangelist -Force:$Force.IsPresent -Silent:$Silent.IsPresent;


    # Finish.
    Write-Verbose ('Finishing: "{0}"' -f $CfgPath);
    $_progStatus  = 'Completed';
    $_progPercent = 100;
    Write-Progress -Activity $_progActivity -Status $_progStatus -PercentComplete $_progPercent;

    if ($PassThru.IsPresent)
    {
        Write-Output $_allSyncItems;
    }

    Write-Progress -Activity $_progActivity -Status $_progStatus -Completed;


    #### TESTS ####

    #$_config;
    #$_allSyncItems;

    # _get-ScriptPath;
    # _get-ScriptConfigPath;
    # _get-DefaultSrcPath;

    #@(@('*') | _list-AllCopyItems $SrcPath -DstLiteralPath 'C:\_Repos\mwalkowi--os-clDNN-1' -ExcludeFilter 'utils/build') | select SyncWos*;
    #help _list-AllCopyItems -Full;

    #_convert-AntLikeWildcard '';
    #_convert-AntLikeWildcard '.';
    #_convert-AntLikeWildcard './';
    #_convert-AntLikeWildcard '**';
    #_convert-AntLikeWildcard './**';
    #_convert-AntLikeWildcard './**/data*/**\file.txt*';
    #_convert-AntLikeWildcard './**/data*/**\fil?.txt*';
    #_convert-AntLikeWildcard './**/name[d*?at\]d*.txt';
    #_convert-AntLikeWildcard '**/.git';
    #_convert-AntLikeWildcard './**/.git';
    #_convert-AntLikeWildcard '****/.git';
    #_convert-AntLikeWildcard './././******/./.git';
    #_convert-AntLikeWildcard 'data/**/.git';
    #_convert-AntLikeWildcard 'data/**/sep/**/.git';
    #_convert-AntLikeWildcard 'data/**/**/**/sep/**/**/.git';
    #help _convert-AntLikeWildcard -Full;

    #_simplify-RelPath '.';
    #_simplify-RelPath './';
    #_simplify-RelPath 'a//z2//.z1//..//../////b';
    #_simplify-RelPath './///../c';
    #_simplify-RelPath '\/\/\/d////z////..';
    #_simplify-RelPath '././\./e\\\f//..\\g';
    #_simplify-RelPath '..a';
    #_simplify-RelPath '.gitconfig';
    #_simplify-RelPath 'src/.ssh/aaas/a../b./.c/..data/f';
    #_simplify-RelPath '.a/src/.ssh/aaas/a../b./.c/..data/f';
    #_simplify-RelPath '..a/src/.ssh/aaas/a../b./.c/..data/f';
    #_simplify-RelPath 'a./src/.ssh/aaas/a../b./.c/..data/f';
    #_simplify-RelPath 'a../src/.ssh/aaas/a../b./.c/..data/f';
    #help _simplify-RelPath -Full;

    #$_testFile    = (_get-ScriptPath) + '.txt';
    #$_testOutFile = (_get-ScriptPath) + '.out.txt';
    #_process-SectionsItem $_testFile $_testOutFile -ExcludeSections 'S2', 'S3', 'S4' -Confirm;
    #help _process-SectionsItem -Full;

    #$_testVFile    = Join-Path (_get-DefaultSrcPath) 'version.json';
    #$_testVOutFile = $_testVFile + '.out.txt';
    #_process-VersionItem $_testVFile $_testVOutFile, $_testVFile -Step 1001 -Position Minor -Force -WhatIf;
    #help _process-VersionItem -Full;

    #_extract-UserName 'GER\test1';
    #_extract-UserName '\test.2';
    #_extract-UserName 'test.3';
    #_extract-UserName 'test.4@ger.corp.intel.com';
    #_extract-UserName 'test.5@';
    #help _extract-UserName -Full;
    #_get-VcsDefaultUser;
    #_get-VcsDefaultUser -VcsType Git;
    #_get-VcsDefaultUser -VcsType Perforce;
    #help _get-VcsDefaultUser -Full;
}
end {}