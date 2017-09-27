# INTEL CONFIDENTIAL
# Copyright 2017 Intel Corporation
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
    Dumps all weights and biases from selected model/topology in Inference Engine.

    Author: <marcin.walkowiak@intel.com> (no guaranties to work in each case!!!)

.DESCRIPTION
    Based of path to .xml file, the script finds corresponding data (.bin) file and parses it to ranges specified by XML model.
    Next, it writes the ranges to files named "<layer name>_weights<idx>" and "<layer name>_biases<idx>"
    in human-readable form. The files are stored in selected output directory.

.PARAMETER ModelPath
    Path to file name of .xml file of the model to dump.

.PARAMETER Filter
    Specify wildcards filter of primitive/layer names which should be dumped.

    If not specified, weights/biases of all primitve/layers are dumped.

.PARAMETER OutputDir
    Name of output directory to which weights will be written.
    
    If not specified, current directory is used.

.PARAMETER WeightsOnly
    Dump only weights.

.PARAMETER BiasesOnly
    Dump only biases.

.PARAMETER Force
    Overwrite dump files, even if they already exist.

#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, HelpMessage = 'Provide path to .xml file with IE model', ValueFromPipeline = $true)]
    [Alias('ModelName', 'Name')]
    [string] $ModelPath,
    [Parameter(ValueFromPipelineByPropertyName = $true)]
    [string[]] $Filter,
    [Parameter(ValueFromPipelineByPropertyName = $true)]
    [Alias('OutDir')]
    [string] $OutputDir,
    [Parameter(ValueFromPipelineByPropertyName = $true)]
    [switch] $WeightsOnly,
    [Parameter(ValueFromPipelineByPropertyName = $true)]
    [switch] $BiasesOnly,
    [Parameter()]
    [switch] $Force
)
begin
{
    function Convert-HalfToFloat([uint16] $Val, [switch] $FlushDenormToZero)
    {
        # FP32 parts extracted from FP16.
        [uint32] $_sign = ($Val -band [uint32] 0x8000) -shl (31 - 15);
        [uint32] $_mantissa = ($Val -band [uint32] 0x03FF) -shl (22 - 9);

        [uint32] $_exp_val_f16 = ($Val -band [uint32] 0x7C00) -shr 10;
        [uint32] $_exp = 0;
        if ($_exp_val_f16 -eq 0)
        {
            # Handling +/-0 and denormals.
            if ($_mantissa -eq 0)
            {
                $_exp = 0;
            }
            elseif ($FlushDenormToZero.IsPresent)
            {
                $_sign = 0;
                $_exp = 0;
                $_mantissa = 0;
            }
            else
            {
                # Denorms conversion to normal numbers.
                $_exp = 127 - 15;
                while (($_mantissa -band [uint32] 0x400000) -eq 0)
                {
                    $_mantissa = $_mantissa -shl 1;
                    --$_exp;
                }
                $_mantissa = ($_mantissa -shl 1) -band [uint32] 0x7FFFFF;
                $_exp = $_exp -shl 23;
            }
        }
        else
        {
            # Handling +/-infinity, NaN and normal numbers.
            if ($_exp_val_f16 -eq 0x1F)
            {
                $_exp = [uint32] 0xFF -shl 23;
            }
            else
            {
                $_exp = ($_exp_val_f16 + 127 - 15) -shl 23;
            }
        }

        [uint32] $_ret = $_sign -bor $_exp -bor $_mantissa;
        $_retBytes = [BitConverter]::GetBytes($_ret);

        return [BitConverter]::ToSingle($_retBytes, 0);
    }

    function Reinterpret-ByteToSByte([byte] $val)
    {
        return [sbyte]([sbyte]($val -band 0x7F) -bor [sbyte]-($val -band 0x80))
    }
}
process
{
    Write-Verbose ('Checking validity of the model path: "{0}".' -f $ModelPath);
    if (!(Test-Path -LiteralPath $ModelPath -PathType Leaf -IsValid))
    {
        Write-Error ('Selected path to model file ("{0}") is not valid.' -f $ModelPath);
        return;
    }

    $_modelDir      = Split-Path -Parent $ModelPath; # Only split needed for some bizzare file storage provider in PS.
    $_modelFileName = Split-Path -Leaf $ModelPath;

    $_modelDataFileName     = [System.IO.Path]::ChangeExtension($_modelFileName, '.bin');
    $_modelTopologyFileName = [System.IO.Path]::ChangeExtension($_modelFileName, '.xml');

    $_modelDataPath     = Join-Path $_modelDir $_modelDataFileName;
    $_modelTopologyPath = Join-Path $_modelDir $_modelTopologyFileName;

    Write-Verbose ('Checking existence of the model data file: "{0}".' -f $_modelDataPath);
    if (!(Test-Path -LiteralPath $_modelDataPath -PathType Leaf))
    {
        Write-Error ('Cannot find model data file at following path "{0}".' -f $_modelDataPath);
        return;
    }
    Write-Verbose ('Checking existence of the model topology file: "{0}".' -f $_modelTopologyPath);
    if (!(Test-Path -LiteralPath $_modelTopologyPath -PathType Leaf))
    {
        Write-Error ('Cannot find model topology file at following path "{0}".' -f $_modelTopologyPath);
        return;
    }

    if ([string]::IsNullOrEmpty($OutputDir))
    {
        $_dumpDir = (Get-Location).Path;
    }
    else
    {
        $_dumpDir = $OutputDir;
    }

    Write-Verbose ('Checking validity of the output directory: "{0}".' -f $_dumpDir);
    if (!(Test-Path -LiteralPath $_dumpDir -PathType Container -IsValid))
    {
        Write-Error ('Selected output directory ("{0}") is not valid.' -f $_dumpDir);
        return;
    }
    Write-Verbose ('Creating output directory if needed: "{0}".' -f $_dumpDir);
    mkdir $_dumpDir -ErrorAction SilentlyContinue | Out-Null;

    Write-Verbose ('Scanning topology file for weights and biases entries: "{0}".' -f $_modelTopologyPath);
    try
    {
        $_weightNodes = @(Select-Xml -LiteralPath $_modelTopologyPath '/net/layers/layer/weights' -ErrorAction Stop);
        $_biasNodes   = @(Select-Xml -LiteralPath $_modelTopologyPath '/net/layers/layer/biases' -ErrorAction Stop);
    }
    catch
    {
        Write-Error ('Selected model topology file ("{0}") is invalid (not XML).' -f $_modelTopologyPath);
        return;
    }

    if ($Filter.Count -eq 0)
    {
        $_filterSB = { $true; }
    }
    else
    {
        $_filterSB = { $_xpMatch = $_; @($Filter | ? { $_xpMatch.ParentNode.name -like $_ }).Count -gt 0 }
    }

    Write-Verbose ('Filtering nodes if necessary: "{0}".' -f $_modelTopologyPath);
    $_weightNodesFiltered = @($_weightNodes.Node | ? { !$BiasesOnly.IsPresent } | ? $_filterSB); 
    $_biasNodesFiltered   = @($_biasNodes.Node | ? { !$WeightsOnly.IsPresent } | ? $_filterSB);

    Write-Verbose ('Gathering primitive/layer information (weights): "{0}".' -f $_modelTopologyPath);
    $_dumpNameCounters = @{};

    $_weightMeta = @($_weightNodesFiltered | % {
        $_primName = $_.ParentNode.name;
        $_primType = $_.ParentNode.type;

        Write-Verbose (' - "{0}" ({1})' -f $_primName, $_primType);
        if ($_dumpNameCounters.ContainsKey($_primName))
        {
            ++$_dumpNameCounters[$_primName];
        }
        else
        {
            $_dumpNameCounters[$_primName] = 0;
        }

        $_input0Dims  = @($_.ParentNode.input.port)[0];
        $_output0Dims = @($_.ParentNode.output.port)[0];

        $_width  = @(Select-Xml '*/@kernel-x' -Xml $_.ParentNode);
        $_height = @(Select-Xml '*/@kernel-y' -Xml $_.ParentNode);
        if ($_width.Count -gt 0)
        {
            $_width = [long] $_width[0].Node.value;
        }
        else
        {
            $_width = [long] @($_input0Dims.dim)[-1];
        }
        if ($_height.Count -gt 0)
        {
            $_height = [long] $_height[0].Node.value;
        }
        else
        {
            $_height = [long] @($_input0Dims.dim)[-2];
        }

        return New-Object PSObject -Property @{
               DumpFileName = ('{0}__weights{1}.txt' -f $_primName, $_dumpNameCounters[$_primName]);
               DataOffset   = [long] $_.offset;
               DataSize     = [long] $_.size;
               InputFeaturesCount  = [long] @($_input0Dims.dim)[-3];
               OutputFeaturesCount = [long] @($_output0Dims.dim)[-3];
               Width               = $_width;
               Height              = $_height;
               Type                = $_primType;
               Kind                = 'weights';
               DataType            = $_.ParentNode.precision;
            }
    });

    Write-Verbose ('Gathering primitive/layer information (biases): "{0}".' -f $_modelTopologyPath);
    $_dumpNameCounters = @{};

    $_biasMeta = @($_biasNodesFiltered | % {
        $_primName = $_.ParentNode.name;
        $_primType = $_.ParentNode.type;

        Write-Verbose (' - "{0}" ({1})' -f $_primName, $_primType);
        if ($_dumpNameCounters.ContainsKey($_primName))
        {
            ++$_dumpNameCounters[$_primName];
        }
        else
        {
            $_dumpNameCounters[$_primName] = 0;
        }

        $_output0Dims = @($_.ParentNode.output.port)[0];

        return New-Object PSObject -Property @{
               DumpFileName = ('{0}__biases{1}.txt' -f $_primName, $_dumpNameCounters[$_primName]);
               DataOffset   = [long] $_.offset;
               DataSize     = [long] $_.size;
               InputFeaturesCount  = [long] 1;
               OutputFeaturesCount = [long] @($_output0Dims.dim)[-3];
               Width               = [long] 1;
               Height              = [long] 1;
               Type                = $_primType;
               Kind                = 'biases';
               DataType            = $_.ParentNode.precision;
            }
    });

    Write-Verbose ('Writing dump files for: "{0}".' -f $_modelTopologyPath);
    $_dataStream = (gi -LiteralPath $_modelDataPath).OpenRead();
    $_dataReader = New-Object System.IO.BinaryReader $_dataStream;
    $_allMeta = @($_weightMeta; $_biasMeta);
    $_allMeta | % {
        switch ($_.DataType)
        {
            'FP32'  { $_dataSize = 4; $_dataTypeName = $_.ToLower(); $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = [BitConverter]::ToSingle($Data, $Index); $Writer.Write('{0,15:E7}', $_dataPoint); }; }
            'FP16'  { $_dataSize = 2; $_dataTypeName = $_.ToLower(); $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = Convert-HalfToFloat ([BitConverter]::ToUInt16($Data, $Index)); $Writer.Write('{0,12:E4}', $_dataPoint); }; }
            'I16'   { $_dataSize = 2; $_dataTypeName = 'int16';      $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = [BitConverter]::ToInt16($Data, $Index); $Writer.Write('{0,6}', $_dataPoint); }; }
            'U16'   { $_dataSize = 2; $_dataTypeName = 'uint16';     $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = [BitConverter]::ToUInt16($Data, $Index); $Writer.Write('{0,5}', $_dataPoint); }; }
            'I8'    { $_dataSize = 1; $_dataTypeName = 'int8';       $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = Reinterpret-ByteToSByte $Data[$Index]; $Writer.Write('{0,4}', $_dataPoint); }; }
            'U8'    { $_dataSize = 1; $_dataTypeName = 'uint8';      $_dataPrint = { param([byte[]] $Data, [int] $Index, [System.IO.StringWriter] $Writer) $_dataPoint = $Data[$Index]; $Writer.Write('{0,3}', $_dataPoint); }; }
            default { Write-Error ('Unknown data format ({0}).' -f $_); return; }
        }

        $_calcSize = $_.OutputFeaturesCount * $_.InputFeaturesCount * $_.Width * $_.Height * $_dataSize;
        if ($_calcSize -ne $_.DataSize)
        {
            Write-Warning ('Calculated size of weights/biases is different than stored size ({0}B != {1}B).' -f $_calcSize, $_.DataSize);

            if ($_calcSize -gt $_.DataSize)
            {
                Write-Error 'Stored size of data is not sufficient to dump all required weights/biases.';
                return;
            }
        }

        Write-Verbose (' - "{0}"' -f $_.DumpFileName);

        $_dataReader.BaseStream.Seek($_.DataOffset, [System.IO.SeekOrigin]::Begin) | Out-Null;
        $_dumpData = $_dataReader.ReadBytes($_.DataSize);
        $_dumpDataPos = 0
        $_dumpDataElemPos = 0;

        $_dumpFile = New-Item (Join-Path $_dumpDir $_.DumpFileName) -ItemType File -Force:$Force.IsPresent;
        $_dumpWriter = New-Object 'System.IO.StringWriter' ([System.Globalization.CultureInfo]::InvariantCulture);

        $_dumpWriter.WriteLine('{');
        $_dumpWriter.WriteLine('  "topology":   "{0}",', $_modelTopologyFileName);
        $_dumpWriter.WriteLine('  "name":       "{0}",', $_.DumpFileName);
        $_dumpWriter.WriteLine('  "kind":       "{0}",', $_.Kind);
        $_dumpWriter.WriteLine('  "data_type":  "{0}",', $_dataTypeName);
        $_dumpWriter.WriteLine('  "user_type":  "{0}",', $_.Type);
        $_dumpWriter.WriteLine('  "dimensions": {{"of": {0}, "if": {1}, "h": {2}, "w": {3}}},', $_.OutputFeaturesCount, $_.InputFeaturesCount, $_.Height, $_.Width);
        $_dumpWriter.WriteLine('  "range":      {{"start": {0}, "size": {1}}},', $_.DataOffset, $_.DataSize);
        $_dumpWriter.WriteLine();
        $_dumpWriter.WriteLine('  "values": [');
        $_dumpWriter.ToString() | Out-File $_dumpFile -Encoding utf8 -Width 65536 -NoNewline; $_dumpWriter.GetStringBuilder().Clear() | Out-Null;
        for ([long] $ofi = 0; $ofi -lt $_.OutputFeaturesCount; ++$ofi)
        {
            $_dumpWriter.WriteLine('    {');
            $_dumpWriter.WriteLine('      "of_index": {0}', $ofi);
            $_dumpWriter.WriteLine('      "values": [');
            for ([long] $ifi = 0; $ifi -lt $_.InputFeaturesCount; ++$ifi)
            {
                $_dumpWriter.WriteLine('        {');
                $_dumpWriter.WriteLine('          "if_index": {0}', $ifi);
                $_dumpWriter.Write('          "values": [');
                $_sepH = $_dumpWriter.NewLine;
                for ([long] $hi = 0; $hi -lt $_.Height; ++$hi)
                {
                    $_dumpWriter.Write($_sepH);
                    $_sepH = ",$($_dumpWriter.NewLine)";

                    $_dumpWriter.Write('            [ ');
                    $_sepW = '';
                    for ([long] $wi = 0; $wi -lt $_.Width; ++$wi)
                    {
                        $_dumpWriter.Write($_sepW);
                        $_sepW = ', ';

                        & $_dataPrint $_dumpData $_dumpDataPos $_dumpWriter;
                        $_dumpDataPos += $_dataSize;
                        ++$_dumpDataElemPos;
                    }
                    $_dumpWriter.Write(' ]');
                }
                $_dumpWriter.WriteLine();
                $_dumpWriter.WriteLine('          ]');
                $_dumpWriter.Write('        }');

                if ($ifi + 1 -ne $_.InputFeaturesCount) { $_dumpWriter.WriteLine(','); } else { $_dumpWriter.WriteLine(''); }
            }
            $_dumpWriter.WriteLine('      ]');
            $_dumpWriter.Write('    }');

            if ($ofi + 1 -ne $_.OutputFeaturesCount) { $_dumpWriter.WriteLine(','); } else { $_dumpWriter.WriteLine(''); }

            # Write larger dump chunks.
            if ($_dumpDataElemPos -gt 100000)
            {
                $_dumpWriter.ToString() | Out-File $_dumpFile -Encoding utf8 -Width 65536 -NoNewline -Append; $_dumpWriter.GetStringBuilder().Clear() | Out-Null;
                $_dumpDataElemPos = 0;
            }
        }
        $_dumpWriter.WriteLine('  ]');
        $_dumpWriter.WriteLine('}');
        $_dumpWriter.ToString() | Out-File $_dumpFile -Encoding utf8 -Width 65536 -NoNewline -Append; $_dumpWriter.GetStringBuilder().Clear() | Out-Null;

        $_dumpWriter.Close();

        Write-Output $_dumpFile;
    }

    $_dataReader.Close();
    $_dataStream.Close();
}
end {}