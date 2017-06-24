/*
Copyright (c) (2013) Intel Corporation All Rights Reserved.

The source code, information and material ("Material") contained herein is owned by Intel Corporation or its suppliers or licensors, and title to such Material remains with Intel Corporation or its suppliers or licensors. The Material contains proprietary information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed or disclosed in any way without Intel's prior express written permission. No license under any patent, copyright or other intellectual property rights in the Material is granted to or conferred upon you, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


Include any supplier copyright notices as supplier requires Intel to use.

Include supplier trademarks or logos as supplier requires Intel to use, preceded by an asterisk. An asterisked footnote can be added as follows: *Third Party trademarks are the property of their respective owners.

Unless otherwise agreed by Intel in writing, you may not remove or alter this notice or any other notice embedded in Materials by Intel or Intel’s suppliers or licensors in any way.
*/

#include "power_intrumentation.h"
#include <string>

#ifdef _WIN32

#include <Windows.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

string g_lastError;
HMODULE g_hModule = NULL;

#define _CRT_SECURE_NO_WARNINGS

static bool GetLibraryLocation(string& strLocation)
{
    char* dir_path = nullptr;
    size_t ds,sz = 0;
    if (_dupenv_s(&dir_path, &ds, "IPG_Dir") != NULL)
        return false;
    if (dir_path == NULL || strlen(dir_path) == 0)
        return false;

    char* version =nullptr;
    if (_dupenv_s(&version, &sz, "IPG_Ver") != NULL)
        return false;
    if (version == NULL || strlen(version) == 0)
        return false;

    int ver_num = static_cast<int>(atof(version)) * 100;
    if (ver_num >= 270)
    {
#if _M_X64
        strLocation = string((const char*) dir_path).append("\\EnergyLib64.dll");
#else
        strLocation = string((const char*) dir_path).append("\\EnergyLib32.dll");
#endif
        return true;
    }
    else
        return false;
}

CIntelPowerGadgetLib::CIntelPowerGadgetLib(void) :
    pInitialize(NULL),
    pGetNumNodes(NULL),
    pGetMsrName(NULL),
    pGetMsrFunc(NULL),
    pGetIAFrequency(NULL),
    pGetGTFrequency(NULL),
    pGetTDP(NULL),
    pGetMaxTemperature(NULL),
    pGetTemperature(NULL),
    pReadSample(NULL),
    pGetTimeInterval(NULL),
    pGetBaseFrequency(NULL),
    pGetPowerData(NULL),
    pStartLog(NULL),
    pStopLog(NULL),
    pGetNumMsrs(NULL),
    pIsGTAvailable(NULL),
    isInitialized(false)
{
    string strLocation;
    if (GetLibraryLocation(strLocation) == false)
    {
        g_lastError = "Intel Power Gadget 2.7 or higher not found. If unsure, check if the path is in the user's path environment variable";
        return;
    }

    g_hModule = LoadLibrary((char*) strLocation.c_str());
    if (g_hModule == NULL)
    {
        g_lastError = "LoadLibrary failed on " + strLocation; 
        return;
    }
    
    pInitialize = (IPGInitialize) GetProcAddress(g_hModule, "IntelEnergyLibInitialize");
    pGetNumNodes = (IPGGetNumNodes) GetProcAddress(g_hModule, "GetNumNodes");
    pGetMsrName = (IPGGetMsrName) GetProcAddress(g_hModule, "GetMsrName");
    pGetMsrFunc = (IPGGetMsrFunc) GetProcAddress(g_hModule, "GetMsrFunc");
    pGetIAFrequency = (IPGGetIAFrequency) GetProcAddress(g_hModule, "GetIAFrequency");
    pGetGTFrequency = (IPGGetGTFrequency) GetProcAddress(g_hModule, "GetGTFrequency");
    pGetTDP = (IPGGetTDP) GetProcAddress(g_hModule, "GetTDP");
    pGetMaxTemperature = (IPGGetMaxTemperature) GetProcAddress(g_hModule, "GetMaxTemperature");
    pGetTemperature = (IPGGetTemperature) GetProcAddress(g_hModule, "GetTemperature");
    pReadSample = (IPGReadSample) GetProcAddress(g_hModule, "ReadSample");
    pGetTimeInterval = (IPGGetTimeInterval) GetProcAddress(g_hModule, "GetTimeInterval");
    pGetBaseFrequency = (IPGGetBaseFrequency) GetProcAddress(g_hModule, "GetBaseFrequency");
    pGetPowerData = (IPGGetPowerData) GetProcAddress(g_hModule, "GetPowerData");
    pStartLog = (IPGStartLog) GetProcAddress(g_hModule, "StartLog");
    pStopLog = (IPGStopLog) GetProcAddress(g_hModule, "StopLog");
    pGetNumMsrs = (IPGGetNumMsrs) GetProcAddress(g_hModule, "GetNumMsrs");
    pIsGTAvailable = (IPGIsGTAvailable) GetProcAddress(g_hModule, "IsGTAvailable");
}

CIntelPowerGadgetLib::~CIntelPowerGadgetLib(void)
{
    if (g_hModule != NULL)
        FreeLibrary(g_hModule);
}

string CIntelPowerGadgetLib::GetLastError()
{
    return g_lastError;
}

bool CIntelPowerGadgetLib::IntelEnergyLibInitialize(void)
{
    if (isInitialized)
        return true;

    if (pInitialize == NULL)
        return false;
    
    bool bSuccess = pInitialize();
    if (!bSuccess)
    {
        g_lastError = "Initializing the energy library failed";
        return false;
    }
    isInitialized = true;
    return isInitialized;
}

bool CIntelPowerGadgetLib::GetNumNodes(int * nNodes)
{
    return pGetNumNodes(nNodes);
}

bool CIntelPowerGadgetLib::GetNumMsrs(int * nMsrs)
{
    return pGetNumMsrs(nMsrs);
}

bool CIntelPowerGadgetLib::GetMsrName(int iMsr, wchar_t *pszName)
{
    return pGetMsrName(iMsr, pszName);
}

bool CIntelPowerGadgetLib::GetMsrFunc(int iMsr, int *funcID)
{
    return pGetMsrFunc(iMsr, funcID);
}

bool CIntelPowerGadgetLib::GetIAFrequency(int iNode, int *freqInMHz)
{
    return pGetIAFrequency(iNode, freqInMHz);
}

bool CIntelPowerGadgetLib::GetGTFrequency(int *freq)
{
    return pGetGTFrequency(freq);
}

bool CIntelPowerGadgetLib::GetTDP(int iNode, double *TDP)
{
    return pGetTDP(iNode, TDP);
}

bool CIntelPowerGadgetLib::GetMaxTemperature(int iNode, int *degreeC)
{
    return pGetMaxTemperature(iNode, degreeC);
}

bool CIntelPowerGadgetLib::GetTemperature(int iNode, int *degreeC)
{
    return pGetTemperature(iNode, degreeC);
}

bool CIntelPowerGadgetLib::ReadSample()
{
    bool bSuccess = pReadSample();
    if (bSuccess == false)
        g_lastError = "MSR overflowed. You can safely discard this sample";
    return bSuccess;
}

bool CIntelPowerGadgetLib::GetTimeInterval(double *offset)
{
    return pGetTimeInterval(offset);
}

bool CIntelPowerGadgetLib::GetBaseFrequency(int iNode, double *baseFrequency)
{
    return pGetBaseFrequency(iNode, baseFrequency);
}

bool CIntelPowerGadgetLib::GetPowerData(int iNode, int iMSR, double *results, int *nResult)
{
    return pGetPowerData(iNode, iMSR, results, nResult);
}

bool CIntelPowerGadgetLib::StartLog(wchar_t *szFilename)
{
    return pStartLog(szFilename);
}

bool CIntelPowerGadgetLib::StopLog()
{
    return pStopLog();
}

bool CIntelPowerGadgetLib::IsGTAvailable()
{
    return pIsGTAvailable();
}

bool CIntelPowerGadgetLib::print_power_results(double fps)
{
    string line;
    std::ifstream power_file("power_log.csv");
    while (std::getline(power_file, line))
    {
        if (line.find("Average Processor Power") == 0)
        {
            auto val = line.substr(line.find_first_of('=') + 2); // + 2 = 1 for space + 1 for '='
            std::cout << "FPS/W (total power)=\t" << fps / std::atof(val.c_str()) << std::endl;
        }
        else if (line.find("Average GT Power") == 0)
        {
            auto val = line.substr(line.find_first_of('=') + 2); // + 2 = 1 for space + 1 for '='
            std::cout << "FPS/W (GPU power)=\t" << fps / std::stof(val) << std::endl;
            return true; // we get what we wanted. nothing else to do.
        }
    }

    return false;
}

#else // LINUX
CIntelPowerGadgetLib::CIntelPowerGadgetLib(void) 
{

}

CIntelPowerGadgetLib::~CIntelPowerGadgetLib(void)
{

}

std::string CIntelPowerGadgetLib::GetLastError()
{
    return "Not implemented on current OS";
}

bool CIntelPowerGadgetLib::IntelEnergyLibInitialize(void)
{
    return false;
}

bool CIntelPowerGadgetLib::GetNumNodes(int*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetNumMsrs(int*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetMsrName(int , wchar_t*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetMsrFunc(int, int*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetIAFrequency(int, int*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetGTFrequency(int*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetTDP(int, double*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetMaxTemperature(int, int *)
{
    return false;
}

bool CIntelPowerGadgetLib::GetTemperature(int, int*)
{
    return false;
}

bool CIntelPowerGadgetLib::ReadSample()
{
    return false;
}

bool CIntelPowerGadgetLib::GetTimeInterval(double*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetBaseFrequency(int, double*)
{
    return false;
}

bool CIntelPowerGadgetLib::GetPowerData(int, int, double*, int*)
{
    return false;
}

bool CIntelPowerGadgetLib::StartLog(wchar_t*)
{
    return false;
}

bool CIntelPowerGadgetLib::StopLog()
{
    return false;
}

bool CIntelPowerGadgetLib::IsGTAvailable()
{
    return false;
}

bool CIntelPowerGadgetLib::print_power_results(double)
{
    return false;
}
#endif