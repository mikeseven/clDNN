/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <limits>
#include <chrono>
#include <ie_plugin_ptr.hpp>
#include <ie_plugin_config.hpp>
#include <ie_cnn_net_reader.h>
#include "../dpd_vcp_dl-scoring_engine/samples/format_reader/format_reader_ptr.h"
#include "../dpd_vcp_dl-scoring_engine/samples/common/samples/common.hpp"
#include <inference_engine.hpp>
#include <sys/stat.h>
#include <cmath>
#include <locale>
#include <codecvt>
#ifdef _WIN32
#include "../dpd_vcp_dl-scoring_engine/samples/common/os/windows/w_dirent.h"
#include "../../examples/common/power_instrumentation.h"
#else
#include <dirent.h>
#include "rapl.h"
#endif

using namespace InferenceEngine::details;
using namespace InferenceEngine;

#define DEFAULT_PATH_P "./lib"

static const char help_message[] = "Print a usage message.";
DEFINE_bool(h, false, help_message);

static const char image_message[] = "Required. Path to a .bmp image.";
DEFINE_string(i, "", image_message);

static const char model_message[] = "Required. Path to an .xml file with a trained model.";
DEFINE_string(m, "", model_message);

static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only";
DEFINE_string(p, "", plugin_message);

static const char plugin_path_message[] = "Path to a plugin folder.";
DEFINE_string(pp, DEFAULT_PATH_P, plugin_path_message);

static const char target_device_message[] = "Specify the target device to infer on; CPU or GPU is acceptable. " \
"Sample will look for a suitable plugin for device specified";
DEFINE_string(d, "", target_device_message);

static const char ni_message[] = "number of running iterations";
DEFINE_uint32(ni, 1, ni_message);

static const char pc_message[] = "used to turn on performance counters";
DEFINE_bool(pc, false, pc_message);

static const char pi_message[] = "used to turn on power instrumentation (supported only on Windows)";
DEFINE_string(pi, "", pi_message);

static const char dump_message[] = "if defined will dump raw outputs to the specified filename";
DEFINE_string(dump, "", dump_message);

static const char layer_message[] = "add layer to outputs";
DEFINE_string(layer, "", layer_message);

static const char im_info_message[] = "replace 1d input with im_info (for faster-rcnn based topologies) - ONLY VALID FOR BATCH 1";
DEFINE_bool(im_info, false, im_info_message);

static const char custom_message[] = "custom layer configuration file";
DEFINE_string(custom, "", custom_message);

static const char dump_kernels_message[] = "dumps the compiled kernels for custom layers";
DEFINE_bool(dump_kernels, false, dump_kernels_message);

static const char compare_message[] = "compare with reference dump";
DEFINE_string(compare, "", compare_message);

static const char csv_message[] = "dumps comparison results to a csv file (only applies when -compare is used)";
DEFINE_string(csv, "", csv_message);

static const char newapi_message[] = "use the new API (IExecutableNetwork, IInferRequest)";
DEFINE_bool(newapi, false, newapi_message);

static const char tuning_message[] = "tuning file to be used/created";
DEFINE_string(tuning, "", tuning_message);

static const char scale_message[] = "scale output for comparison";
DEFINE_double(scale, 1.0, scale_message);

extern double MAX_ENERGY_STATUS_JOULES;
extern double MAX_THROTTLED_TIME_SECONDS;

extern "C" {
#include "md5.h"  // taken from http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
}
static const size_t seed = 0xabcd1234dcba4321;
struct MD5_Result {
    unsigned char buffer[16];
};
MD5_Result BlobMD5(InferenceEngine::Blob::Ptr blob) {
    MD5_CTX ctx = *(reinterpret_cast<const MD5_CTX*>(&seed));
    MD5_Init(&ctx);
    auto buf = static_cast<char *>(blob->buffer());
    unsigned long blobSize = static_cast<unsigned long>(blob->byteSize());
    for (unsigned long pos = 0; pos < blobSize; pos += 64) {
        unsigned long size = (pos + 64) < blobSize ? 64 : blobSize - pos;
        MD5_Update(&ctx, buf + pos, size);
    }
    MD5_Result res;
    MD5_Final(reinterpret_cast<unsigned char*>(&res), &ctx);
    return res;
}
std::ostream& operator<< (std::ostream& os, const MD5_Result& md5) {
    std::ios::fmtflags flags(os.flags());//save format flags
    os << "0x";
    for (int i = 0; i < sizeof(MD5_Result); i++) {
        os << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << static_cast<unsigned>(md5.buffer[i]);
    }
    os.flags(flags);
    os.fill(' ');
    return os;
}

// copied since i can't change argsParser (its static there) and i don't want to create a dependency on the classification sample
std::vector<std::string> CollectImageNames(const std::vector<std::string>& filesAndFolders) {
    std::vector<std::string> imageNames;
    for (const auto& name : filesAndFolders) {
        struct stat sb;
        if (stat(name.c_str(), &sb) != 0) {
            std::cout << "[WARNING] File " << name << " cannot be opened!" << std::endl;
            return imageNames;
        }
        if (S_ISDIR(sb.st_mode)) {
            DIR *dp;
            dp = opendir(name.c_str());
            if (dp == nullptr) {
                std::cout << "[WARNING] Directory " << name << " cannot be opened!" << std::endl;
                return imageNames;
            }

            struct dirent *ep;
            while (nullptr != (ep = readdir(dp))) {
                std::string fileName = ep->d_name;
                if (fileName == "." || fileName == "..") continue;
                std::cout << "[INFO] Add file  " << ep->d_name << " from directory " << name << "." << std::endl;
                imageNames.push_back(name + "/" + ep->d_name);
            }
        }
        else {
            imageNames.push_back(name);
        }
    }
    return imageNames;
}

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "generic_sample [OPTION] [OPTION] ..." << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                  " << help_message << std::endl;
    std::cout << "    -i <path>           " << image_message << std::endl;
    std::cout << "    -m <path>           " << model_message << std::endl;
    std::cout << "    -ni <iter>          " << ni_message << std::endl;
    std::cout << "    -pc                 " << pc_message << std::endl;
    std::cout << "    -pi <csv filename>  " << pi_message << std::endl;
    std::cout << "    -dump <filename>    " << dump_message << std::endl;
    std::cout << "    -layer <layername>  " << layer_message << std::endl;
    std::cout << "    -im_info            " << im_info_message << std::endl;
    std::cout << "    -custom <filename>  " << custom_message << std::endl;
    std::cout << "    -dump_kernels       " << dump_kernels_message << std::endl;
    std::cout << "    -compare <filename> " << compare_message << std::endl;
    std::cout << "    -csv <filename>     " << csv_message << std::endl;
    std::cout << "    -tuning <filename>  " << tuning_message << std::endl;
}

std::vector<std::string> ParseFlagList(const std::vector<std::string>& args, const std::string& flag) {
    std::vector<std::string> result;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].compare(flag) == 0) {
            result.push_back(args[++i]);
        }
    }
    return result;
}

#ifndef _WIN32
double get_rapl_energy_info(unsigned int power_domain, unsigned int node)
{
    int          err;
    double       total_energy_consumed = 0.0;

    switch (power_domain) {
    case PKG:
        err = get_pkg_total_energy_consumed(node, &total_energy_consumed);
        break;
    case PP0:
        err = get_pp0_total_energy_consumed(node, &total_energy_consumed);
        break;
    case PP1:
        err = get_pp1_total_energy_consumed(node, &total_energy_consumed);
        break;
    case DRAM:
        err = get_dram_total_energy_consumed(node, &total_energy_consumed);
        break;
    default:
        err = MY_ERROR;
        break;
    }

    return total_energy_consumed;
}
#endif
/**
* \brief The main function of inference engine sample application
* @param argc - The number of arguments
* @param argv - Arguments
* @return 0 if all good
*/
int main(int argc, char *argv[]) {


    // Always intialize the power_gov library first


    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << "\n";
    std::string commandLine;
    for (int i = 0; i < argc; i++) {
        commandLine.append(argv[i]).append(" ");
    }
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 1;
    }

    bool noPluginAndBadDevice = FLAGS_p.empty() && FLAGS_d.compare("CPU") && FLAGS_d.compare("GPU");
    if (FLAGS_i.empty() || FLAGS_m.empty() || noPluginAndBadDevice) {
        if (noPluginAndBadDevice) std::cout << "ERROR: device is not supported" << std::endl;
        if (FLAGS_m.empty()) std::cout << "ERROR: file with model - not set" << std::endl;
        if (FLAGS_i.empty()) std::cout << "ERROR: image(s) for inference - not set" << std::endl;
        showUsage();
        return 2;
    }

    if (!FLAGS_dump.empty() && !FLAGS_compare.empty() && !FLAGS_dump.compare(FLAGS_compare))
    {
        std::cout << "ERROR: dump and reference filenames are identical" << std::endl;
        return 2;
    }

#ifndef OS_LIB_FOLDER
# define OS_LIB_FOLDER "/"
#endif

    try {
        // Load plugin
        InferenceEngine::InferenceEnginePluginPtr _plugin(
            selectPlugin({ "", FLAGS_pp, OS_LIB_FOLDER, DEFAULT_PATH_P, "" /* This means "search in default paths including LD_LIBRARY_PATH" */ }, FLAGS_p, FLAGS_d));

        const PluginVersion *pluginVersion;
        _plugin->GetVersion((const InferenceEngine::Version *&)pluginVersion);
        std::cout << pluginVersion << std::endl;

        // Performance counters
        if (FLAGS_pc) {
            _plugin->SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } }, nullptr);
        }

        // Kernels dump
        if (FLAGS_dump_kernels) {
            _plugin->SetConfig({ { PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES } }, nullptr);
        }

        // Tuning
        if (FLAGS_tuning.size()) {
            _plugin->SetConfig({ { PluginConfigParams::KEY_TUNING_FILE, FLAGS_tuning } }, nullptr);
            _plugin->SetConfig({ { PluginConfigParams::KEY_TUNING_MODE, PluginConfigParams::TUNING_CREATE } }, nullptr);
        }

        // Power gadget
#ifndef _WIN32
        // Always intialize the power_gov library first
        init_rapl();
#else
        CIntelPowerGadgetLib energyLib;
        if (!FLAGS_pi.empty()) {

            std::ofstream outFile(FLAGS_pi);
            if (!outFile.is_open()) {
                THROW_IE_EXCEPTION << "Can't open " << FLAGS_pi << " for writing!";
            }
            if (energyLib.IntelEnergyLibInitialize() == false) {
                std::cout << "Intel Power Gadget isn't initialized! Message: " << energyLib.GetLastError() << std::endl;
                FLAGS_pi.clear();
            }
        }
#endif



        // Read network
        InferenceEngine::CNNNetReader network;
        network.ReadNetwork(FLAGS_m);
        if (!network.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
        if (network.getNetwork().getBatchSize() != 1) {
            THROW_IE_EXCEPTION << "Only handling batch size 1 networks";
        }

        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        network.ReadWeights(binFileName.c_str());

        if (!FLAGS_d.empty()) {
            network.getNetwork().setTargetDevice(getDeviceFromStr(FLAGS_d));
        }

        std::vector<std::string> args = gflags::GetArgvs();
        if (!FLAGS_layer.empty()) {
            auto layers = ParseFlagList(args, "-layer");
            for (const auto& l : layers) {
                network.getNetwork().addOutput(l);
            }
        }

        InputsDataMap networkInputs;
        networkInputs = network.getNetwork().getInputsInfo();

        // collect input images
        std::vector<std::string> inputNames = CollectImageNames(ParseFlagList(args, "-i"));

        // collect input sizes (before adjusting batch sizes)
        std::vector<size_t> inputSizes;
        for (const auto& in : networkInputs) {
            auto dims = in.second->getDims();
            if (FLAGS_im_info && dims.size() == 2)
                continue; // skip im_info type input size
            inputSizes.push_back(std::accumulate(
                dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>()));
        }

        // set correct batch size
        if ((inputNames.size() % (FLAGS_im_info ? networkInputs.size() - 1 : networkInputs.size())) != 0) {
            THROW_IE_EXCEPTION << "input files aren't a multiple of the network's inputs";
        }
        size_t batchSize = inputNames.size() / (networkInputs.size() - (FLAGS_im_info ? 1 : 0));
        if (FLAGS_im_info) {
            if (batchSize != 1) {
                THROW_IE_EXCEPTION << "im_info only supported for batch size 1";//cant override changes in batch sizes along the network in these types of networks
            }
        }
        else {
            network.getNetwork().setBatchSize(batchSize);
        }
        // read images
        std::vector<std::shared_ptr<unsigned char>> readImages;
        for (size_t i = 0; i < inputNames.size(); i++) {
            FormatReader::ReaderPtr reader(inputNames[i].c_str());
            if (reader.get() == nullptr) {
                THROW_IE_EXCEPTION << "[ERROR]: Image " << inputNames[i] << " cannot be read!";
            }
            if (reader->size() != inputSizes[i % inputSizes.size()]) {
                THROW_IE_EXCEPTION << "[WARNING]: Input sizes mismatch, got " << reader->size() << " bytes, expecting " << inputSizes[i % inputSizes.size()];
            }
            readImages.push_back(reader->getData());
        }

        // create input blobs
        InferenceEngine::BlobMap inputBlobs;
        networkInputs = network.getNetwork().getInputsInfo();//get inputs again after updating batch size
        InferenceEngine::SizeVector imageDims;//for im_info style topologies (frcnn, pvanet) - assuming this will only be used for batch 1
        InferenceEngine::Blob::Ptr im_infoBlob = nullptr;
        InferenceEngine::SizeVector im_infoDims;
        size_t imIndex = 0;
        for (auto& netInput : networkInputs) {
            auto dims = netInput.second->getDims();
            if (FLAGS_im_info && (dims.size() == 2)) {
                if (im_infoBlob != nullptr) THROW_IE_EXCEPTION << "More than 1 im_info type input detected";
                im_infoDims = dims;
                im_infoBlob = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, dims);
                im_infoBlob->allocate();
                inputBlobs[netInput.first] = im_infoBlob;
                continue;
            }
            //normal input
            imageDims = dims;//save in case of im_info


                             // merge images (assume b0[in0 in1 ...] b1[in0 in1 ...] ...
            std::shared_ptr<unsigned char> batchImageData;
            batchImageData.reset(new unsigned char[inputSizes[imIndex] * batchSize], std::default_delete<unsigned char[]>());
            size_t offset = 0;
            for (size_t b = 0; b < batchSize; b++) {
                auto imData = readImages[imIndex + (b*inputSizes.size())];
                for (size_t i = 0; i < inputSizes[imIndex]; i++, offset++) {
                    batchImageData.get()[offset] = imData.get()[i];
                }
            }
            // allocate the input blob
            auto inputBlob = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, dims);
            inputBlob->allocate();

            // convert from byxf(rgb) to bfyx
            InferenceEngine::ConvertImageToInput(batchImageData.get(), inputSizes[imIndex] * batchSize, *inputBlob);

            // set the blobs
            inputBlobs[netInput.first] = inputBlob;
            imIndex++;
        }
        readImages.clear();

        // handle im_info
        if (im_infoBlob != nullptr) {
            auto im_infoData = static_cast<float*>(im_infoBlob->buffer());
            for (size_t b = 0, offset = 0; b < batchSize; b++) {
                im_infoData[offset++] = static_cast<float>(imageDims[1]);
                im_infoData[offset++] = static_cast<float>(imageDims[0]);
                for (size_t i = 2; i < im_infoDims[0]; i++) {
                    im_infoData[offset++] = 1.f;
                }
            }
        }

        InferenceEngine::ResponseDesc dsc;
        InferenceEngine::StatusCode sts;
        // Load custom kernel configuration files
        auto configFiles = ParseFlagList(args, "-custom");
        for (const auto& xml : configFiles) {
            sts = _plugin->SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, xml } }, &dsc);
            if (sts != OK) {
                THROW_IE_EXCEPTION << "Configuration could not be loaded: " << dsc.msg;
            }
            else {
                std::cout << "[INFO] Loaded configuration file: " << xml << std::endl;
            }
        }

        // Load model to plugin
        InferenceEngine::IInferRequest::Ptr req;
        if (FLAGS_newapi) {
            std::map<std::string, std::string> config;
            if (FLAGS_pc) {
                config[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
            }
            if (!configFiles.empty()) {
                // custom layers
                for (auto& file : configFiles) {
                    config[PluginConfigParams::KEY_CONFIG_FILE] += file + " ";
                }
                if (FLAGS_dump_kernels) {
                    // dump custom kernels
                    config[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::YES;
                }
            }
            InferenceEngine::IExecutableNetwork::Ptr net;
            sts = _plugin->LoadNetwork(net, network.getNetwork(), config, &dsc);
            if (sts == InferenceEngine::OK) {
                sts = net->CreateInferRequest(req, &dsc);
            }
        }
        else {
            sts = _plugin->LoadNetwork(network.getNetwork(), &dsc);
        }
        if (sts == InferenceEngine::GENERAL_ERROR) {
            THROW_IE_EXCEPTION << dsc.msg;
        }
        else if (sts == InferenceEngine::NOT_IMPLEMENTED) {
            THROW_IE_EXCEPTION << "Model cannot be loaded! Plugin doesn't support this model!";
        }

        InferenceEngine::BlobMap outputBlobs;
        if (FLAGS_newapi) {
            THROW_IE_EXCEPTION << "NewAPI option is currently broken!";
            //             sts = req->SetInput(inputBlobs, &dsc);
            //             if (sts != InferenceEngine::OK) {
            //                 THROW_IE_EXCEPTION << "Error setting inputs." << dsc.msg;
            //             }
            //             sts = req->GetOutput(outputBlobs, &dsc);
            //             if (sts != InferenceEngine::OK) {
            //                 THROW_IE_EXCEPTION << "Error getting outputs." << dsc.msg;
            //             }
        }
        else {
            InferenceEngine::OutputsDataMap out;
            out = network.getNetwork().getOutputsInfo();
            for (auto &&item : out) {
                InferenceEngine::SizeVector outputDims = item.second->dims;
                InferenceEngine::TBlob<float>::Ptr output;
                item.second->precision = InferenceEngine::Precision::FP32;
                output = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, outputDims);
                output->allocate();
                outputBlobs[item.first] = output;
            }
        }

        // Start measuring power
#ifndef _WIN32
#else
        if (!FLAGS_pi.empty()) {
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            std::wstring pi_filename = converter.from_bytes(FLAGS_pi);
            energyLib.StartLog((wchar_t*)pi_filename.c_str());
        }
#endif
        // Infer model
        auto pos = FLAGS_m.find_last_of("\\/");
        std::string modelName = FLAGS_m.substr(pos >= FLAGS_m.length() ? 0 : pos + 1);
        std::cout << "Model: " << modelName << std::endl;
        std::cout << "Batch: " << batchSize << std::endl;
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;
        double total = 0.0;
        double framesPerSecond = 0.0;
        uint32_t niter = FLAGS_ni;
#ifndef _WIN32       
        double *energyConsumptionPackage = new double[niter + 1];
        //double *energyConsumptionCpu= new double[niter]; for CPU energy consumption
        double *energyConsumptionGpu = new double[niter + 1];
        //double *energyConsumptionDram = new double[niter]; for DRAM energy consumption
        double *timeConsumption = new double[niter];
        double gpuSum = 0.0;
        double packageSum = 0.0;
        double iterationPower = 0.0;
        energyConsumptionPackage[0] = get_rapl_energy_info(0, 0);
        //energyConsumptionCpu[0] = get_rapl_energy_info(1, 0); CPU
        energyConsumptionGpu[0] = get_rapl_energy_info(2, 0);
        //energyConsumptionDram[0] = get_rapl_energy_info(3, 0); DRAM

        for (uint32_t i = 1; i <= niter; ++i) {
#else 
        for (uint32_t i = 0; i < niter; ++i) {
#endif
            auto t0 = Time::now();
            if (FLAGS_newapi) {
                sts = req->Infer(&dsc);
            }
            else {
                sts = _plugin->Infer(inputBlobs, outputBlobs, &dsc);
            }
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += static_cast<double>(d.count());
            if (!FLAGS_pi.empty()) {
#ifndef _WIN32
                if (i <= niter) {
                    energyConsumptionPackage[i] = get_rapl_energy_info(0, 0);
                    //energyConsumptionCpu[i] = get_rapl_energy_info(1, 0); for CPU energy consumption
                    energyConsumptionGpu[i] = get_rapl_energy_info(2, 0);
                    //energyConsumptionDram[i] = get_rapl_energy_info(3, 0); for DRAM energy consumption
                    timeConsumption[i] = static_cast<double>(d.count());
                    packageSum += (energyConsumptionPackage[i] - energyConsumptionPackage[i - 1]) / (timeConsumption[i] / 1000.0);
                    gpuSum += (energyConsumptionGpu[i] - energyConsumptionGpu[i - 1]) / (timeConsumption[i] / 1000.0);
                }
#else
                if (i < (niter - 1)) {
                    energyLib.ReadSample();
                }
                else {
                    energyLib.StopLog();
                }
#endif
            }

        }

        std::cout << "Average running time of one iteration: " << total / static_cast<double>(niter) << " ms" << std::endl;
        framesPerSecond = (static_cast<double>(batchSize) * 1000.0) / (total / static_cast<double>(niter));
        std::cout << "FPS: " << framesPerSecond << std::endl;

#ifndef _WIN32
        double packagePower = packageSum / static_cast<double>(niter);
        double gpuPower = gpuSum / static_cast<double>(niter);
        std::cout << "Total Package Power [W]: " << packagePower << std::endl;
        std::cout << "Total Gpu Power [W]: " << gpuPower << std::endl;
        std::cout << "FPS/Package Power [FPS/W]: " << framesPerSecond / packagePower << std::endl;
        std::cout << "FPS/Gpu Power [FPS/W]: " << framesPerSecond / gpuPower << std::endl;
        delete[] energyConsumptionPackage;
        //delete[] energyConsumptionCpu;
        delete[] energyConsumptionGpu;
        //delete[] energyConsumptionDram;
#endif

        // Check errors
        if (sts == InferenceEngine::GENERAL_ERROR) {
            THROW_IE_EXCEPTION << "Scoring failed! Critical error: " << dsc.msg;
        }
        else if (sts == InferenceEngine::NOT_IMPLEMENTED) {
            THROW_IE_EXCEPTION << "Scoring failed! Input data is incorrect and not supported!";
        }
        else if (sts == InferenceEngine::NETWORK_NOT_LOADED) {
            THROW_IE_EXCEPTION << "Scoring failed! " << dsc.msg;
        }

        // output hashes
        std::cout << "Input/Output Hashes" << std::endl;
        for (auto& in : inputBlobs) {
            std::cout << "Input: \"" << in.first << "\" " << BlobMD5(in.second) << std::endl;
        }
        for (auto& out : outputBlobs) {
            std::cout << "Output: \"" << out.first << "\" " << BlobMD5(out.second) << std::endl;
        }

        if (FLAGS_pc) {
            long long totalTime = 0;
            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;
            // Get perfomance counts
            if (FLAGS_newapi) {
                req->GetPerformanceCounts(perfomanceMap, nullptr);
            }
            else {
                _plugin->GetPerformanceCounts(perfomanceMap, nullptr);
            }
            // Print perfomance counts
            std::cout << std::endl << "Perfomance counts:" << std::endl << std::endl;
            for (std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>::const_iterator it = perfomanceMap.begin();
                it != perfomanceMap.end(); ++it) {
                std::cout << std::setw(30) << std::left << it->first + ":";
                switch (it->second.status) {
                case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                    std::cout << std::setw(15) << std::left << "EXECUTED";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                    std::cout << std::setw(15) << std::left << "NOT_RUN";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                    std::cout << std::setw(15) << std::left << "OPTIMIZED_OUT";
                    break;
                }
                std::cout << std::setw(30) << std::left << "layerType: " + std::string(it->second.layer_type) + " ";
                std::cout << std::setw(20) << std::left << "realTime: " + std::to_string(it->second.realTime_uSec);
                std::cout << std::setw(20) << std::left << " cpu: " + std::to_string(it->second.cpu_uSec);
                std::cout << " execType: " << it->second.exec_type << std::endl;
                if (it->second.realTime_uSec > 0) {
                    totalTime += it->second.realTime_uSec;
                }
            }
            std::cout << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
        }

        //dump raw outputs
        if (!FLAGS_dump.empty()) {
            std::ofstream outFile(FLAGS_dump, std::ofstream::out);
            outFile.precision(10); // f32 should use up to 7 decimal digits
            outFile.fill(' ');
            outFile << "Command line: " << commandLine << std::endl;
            for (auto& output : outputBlobs) {
                outFile << std::endl << "Output " << output.first << ": [";
                auto dims = output.second->dims();
                for (const auto& d : dims) {
                    outFile << d << ",";
                }
                outFile << "] " << BlobMD5(output.second) << std::endl;

                // print raw values
                const TBlob<float>::Ptr pBlob = std::dynamic_pointer_cast<TBlob<float>>(output.second);
                float* pData = pBlob->data();
                std::vector<size_t> newline;
                size_t dimsProd = 1;
                for (const auto& d : dims) {
                    newline.push_back(dimsProd *= d);
                }
                for (size_t i = 0; i < output.second->size(); i++) {
                    for (const auto& d : newline) {
                        if (i%d == 0) {
                            outFile << std::endl;
                        }
                    }
                    outFile << std::setfill(' ') << std::setw(15) << pData[i] << ", ";
                }
                outFile << std::endl;
            }
        }

        // compare with ref
        if (!FLAGS_compare.empty())
        {
            std::vector<size_t> diffBins(8, 0); // counts the number of values with diff  >1e0, >1e-1, >1e-2, ..., >1e-7
            std::vector<float> diffBinsEps;
            for (int i = 0; i < diffBins.size(); i++)
            {
                diffBinsEps.push_back(pow(10.0f, -i));
            }

            // open the csv file
            std::ofstream csvFile;
            if (!FLAGS_csv.empty())
            {
                csvFile.open(FLAGS_csv);
                if (csvFile.is_open())
                {
                    // write the csv header
                    csvFile << "Output, Identical, ";
                    for (int i = int(diffBinsEps.size()) - 1; i >= 0; i--)
                    {
                        csvFile << ">1e-" << i << ", ";
                    }
                    csvFile << "Max Difference\n";
                }
                else
                {
                    std::cout << "[ERROR] CSV file " << FLAGS_csv << " couldn't be created!\n";
                }
            }

            // open ref 
            std::ifstream refFile(FLAGS_compare);
            if (!refFile.is_open())
            {
                std::cout << "[ERROR] Reference file " << FLAGS_compare << " couldn't be opened!\n";
            }
            else {
                // load all layer values from ref file
                std::map<std::string, std::vector<float>> refOutputs; // name, values
                std::string line;
                while (!refFile.eof()) {
                    refFile.clear();//clear flags
                    while (getline(refFile, line) && (line.compare(0, std::string("Output").length(), "Output") != 0));

                    if (!refFile.eof()) {
                        // handle one output

                        // get output name
                        size_t startPos = line.find_first_of(' ') + 1;
                        size_t endPos = line.find_first_of(':');
                        std::string outputName = line.substr(startPos, endPos - startPos);
                        if (outputBlobs.find(outputName) == outputBlobs.end()) {
                            std::cout << "[WARNING] Reference output " << outputName << " was not found in the network outputs!\n";
                            continue;
                        }
                        // read values
                        std::vector<float> outputValues;
                        auto dims = outputBlobs.at(outputName)->dims();
                        outputValues.reserve(std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>()));
                        while (!refFile.fail() && !refFile.eof() && !refFile.bad()) {
                            char tmp;
                            float val;
                            refFile >> val;
                            if (!refFile.fail()) {
                                outputValues.push_back(val);
                                refFile >> tmp;
                            }
                        }

                        // push to map
                        refOutputs[outputName] = outputValues;
                    }
                }

                // compare layer by layer - check existence
                for (auto& output : outputBlobs)
                {
                    if (refOutputs.find(output.first) == refOutputs.end())
                    {
                        std::cout << "[WARNING] Output " << output.first << " was not found in reference dump!\n";
                    }
                    else if (output.second->size() != refOutputs.at(output.first).size()) {
                        std::cout << "[WARNING] Output " << output.first << " has different size than the reference: output("
                            << output.second->size() << ") ref(" << refOutputs.at(output.first).size() << ")\n";
                    }
                    else {
                        float maxDiff = 0.0f;
                        float maxDiffOutput = 0.0f;
                        float maxDiffReference = 0.0f;
                        for (auto& d : diffBins) {
                            d = 0;
                        }

                        const TBlob<float>::Ptr pBlob = std::dynamic_pointer_cast<TBlob<float>>(output.second);
                        float* pOutputValues = pBlob->data();
                        auto& refValues = refOutputs.at(output.first);
                        for (size_t i = 0; i < refValues.size(); i++)
                        {
                            float diff = fabs((pOutputValues[i] * float(FLAGS_scale)) - refValues[i]);
                            if (diff > maxDiff) {
                                maxDiff = diff;
                                maxDiffOutput = pOutputValues[i] * float(FLAGS_scale);
                                maxDiffReference = refValues[i];
                            }

                            for (int i = 0; i < diffBins.size(); i++)
                            {
                                if (diff > diffBinsEps[i])
                                {
                                    diffBins[i]++;
                                    break;
                                }
                            }
                        }

                        // print results
                        auto numValues = refValues.size();
                        std::cout << "Comparison for output \"" << output.first << "\":\n";
                        std::cout << "  FP32 Identical values:       " << std::setw(5) << std::fixed << std::right << std::setprecision(1) << 100.0f *
                            (numValues - std::accumulate(diffBins.begin(), diffBins.end(), size_t(0))) / numValues << "%\n";

                        for (int i = int(diffBinsEps.size()) - 1; i >= 0; i--)
                        {
                            std::cout << "  Error magnitude >1e-" << i << " values:" << std::setprecision(1) << std::setw(5) << std::fixed << std::right
                                << 100.0f * diffBins[i] / numValues << "%\n";
                        }
                        std::cout << std::setprecision(10) << "  Max absolute difference: " << maxDiff
                            << " (output: " << maxDiffOutput << ", reference: " << maxDiffReference << ")\n";

                        // output to csv
                        if (csvFile.is_open())
                        {
                            csvFile << output.first << ", " << 100.0f * (numValues - std::accumulate(diffBins.begin(), diffBins.end(), size_t(0))) / numValues;
                            for (int i = int(diffBinsEps.size()) - 1; i >= 0; i--)
                            {
                                csvFile << ", " << 100.0f * diffBins[i] / numValues;
                            }
                            csvFile << ", " << maxDiff << "\n";
                        }
                    }
                }
            }
        }

        }
    catch (InferenceEngineException ex) {
        std::cerr << ex.what() << std::endl;
        return 3;
    }
    return 0;
    }
