#!/usr/bin/python
import os
import os.path
import errno
import sys
import csv
import ConfigParser
import time
import subprocess
import glob
import argparse
from collections import namedtuple
from winsound import Beep

#################################################################################################
Network = namedtuple("Network", "name binary xml image_dir inputs")

#################################################################################################
class NetRunner:
    #############################################################################################
    def __init__(self):
        logFileName = "log_{}.txt".format(time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log = open(logFileName, 'wb')
        parser = argparse.ArgumentParser(description="Net Runner", version="Version 0.2")
        parser.add_argument('dir', help="Workloads directory")
        parser.add_argument('-i', help="Number of running iterations", action="store", dest="iterations", type=int,
                            default=3, metavar=('iterations'))
        parser.add_argument('-b', help="Batch size", action="store", dest="batch", type=int, default=-1,
                            metavar=('batch'))
        parser.add_argument('-q', '--quick', help="Run single example from each topology", action="store_true")
        parser.add_argument('-p', help="Plugin name", action="store", dest="plugin", default="clDNNPlugin",
                            metavar=('plugin.dll'))
        parser.add_argument('-dump', help="Create reference results in directory", dest="create_ref_dir", default="",
                            metavar=('<dir>'))
        parser.add_argument('-compare', help="Compare to reference results in directory", dest="compare_ref_dir",
                            default="", metavar=('<dir>'))
        self.args = parser.parse_args()
        self.nets = [
            Network("Alexnet no mean", "generic_sample.exe", "bvlc_alexnet_fp32_no_mean.xml", "227x227", 1),
            Network("Alexnet_fp16", "generic_sample.exe", "bvlc_alexnet_fp16.xml", "227x227", 1),
            Network("Alexnet mean file", "generic_sample.exe", "bvlc_alexnet_fp32_mf.xml", "227x227", 1),
            Network("Alexnet_fp16 mean file", "generic_sample.exe", "bvlc_alexnet_fp16_mf.xml", "227x227", 1),
            Network("SqueezeNet", "generic_sample.exe", "SqueezeNet_1_0_fp32.xml", "227x227", 1),
            Network("SqueezeNet_fp16", "generic_sample.exe", "SqueezeNet_1_0_fp16.xml", "227x227", 1),
            Network("VGG16", "generic_sample.exe", "VGG_ILSVRC_16_layers_fp32_no_mean.xml", "224x224", 1),
            Network("VGG16_fp16", "generic_sample.exe", "VGG_ILSVRC_16_layers_fp16.xml", "224x224", 1),
            Network("GoogleNet_v1", "generic_sample.exe", "bvlc_googlenet_fp32_no_mean.xml", "224x224", 1),
            Network("GoogleNet_v1_fp16", "generic_sample.exe", "GoogleNet_f16.xml", "224x224", 1),
            Network("GoogleNet_v2", "generic_sample.exe", "CommunityGoogleNetV2.xml", "224x224", 1),
            Network("GoogleNet_v2_fp16", "generic_sample.exe", "CommunityGoogleNetV2_fp16.xml", "224x224", 1),
            Network("GoogleNet_v3", "generic_sample.exe", "GoogleNetV3_fp32.xml", "299x299", 1),
            Network("GoogleNet_v3_fp16", "generic_sample.exe", "GoogleNetV3_fp16.xml", "299x299", 1),
            Network("Resnet-50", "generic_sample.exe", "ResNet-50-no_scaleshift.xml", "224x224", 1),
            Network("Resnet-50_fp16", "generic_sample.exe", "ResNet-50_fp16.xml", "224x224", 1),
            Network("Resnet-152", "generic_sample.exe", "ResNet-152_fp32.xml", "224x224", 1),
            Network("Resnet-152_fp16", "generic_sample.exe", "ResNet-152_fp16.xml", "224x224", 1),
            Network("Resnet-101", "generic_sample.exe", "ResNet-101.xml", "224x224", 1),
            Network("Resnet-101_fp16", "generic_sample.exe", "ResNet-101_fp16.xml", "224x224", 1),
            Network("VGG_Face_16", "generic_sample.exe", "VGG_FACE_16_layers.xml", "224x224", 1),
            Network("VGG_Face_16_fp16", "generic_sample.exe", "VGG_FACE_16_layers_fp16.xml", "224x224", 1),
            Network("VGG19", "generic_sample.exe", "VGG_ILSVRC_19_layers_fp32.xml", "224x224", 1),
            Network("VGG19_fp16", "generic_sample.exe", "VGG_ILSVRC_19_layers_fp16.xml", "224x224", 1),
            Network("PerC_Faster-RCNN", "generic_sample.exe -im_info -layer proposal", "Perc_ZF_fasterRCNN_fp32.xml", "frcnn_images", 1),
            Network("PerC_Faster-RCNN_fp16", "generic_sample.exe -im_info -layer proposal", "Perc_ZF_fasterRCNN_fp16.xml", "frcnn_images", 1),
            Network("PerC_Faster-RCNN_dropout_fuse", "generic_sample.exe -im_info -layer proposal", "Perc_ZF_fasterRCNN_fp32_dropout_fuse.xml", "frcnn_images", 1),
            Network("PerC_Faster-RCNN_dropout_fuse_fp16", "generic_sample.exe -im_info -layer proposal", "Perc_ZF_fasterRCNN_fp16_dropout_fuse.xml", "frcnn_images", 1),
            Network("FCN8_fp32", "generic_sample.exe", "fcn_f32.xml", "500x500", 1),
            Network("FCN8_fp16", "generic_sample.exe", "fcn_f16.xml", "500x500", 1),
            Network("SSD_Public_fp32", "generic_sample.exe", "SSD_300x300_deploy_f32.xml", "300x300", 1),
            Network("SSD_Public_fp16", "generic_sample.exe", "SSD_300x300_deploy_fp16.xml", "300x300", 1),
            Network("SSD_GoogleNetV2_fp32", "generic_sample.exe", "SSD_GoogleNet_v2_fp32.xml", "300x300", 1),
            Network("SSD_GoogleNetV2_fp16", "generic_sample.exe", "SSD_GoogleNet_v2_fp16.xml", "300x300", 1),
            Network("PCDetect0026(PVANET)_fp32", "generic_sample.exe -im_info -layer proposal", "PVANET_fp32.xml", "992x544", 1),
            Network("PCDetect0026(PVANET)_fp16", "generic_sample.exe -im_info -layer proposal", "PVANET_fp16.xml", "992x544", 1),
            Network("PVANET_reid_fp32", "generic_sample.exe", "PVANET_Reid_fp32.xml", "64x128", 2),
            Network("PVANET_reid_fp16", "generic_sample.exe", "PVANET_Reid_fp16.xml", "64x128", 2),
            Network("YOLO_v1_tiny_fp32", "generic_sample.exe", "yolo_v1_tiny_yanli_fp32.xml", "448x448", 1),
            Network("YOLO_v1_tiny_fp16", "generic_sample.exe", "yolo_v1_tiny_yanli_fp16.xml", "448x448", 1),
            Network("MobileNet_TF_fp32", "generic_sample.exe", "mobileNet.xml", "224x224", 1),
        ]
        self.batchsizes = [1, 8]
        if self.args.batch > 0:
            self.batchsizes = [self.args.batch]

        if self.args.quick:
            self.batchsizes = [1]

        if len(self.args.create_ref_dir) > 0:
            if not os.path.exists(self.args.create_ref_dir):
                os.makedirs(self.args.create_ref_dir)

        if len(self.args.compare_ref_dir) > 0:
            if not os.path.exists(self.args.compare_ref_dir):
                os.makedirs(self.args.compare_ref_dir)

    #############################################################################################
    def GetIdenticalValues(self, res):
        for line in res.split('\n'):
            words = line.split()
            if ((len(words) > 0) and (words[0] == "FP32")):
                return words[-1]

    #############################################################################################
    def RunAndLog(self, cmd):
        try:
            print("Running {}\n".format(cmd))
            self.log.write("Running {}\n".format(cmd))
            res, err = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
            self.log.write("{}\n".format(res))
            if len(self.args.compare_ref_dir) > 0:
                sameVals = self.GetIdenticalValues(res)
                if (sameVals != "100.0%"):
                    self.log.write("\n===Found output with only {} identical values!===\n\n".format(sameVals))
        except Exception, e:
            self.log.write("Error in command \"{}\" : {}!\n".format(cmd, str(e)))

    #############################################################################################
    def CollectImages(self, dir, ext):
        imageList = []
        for file in os.listdir(dir):
            if file.endswith(ext):
                imageList.append(file)
        return imageList

    #############################################################################################
    def CreateRefArgs(self, net, batch):
        if len(self.args.create_ref_dir) == 0:
            return ""
        return "-dump {}\{}_b{}_ref.txt".format(self.args.create_ref_dir, net.xml[:-4], batch)

    #############################################################################################
    def CompareRefArgs(self, net, batch):
        if len(self.args.compare_ref_dir) == 0:
            return ""
        return "-compare {}\{}_b{}_ref.txt -csv {}\{}_b{}.csv".format(
            self.args.compare_ref_dir, net.xml[:-4], batch,
            self.args.compare_ref_dir, net.xml[:-4], batch)

    #############################################################################################
    def RunSingleNet(self, net):
        imagesDir = self.args.dir + "\\" + net.image_dir
        imageList = self.CollectImages(imagesDir, ".bmp")
        self.log.write("Running {} for images in {} (found {} images)\n".format(net.name, imagesDir, len(imageList)))
        imageList = imageList * net.inputs * (
            1 + max(self.batchsizes) / len(imageList))  # make sure we've got anough for all batch sizes
        for b in self.batchsizes:
            self.log.write("Running Batch Size {}\n".format(b))
            b2 = (b * net.inputs);  # multiply by number of inputs for actual number of images to collect
            imageString = ""
            for i in range(b2):
                imageString += " -i \"{}\\{}\\{}\"".format(self.args.dir, net.image_dir, imageList[i])
            self.RunAndLog("{} -p {} -m {}\\{} {} -ni {} {} {}"
                           .format(net.binary, self.args.plugin, self.args.dir, net.xml, imageString,
                                   self.args.iterations, self.CreateRefArgs(net, b), self.CompareRefArgs(net, b)))
            if net.xml.startswith("Perc_ZF_fasterRCNN") or net.name.startswith("PCDetect0026"):  # only B1 nets
                break;
            if self.args.quick:
                break;

    #############################################################################################
    def RunNets(self):
        for n in self.nets:
            self.RunSingleNet(n)
    #############################################################################################
def main():
    nr = NetRunner()
    nr.RunNets()

#################################################################################################

main()
