#!/usr/bin/env python2
# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation
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


from pathlib import Path
import os
import argparse
from xml.etree.ElementTree import ElementTree
import glob


class consts:
    workloads_folder = "\\\samba-users.igk.intel.com\\samba\\Users\\leszczyn\\clDNN_Validation\\workloads\\"
    images_folder = "\\\samba-users.igk.intel.com\\samba\\Users\\leszczyn\\clDNN_Validation\\workloads\\images\\"
    win_generic_sample_path = "C:\\Tests\\IE\\bin\\IE\\generic_sample.exe"
    win_models_path = "C:\\Tests\\IE\\bin\\models\\"
    win_images_path = "C:\\Tests\\IE\\bin\\images\\"
    win_golden_dumps_path = "C:\\Tests\\golden_dumps\\"
    win_pi_power = "-pi C:\Tests\power.csv"
    win_tuning = "-tuning C:\Tests\AT_CACHE"
    linux_workaround = "workaround.sh, && sudo "
    linux_generic_sample_path = "/tmp/Tests/IE/bin/IE_LNX/generic_sample"
    linux_models_path = "/tmp/Tests/IE/bin/models/"
    linux_images_path = "/tmp/Tests/IE/bin/images/"
    linux_pi_power = "-pi power"
    batch_str = {1: "b001", 2: "b002", 4: "b004", 8: "b008", 10: "b010", 16: "b016", 32: "b032", 64: "b064", 128: "b128"}
    suffix = ",,1"
    break_line = "-------------------"

class topology_info:
    def __init__(self, name="", x=0, y=0):
        self.name = name
        self.size_x = x
        self.size_y = y
        self.images_dir_name = self.get_images_dir_name()
        self.fp32 = self.is_fp32()
        pass

    def is_fp32(self):
        if "fp32" in self.name.lower():
            return True
        return False

    def get_images_dir_name(self):
        suffix = ""
        if "PNet_fp16" in self.name:
            suffix = "_face"
        if os.path.exists(consts.images_folder + str(self.size_x)):
            return str(self.size_x) + suffix
        elif os.path.exists(consts.images_folder + str(self.size_y)):
            return str(self.size_y) + suffix
        elif os.path.exists(consts.images_folder + str(self.size_x) + "x" + str(self.size_y)):
            return str(self.size_x) + "x" + str(self.size_y) + suffix
        elif os.path.exists(consts.images_folder + str(self.size_y) + "x" + str(self.size_x)):
            return str(self.size_y) + "x" + str(self.size_x) + suffix
        raise Exception("NO DIRECTORY WITH IMAGES FOR TOPOLOGY: " + self.name)


class ie_generic:
    def __init__(self):
        self.folder_name = "\\cldnn_ie_generic\\scenarios"
        self.file_filter = ['IE_b', '_FP']
        self.batch_filter = ["1", "2", "4", "8", "10", "16", "32", "64", "128"]
        self.fp_filter = ["16", "32"]
        pass

    def check_file(self, file_name):
        for bf in self.batch_filter:
            for fp in self.fp_filter:
                pattern = self.file_filter[0] + bf + self.file_filter[1] + fp + ".csv"
                if pattern == file_name:
                    return True
        return False

    def return_registry(self, topol_info, batch_key):
        ret = "YES,400," + topol_info.name + "_" + consts.batch_str[batch_key] + "," + consts.win_generic_sample_path + "," + \
            " -ni 20 -p clDNNPlugin -m " + consts.win_models_path + topol_info.name + ".xml" + \
            " -i " + consts.win_images_path + topol_info.images_dir_name + "\\" + str(batch_key) + \
            " " + consts.win_pi_power + consts.suffix
        return ret

class ie_generic_lnx:
    def __init__(self):
        self.folder_name = "\\cldnn_ie_generic_lnx\\scenarios"
        self.file_filter = ['IE_b', '_FP']
        self.batch_filter = ["1", "2", "4", "8", "10", "16", "32", "64", "128"]
        self.fp_filter = ["16", "32"]
        pass

    def check_file(self, file_name):
        for bf in self.batch_filter:
            for fp in self.fp_filter:
                pattern = self.file_filter[0] + bf + self.file_filter[1] + fp + ".csv"
                if pattern == file_name:
                    return True
        return False

    def return_registry(self, topol_info, batch_key):
        ret = "YES,400," + topol_info.name + "_" + consts.batch_str[batch_key] + "," + consts.linux_workaround + \
            consts.linux_generic_sample_path + " -ni 10 -p clDNNPlugin -m " + consts.linux_models_path + topol_info.name + \
            ".xml -i " + consts.linux_images_path + topol_info.images_dir_name + "/" + str(batch_key) + \
            " " + consts.linux_pi_power + consts.suffix
        return ret

class ie_generic_dump:
    def __init__(self):
        self.folder_name = "\\cldnn_ie_generic_dump\\scenarios"
        self.file_filter = ['AT_IE_b', "_FP"]
        self.batch_filter = ["1", "16", "32"]
        self.fp_filter = ["16", "32"]
        self.special_b1b8 = "AT_IE_b1b8_FP16"
        pass

    def check_file(self, file_name):
        for bf in self.batch_filter:
            for fp in self.fp_filter:
                pattern = self.file_filter[0] + bf + self.file_filter[1] + fp + ".csv"
                if pattern == file_name:
                    return True
        if self.special_b1b8 + ".csv" == file_name:
            return True
        return False

    def return_registry(self, topol_info, batch_key, fp_value_str, special_b1b8_fp16=False):
        end_line = " " + consts.win_tuning + "_"+ fp_value_str + "_b" + str(batch_key) + ".txt" + consts.suffix
        if special_b1b8_fp16 is True:
            end_line = " " + consts.win_tuning + fp_value_str + "_B1B8.txt" + consts.suffix
        ret = "YES,5000," + topol_info.name + "_" + consts.batch_str[batch_key] + "," + consts.win_generic_sample_path + "," + \
            " -ni 20 -p clDNNPlugin -m " + consts.win_models_path + topol_info.name + ".xml" + \
            " -i " + consts.win_images_path + topol_info.images_dir_name + "\\" + str(batch_key) + end_line
        return ret


class ie_generic_conform:
    def __init__(self):
        self.folder_name = "\\cldnn_ie_generic_conform\\scenarios"
        self.file_filter = ['IE_FP']
        self.suffix_filter = ["_conform", "_conform_all", "_conform_all_dump"]
        self.fp_filter = ["16", "32"]
        pass

    def check_file(self, file_name):
        for fp in self.fp_filter:
            for sf in self.suffix_filter:
                pattern = self.file_filter[0] + fp + sf + ".csv"
                if pattern == file_name:
                    return True
        return False

    def return_registry(self, topol_info, batch_key, dump=False):
        comp_or_dump = " -compare "
        if dump is True:
            comp_or_dump = " -dump "
        ret = "YES,400," + topol_info.name + "_" + consts.batch_str[batch_key] + "," + consts.win_generic_sample_path + "," + \
            " -ni 1 -p clDNNPlugin -m " + consts.win_models_path + topol_info.name + ".xml" + \
            " -i " + consts.win_images_path + topol_info.images_dir_name + "\\" + str(batch_key) + \
            comp_or_dump + consts.win_golden_dumps_path + topol_info.name + "_" + consts.batch_str[batch_key] + ".txt" + consts.suffix
        return ret


def return_csvs(ie_gen_class):
    dir = os.getcwd() + ie_gen_class.folder_name
    pathlist = Path(dir).glob('**/*.csv')
    fis = []
    for path in pathlist:
        path_in_str = str(path)
        file_name = path_in_str[len(dir)+1:]
        if ie_gen_class.check_file(file_name) is True:
            fis.append(path_in_str)
    return fis


def main():
    ie_gen = ie_generic()
    ie_gen_lnx = ie_generic_lnx()
    ie_gen_conform = ie_generic_conform()
    ie_gen_dump = ie_generic_dump()

    ie_gen_csvs = return_csvs(ie_gen)
    ie_gen_lnx_csvs = return_csvs(ie_gen_lnx)
    ie_gen_conform_csvs = return_csvs(ie_gen_conform)
    ie_gen_dump_csvs = return_csvs(ie_gen_dump)

    all_csvs = [ie_gen_csvs, ie_gen_lnx_csvs, ie_gen_conform_csvs, ie_gen_dump_csvs]

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--add", type=str, nargs='+',
                        help="add topology to csvs")
    parser.add_argument("-r", "--remove", type=str, nargs='+',
                        help="remove topology from csvs")
    parser.add_argument("-i", "--info",
                        help="info about the script", action="store_true")

    args = parser.parse_args()
    add_topologies_names = args.add
    remove_topologies_names = args.remove
    if args.info:
        print("Add topology (-add) or remove (-remove) from csvs, which are in the same directory as this script.")
        exit(0)

    #-------------------- ADDING TOPOLGOIES TO CSVS---------------------------
    for arg in add_topologies_names:
        if ".xml" not in arg:
            full_name = arg + ".xml"
        xml_root = ElementTree()
        xml_root.parse(consts.workloads_folder + full_name)
        input = xml_root.find("input")
        input_arr = []

        if input is not None:
            dim_objs = input.findall("dim")
        elif input is None:
            layers = xml_root.find("layers")
            for layer in layers:
                if layer.attrib['type'] == 'Input' or layer.attrib['type'] == 'input':
                    input_arr.append(layer)

        if len(input_arr) == 1:
            input = input_arr[0]
        elif len(input_arr) == 2:
            for inp in input_arr:
                if inp.attrib['name'] != 'im_info':
                    input = inp


        if input is not None:
            dim_objs = input.findall("output//port//dim")
        dim = []
        for d in dim_objs:
            if d.text != '1' and d.text != '3':
                dim.append(int(d.text))
        if len(dim) != 2:
            raise Exception("Is the input of the topology " + arg + " has the correct sizes?")

        topo_info = topology_info(arg, dim[0], dim[1])
        verbose_print = {}


        for csv_group in all_csvs:
            for csv in csv_group:
                verbose_print["FILE NAME: " + csv] = ["", ""]


        for csv in ie_gen_csvs:
            verbose_print["FILE NAME: " + csv] = ["", ""]
            batch = int(csv[csv.find("_b") + 2: csv.find("_", csv.find("_b") + 2)])
            if topo_info.fp32 is True:
                if "FP32" in csv:
                    new_registry = ie_gen.return_registry(topo_info, batch)
                    with open(csv, 'a') as outfile:
                        outfile.write("\n" + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry
            else:
                if "FP16" in csv:
                    new_registry = ie_gen.return_registry(topo_info, batch)
                    with open(csv, 'a') as outfile:
                        outfile.write("\n" + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry

        for csv in ie_gen_lnx_csvs:
            verbose_print["FILE NAME: " + csv] = ["", ""]
            batch = int(csv[csv.find("_b") + 2: csv.find("_", csv.find("_b") + 2)])
            if topo_info.fp32 is True:
                if "FP32" in csv:
                    new_registry = ie_gen_lnx.return_registry(topo_info, batch)
                    with open(csv, 'a') as outfile:
                        outfile.write("\n" + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry
            else:
                if "FP16" in csv:
                    new_registry = ie_gen_lnx.return_registry(topo_info, batch)
                    with open(csv, 'a') as outfile:
                        outfile.write("\n" + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry

        for csv in ie_gen_conform_csvs:
            verbose_print["FILE NAME: " + csv] = ["", ""]
            conf_all = False
            dump_all = False
            if "all" in csv and "dump" not in csv:
                conf_all = True
            if "_all_dump" in csv:
                dump_all = True
            if topo_info.fp32 is True:
                if "FP32" in csv:
                    if conf_all:
                        infos = []
                        for batch, _ in consts.batch_str.items():
                            new_registry = ie_gen_conform.return_registry(topo_info, batch, dump=False)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
                    elif dump_all:
                        infos = []
                        for batch, _ in consts.batch_str.items():
                            new_registry = ie_gen_conform.return_registry(topo_info, batch, dump=True)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
                    else:
                        infos = []
                        batch = [1, 8]
                        for b in batch:
                            new_registry = ie_gen_conform.return_registry(topo_info, b, dump=False)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
            else:
                if "FP16" in csv:
                    if conf_all:
                        infos = []
                        for batch, _ in consts.batch_str.items():
                            new_registry = ie_gen_conform.return_registry(topo_info, batch, dump=False)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
                    elif dump_all:
                        infos = []
                        for batch, _ in consts.batch_str.items():
                            new_registry = ie_gen_conform.return_registry(topo_info, batch, dump=True)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
                    else:
                        infos = []
                        batch = [1, 8]
                        for b in batch:
                            new_registry = ie_gen_conform.return_registry(topo_info, b, dump=False)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos

        for csv in ie_gen_dump_csvs:
            verbose_print["FILE NAME: " + csv] = ["", ""]
            special_b1b8_fp16 = False
            batches = [1, 16, 32]
            if "AT_IE_b1b8_FP16.csv" in csv:
                special_b1b8_fp16 = True

            if topo_info.fp32 is True:
                if "FP32" in csv:
                    batch = int(csv[csv.find("_b") + 2: csv.find("_", csv.find("_b") + 2)])
                    new_registry = ie_gen_dump.return_registry(topo_info, batch, "FP32", False)
                    with open(csv, 'a') as outfile:
                        outfile.write("\n" + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry
            else:
                if "FP16" in csv:
                    if special_b1b8_fp16 is True:
                        infos = []
                        for b in [1, 8]:
                            new_registry = ie_gen_dump.return_registry(topo_info, b, "FP16", special_b1b8_fp16=True)
                            with open(csv, 'a') as outfile:
                                outfile.write("\n" + new_registry)
                                infos.append("Added line: " + new_registry)
                        verbose_print["FILE NAME: " + csv][0] = infos
                    else:
                        batch = int(csv[csv.find("_b") + 2: csv.find("_", csv.find("_b") + 2)])
                        new_registry = ie_gen_dump.return_registry(topo_info, batch, "FP16", special_b1b8_fp16=False)
                        with open(csv, 'a') as outfile:
                            outfile.write("\n" + new_registry)
                            verbose_print["FILE NAME: " + csv][0] = "Added line: " + new_registry



    #---------- REMOVING TOPOLOGIES FROM CSVS------------------------
    full_names = []
    for arg in remove_topologies_names:
        if ".xml" not in arg:
            arg = arg + ".xml"
        full_names.append(arg)


    for csv_group in all_csvs:
        for csv in csv_group:
            temp = ""
            with open(csv, 'r+') as infile:
                for line in infile:
                    found = False
                    for name in full_names:
                        if "C:/Tests/IE/bin/models/" + name in line or "C:\\Tests\\IE\\bin\\models\\" + name in line:
                            found = True
                            verbose_print["FILE NAME: " + csv][1] = "Removed line: " + line
                            break
                        else:
                            found = False
                        if found is False:
                            temp += line
                infile.seek(0)
                infile.truncate()
            with open(csv, 'a') as outfile:
                outfile.write(temp)

    for file_name, values in verbose_print.items():
        if values[0] == "" and values[1] == "":
            continue
        print("----" + file_name + "----")
        print(values[0])
        print(values[1])
    pass

if __name__ == "__main__":
    main()

