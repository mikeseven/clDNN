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

 __future__ import print_function
import json
import os
import subprocess
import argparse
from collections import OrderedDict
from xml.etree.ElementTree import ElementTree



class consts:
    confom_json_file_name = "conform.json"
    error_msg = "ERROR:"
    output_msg = "Output:"
    batches = [1, 2, 4, 8, 10, 16, 32, 64, 128]
    workloads_folder = "\\\samba-users.igk.intel.com\\samba\\Users\\leszczyn\\clDNN_Validation\\workloads\\"
    images_folder = "\\\samba-users.igk.intel.com\\samba\\Users\\leszczyn\\clDNN_Validation\\workloads\\images\\"
    break_line = "-------------------"

    
class conform_obj:
    def __init__(self):
        self.hashes = OrderedDict({'1': "",
                       '2': "",
                       '4': "",
                       '8': "",
                       '10': "",
                       '16': "",
                       '32': "",
                       '64': "",
                       '128': ""})
        self.proposal = OrderedDict({
                    "bbox_pred_reshape": "",
                    "cls_prob_reshape": "",
                    "proposal": ""
                    })
        self.two_dim_output = OrderedDict({
                    "1": {
                    },
                    "2": {
                    },
                    "4": {
                    },
                    "8": {
                    },
                    "10": {

                    },
                    "16": {

                    },
                    "32": {

                    },
                    "64": {
                    },
                    "128": {
                    }
                })
        pass

    def set_2_dim_output_hash(self, out_name_1, hash_1, out_name_2, hash_2, batch):
        self.two_dim_output[batch][out_name_1] = hash_1
        self.two_dim_output[batch][out_name_2] = hash_2
        pass


class topology:
    def __init__(self, name, inp_size_x, inp_size_y, hashes, is_new_topology=False, count_output=1):
        self.name = name
        self.size_x = inp_size_x
        self.size_y = inp_size_y
        self.hashes = hashes  #conform_obj()
        self.new = is_new_topology
        self.count_output = count_output
        pass

    def get_images_dir_name(self):
        if os.path.exists(consts.images_folder + str(self.size_x)):
            return str(self.size_x)
        elif os.path.exists(consts.images_folder + str(self.size_y)):
            return str(self.size_y)
        elif os.path.exists(consts.images_folder + str(self.size_x) + "x" + str(self.size_y)):
            return str(self.size_x) + "x" + str(self.size_y)
        elif os.path.exists(consts.images_folder + str(self.size_y) + "x" + str(self.size_x)):
            return str(self.size_y) + "x" + str(self.size_x)
        raise Exception("NO DIRECTORY WITH IMAGES FOR TOPOLOGY: " + self.name)


class output_cmd:
    def __init__(self, name, hash):
        self.name = name
        self.hash = hash
        pass

    def __str__(self):
        return self.name + ": " + self.hash


def get_key(key):
    try:
        return int(key)
    except ValueError:
        return key


def open_json_file(file_name):
    json_file = None
    for fn in os.listdir(os.getcwd()):
        if file_name == fn:
            json_file = json.load(open(fn, "r"))
            break

    new_json_file = OrderedDict(sorted(json_file['IE'].items()))
    for key, value in new_json_file.items():
        new_json_file[key] = OrderedDict(sorted(value.items(), key=lambda t: get_key(t[0])))
    ret = {}
    ret['IE'] = new_json_file
    return ret


def update_json_file(conform_file, topo):
    blank = conform_file['IE']['blank']
    del conform_file['IE']['blank']
    if topo.count_output == 1:
        conform_file['IE'][topo.name] = topo.hashes.hashes
    elif topo.count_output == 2:
        conform_file['IE'][topo.name] = topo.hashes.two_dim_output
    elif topo.count_output == 3:
        conform_file['IE'][topo.name] = topo.hashes.proposal
    conform_file['IE']['blank'] = blank
    save_conform_json_file(conform_file)
    pass


def save_conform_json_file(conform_file):
    blank = conform_file['IE']['blank']
    del conform_file['IE']['blank']
    conform_file['IE']['blank'] = blank
    with open(consts.confom_json_file_name, 'w') as json_data:
        json.dump(conform_file, json_data, indent=4)
    pass


def parse_conform_file(conform_file, topology_name, count_output):
    conform_object = conform_obj()
    new_topology = False
    for name, hashes in conform_file['IE'].items():
        if name != topology_name:
            new_topology = True
            continue
        for batch, hash in hashes.items():
            if count_output == 1:
                conform_object.hashes[batch] = hash
            elif count_output == 2:
                conform_object.two_dim_output[batch] = hash
            elif count_output == 3:
                conform_object.proposal[batch] = hash
            new_topology = False
        print(topology_name + "already exists. Hashes will be updated to " + consts.confom_json_file_name)
        break
    if new_topology is True:
        print(topology_name + "is new topology. Will be added to " + consts.confom_json_file_name)
        blank = conform_file['IE']['blank']
        del conform_file['IE']['blank']
        if count_output != 3:
            conform_file['IE'][topology_name] = OrderedDict({
                        "1": "",
                        "2": "",
                        "4": "",
                        "8": "",
                        "10": "",
                        "16": "",
                        "32": "",
                        "64": "",
                        "128": ""
                    })
        else:
            conform_file['IE'][topology_name] = OrderedDict({
                    "bbox_pred_reshape": "",
                    "cls_prob_reshape": "",
                    "proposal": ""
                    })
        conform_file['IE']['blank'] = blank
        save_conform_json_file(conform_file)
    return conform_object, new_topology


def parse_topology_proposal(topology):
    print(consts.break_line)
    print("RUNNING TOPOLOGY: " + topology.name)
    cmd_line_generic = "generic_sample.exe -p clDNNPlugin "
    cmd_line_model = "-m " + consts.workloads_folder + topology.name + ".xml "
    cmd_line_proposal = "-im_info -layer proposal "
    conform_hashes = conform_obj()
    image_folder = consts.images_folder + topology.get_images_dir_name()
    images = []
    for dirpath, dirnames, files in os.walk(os.path.abspath(image_folder)):
        for file in files:
            if '.bmp' in file:
                images.append(file)

    if len(images) == 0:
        raise Exception("ERROR. IS IMAGE IN FOLDER WITH IMAGES?")
    if len(images) > 1:
        raise Exception("ERROR. TOO MUCH IMAGES. FOR PROPOSAL TOPOLOGIES WE DO CONFORM CHECK FOR BATCH 1")

    cmd_line_images = "-i " + consts.images_folder + topology.get_images_dir_name() + "\\1\\" + images[0]

    try:
        cmd_line_output = subprocess.check_output(cmd_line_generic + cmd_line_proposal + cmd_line_model + cmd_line_images, shell=True).decode("utf-8", "ignore").splitlines()
    except:
        print("EXECUTION ERROR")
        cmd_line_output = [""]
    for line in cmd_line_output:
        print(line)
    output = []
    error = []
    for line in cmd_line_output:
        line = line.replace('\"',"")
        if consts.output_msg in line:
            splited_string = line.split()
            output.append(output_cmd(splited_string[1], splited_string[2]))
        if consts.error_msg in line:
            error.append(line)
    if len(error) == 0 and len(output) == 3:
        for out in output:
            print(out)
        conform_hashes.proposal[output[0].name] = output[0].hash
        conform_hashes.proposal[output[1].name] = output[1].hash
        conform_hashes.proposal[output[2].name] = output[2].hash
    if len(error) != 0:
        for err in error:
            print(err)
        for key,value in conform_hashes.proposal.items():
            conform_hashes.proposal[key] = 'ERROR'
    if len(error) == 0 and len(output) == 0:
        print("ERROR! NO OUTPUT HASH")
        for key,value in conform_hashes.proposal.items():
            conform_hashes.proposal[key] = 'x'

    if topology.count_output == 3:
        if topology.new is False:
            for key, value in topology.hashes.proposal.items():
                if value == conform_hashes.proposal[key]:
                    print(key + " hash has not changed.")
                    continue
                if value != conform_hashes.proposal[key]:
                    topology.hashes.proposal[key] = conform_hashes.proposal[key]
                    print(key + " hash updated.")
        else:
            for key, value in topology.hashes.proposal.items():
                topology.hashes.proposal[key] = conform_hashes.proposal[key]
                print(key + " hash has been created.")
    return topology


def parse_topology(topology):
    print(consts.break_line)
    print("RUNNING TOPOLOGY: " + topology.name)
    cmd_line_generic = "generic_sample.exe -p clDNNPlugin "
    cmd_line_model = "-m " + consts.workloads_folder + topology.name + ".xml "
    conform_hashes = conform_obj()

    for batch in consts.batches:
        batch_str = str(batch)
        cmd_line_images = "-i " + consts.images_folder + topology.get_images_dir_name() + "\\"
        cmd_line_images += batch_str
        print("BATCH: " + batch_str)
        
        try:
            cmd_line_output = subprocess.check_output(cmd_line_generic + cmd_line_model + cmd_line_images, shell=True).decode("utf-8", "ignore").splitlines()
        except:
            print("EXECUTION ERROR")
            cmd_line_output = [""]

        output = []
        error = []
        for line in cmd_line_output:
            line = line.replace('\"',"")
            if consts.output_msg in line:
                splited_string = line.split()
                output.append(output_cmd(splited_string[1], splited_string[2]))
            if consts.error_msg in line:
                error.append(line)
        if len(error) == 0 and len(output) == 1:
            for out in output:
                print(out)
            conform_hashes.hashes[batch_str] = output[0].hash
        if len(error) == 0 and len(output) == 2:
            for out in output:
                print(out)
            conform_hashes.set_2_dim_output_hash(json.loads(json.dumps(output[0].name)), output[0].hash, json.loads(json.dumps(output[1].name)), output[1].hash, batch_str)
        if len(error) == 0 and len(output) == 3:
            for out in output:
                print(out)
            conform_hashes.set_2_dim_output_hash(json.loads(json.dumps(output[0].name)), output[0].hash, json.loads(json.dumps(output[1].name)), output[1].hash, batch_str)
        if len(error) != 0:
            for err in error:
                print(err)
            conform_hashes.hashes[batch_str] = "ERROR"
        if len(error) == 0 and len(output) == 0:
            print("ERROR! NO OUTPUT HASH")
            conform_hashes.hashes[batch_str] = "x"

    if topology.count_output == 1:
        if topology.new is False:
            for key, value in topology.hashes.hashes.items():
                if value == conform_hashes.hashes[key]:
                    print("Batch: " + key + " hash has not changed.")
                    continue
                if value != conform_hashes.hashes[key]:
                    topology.hashes.hashes[key] = conform_hashes.hashes[key]
                    print("Batch: " + key + " hash updated.")
        else:
            for key, value in topology.hashes.hashes.items():
                topology.hashes.hashes[key] = conform_hashes.hashes[key]
                print("Batch: " + key + " hash has been created.")
    elif topology.count_output == 2:
        if topology.new is False:
            for key, value in topology.hashes.two_dim_output.items():
                if value == conform_hashes.two_dim_output[key]:
                    print("Batch: " + key + " hash has not changed.")
                    continue
                if value != conform_hashes.two_dim_output[key]:
                    topology.hashes.two_dim_output[key] = conform_hashes.two_dim_output[key]
                    print("Batch: " + key + " hash updated.")
        else:
            for key, value in topology.hashes.two_dim_output.items():
                topology.hashes.two_dim_output[key] = conform_hashes.two_dim_output[key]
                print("Batch: " + key + " hash has been created.")
    return topology


def main(): # conform json and generic_sample with all dependecies need to be in the same folder as this script. Topology .xml file and images need to be in the cldnnValidation workload folder.
    parser = argparse.ArgumentParser(description='Dump output hashes to conform.json')
    parser.add_argument("-i", "--info",
                        help="info about the script", action="store_true")
    parser.add_argument('-t', "--topology", type=str, nargs='+',
                        help='names of topologies')
    args = parser.parse_args()

    if args.info:
        print("conform json and generic_sample with all dependecies need to be in the same folder as this script. Topology .xml file and images need to be in the cldnnValidation workload folder.")
        print("example command line: dump_hash_to_conform_json.py --topology CommunityGoogleNetV2 ResNet-18_fp16 NewTopology_TincaTinca-fp128")
        exit(0)


    print("WORKLOAD FOLDER: " + consts.workloads_folder)
    print("IMAGES FOLDER: " + consts.images_folder)
    print("CONFORM FILE NAME: " + consts.confom_json_file_name)
    conform_file = open_json_file(consts.confom_json_file_name)

    array_of_topologies = []

    for arg in args.topology:
        if ".xml" not in arg:
            full_name = arg + ".xml"
        xml_root = ElementTree()
        xml_root.parse(consts.workloads_folder + full_name)
        input = xml_root.find("input")
        count_output = 1
        input_arr = []
        if "age_gender" in arg:
            count_output = 2
        dim_objs = None
        if input is not None:
            dim_objs = input.findall("dim")
        elif input is None:
            layers = xml_root.find("layers")
            for layer in layers:
                if layer.attrib['type'] == 'Input' or layer.attrib['type'] == 'input':
                    input_arr.append(layer)
                if layer.attrib['type'] == 'Proposal' or layer.attrib['type'] == 'proposal':
                    proposal = layer
                    count_output = 3

        if len(input_arr) == 1:
            input = input_arr[0]
        elif len(input_arr) == 2:
            for inp in input_arr:
                if inp.attrib['name'] != 'im_info':
                    input = inp


        if dim_objs is None:
            dim_objs = input.findall("output//port//dim")
        dim = []
        for d in dim_objs:
            if d.text != '1' and d.text != '3':
                dim.append(int(d.text))
        if len(dim) != 2:
            raise Exception("Is the input of the topology " + arg + " has the correct sizes?")

        print(count_output)
        hashes, is_new_topology = parse_conform_file(conform_file, arg, count_output)
        array_of_topologies.append(topology(arg, dim[0], dim[1], hashes, is_new_topology, count_output))
        print("LOADED TOPOLOGY CORRECTLY: " + arg)

    for topo in array_of_topologies:
        if topo.count_output != 3:
            parsed_topo = parse_topology(topo)
        else:
            parsed_topo = parse_topology_proposal(topo)
        update_json_file(conform_file, parsed_topo)
        print("Saved topology: " + parsed_topo.name + " to " + consts.confom_json_file_name)

    ret = open_json_file(consts.confom_json_file_name) #for ordering purpose.
    save_conform_json_file(ret)
    print(consts.break_line)
    print("FINISHED CORRECTLY.")
    pass


if __name__ == "__main__":
    main()


