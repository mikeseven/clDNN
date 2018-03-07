1. create_image_batches.py -> creating folders with images(batches). Accepts many images (of equal or different sizes).
			      1. Images has to be in the script working directory.
	example command line: create_image_batches.py --images bear227 cat127 cat227


2. dump_hash_to_conform_json.py -> update (add or modify actual hashes) conform.json with topology (or many topologies)
				   1. conform.json have to be in same directory as this script.
			       2. generic_sample with all needed dependencies need to be in the same directory as this script
				   3. topology have to be placed in \\samba-users.igk.intel.com\samba\Users\leszczyn\clDNN_Validation\workloads
				   4. images have to be placed in   \\samba-users.igk.intel.com\samba\Users\leszczyn\clDNN_Validation\workloads\images
	example command line: dump_hash_to_conform_json.py --topology CommunityGoogleNetV2 ResNet-18_fp16 NewTopology_TincaTinca-fp128
			     dump_hash_to_conform_json.py --all


3. modify_csv.py -> add or remove topologies from csv's.
		1. Accepts csvs from folders: 
			-ie_generic
			-ie_generic_lnx
			-ie_generic_conform
			-ie_generic_dump
		2. Only pattern csv's will be modifed (i.e IE_AlexNet_FP32_mf.csv from ie_generic will not be modifed, but all IE_batch<b>_fp<fp>.csv will be modfied.
		3. Directories with csv's need to be in the same directory as this script.  
		4. New topologies xml need to be placed in \\samba-users.igk.intel.com\samba\Users\leszczyn\clDNN_Validation\workloads
		5. New topologies images need to be placed in \\samba-users.igk.intel.com\samba\Users\leszczyn\clDNN_Validation\workloads\images
		6. Removed topologies xmls and images will not be deleted. 
	example command line: modify_csv.py -add new_topology_1 new_topology_2 --remove old_topo_1 old_topo2 old_topo3