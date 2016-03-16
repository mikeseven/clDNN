#!/bin/bash
LIBS="glog-devel boost-devel gflags-devel protobuf-devel openblas-devel hdf5-devel lmdb-devel"
echo Libraries to be installed: $LIBS
sudo yum install -y $LIBS
