#!/bin/bash


# Copyright 2016 Intel Corporation All Rights Reserved.
#
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation
# or its suppliers or licensors. Title to the Material remains with
# Intel Corporation or its suppliers and licensors. The Material contains
# trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide
# copyright and trade secret laws and treaty provisions. No part
# of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel’s prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel
# or otherwise. Any license under such intellectual property rights must be
# express and approved by Intel in writing.


# -----------------------------------------------------------------------------

# This file is based on prepare_rpm_based.sh from neo project
# (original author: Winiarski, Michal <michal.winiarski@intel.com>).
#
# Modified for TeamCity by: Walkowiak, Marcin <marcin.walkowiak@intel.com>.

# -----------------------------------------------------------------------------


if [[ -e /etc/fedora-release ]]; then
    PKG_MGR=dnf
    DISTRO=fedora
else
    PKG_MGR=yum
    DISTRO=centos
fi

# Update package information.
${PKG_MGR} -y update
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} clean all
    ${PKG_MGR} -y swap fakesystemd systemd
    ${PKG_MGR} -y update
fi

# Ensure that basic tools are installed.
${PKG_MGR} -y install tar xz python unzip

# Compilers and standard libraries.
${PKG_MGR} -y install gcc gcc-c++ clang make glibc-static glibc-devel libstdc++-static libstdc++-devel libstdc++ libgcc libX11-devel \
                      glibc-static.i686 glibc-devel.i686 libstdc++-static.i686 libstdc++.i686 libgcc.i686 libX11-devel.i686

# Development toolset (version 4, 6 and 7).
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install centos-release-scl
    ${PKG_MGR} clean all
    ${PKG_MGR} -y install devtoolset-4-toolchain
    ${PKG_MGR} -y install devtoolset-6-toolchain
    ${PKG_MGR} -y install devtoolset-7-toolchain
else
    ${PKG_MGR} -y install dnf-plugins-core
    ${PKG_MGR} -y copr enable mlampe/devtoolset-4
    ${PKG_MGR} -y copr enable mlampe/devtoolset-6
    ${PKG_MGR} -y copr enable mlampe/devtoolset-7
fi

# Repository utilities.
${PKG_MGR} -y install git

# Build generators and build utilities.
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install epel-release
    ${PKG_MGR} -y install p7zip cmake3 python34
    /usr/sbin/alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 50
    git clone --depth 1 --branch v1.7.1 https://github.com/ninja-build/ninja.git
    pushd ninja; ./configure.py --bootstrap; cp ninja /usr/local/bin; popd; rm -r ninja
else
    ${PKG_MGR} -y install p7zip cmake ninja-build
fi

# Clang package (version 3.8.0).
curl -k "http://releases.llvm.org/3.8.0/clang+llvm-3.8.0-linux-x86_64-centos6.tar.xz" > /tmp/clang-3.8.0-centos6.tar.xz
mkdir /opt
cd /opt
tar -Jvxf /tmp/clang-3.8.0-centos6.tar.xz
chmod 755 clang+llvm-3.8.0-linux-x86_64-centos6/bin/*

# TeamCity agent elements.
if [[ $USE_ORACLE_JAVA == 'y' ]]; then
    rpm -i http://javadl.oracle.com/webapps/download/AutoDL?BundleId=207764
else
    ${PKG_MGR} -y install java-1.8.0-openjdk-headless
fi

# Cleanup dnf cache
${PKG_MGR} clean all
