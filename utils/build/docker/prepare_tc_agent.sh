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

# Ensure that basic tools are installed.
${PKG_MGR} -y install tar xz python p7zip

# Compilers and standard libraries.
${PKG_MGR} -y install gcc gcc-c++ clang make glibc-static glibc-devel libstdc++-static libstdc++-devel libstdc++ libgcc libX11-devel \
                      glibc-static.i686 glibc-devel.i686 libstdc++-static.i686 libstdc++.i686 libgcc.i686 libX11-devel.i686

# Development toolset (version 4).
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install centos-release-scl
    ${PKG_MGR} -y install devtoolset-4
else
    ${PKG_MGR} -y install ${PKG_MGR}-plugins-core
    ${PKG_MGR} -y copr enable mlampe/devtoolset-4
fi
scl enable devtoolset-4 bash

# Repository utilities.
${PKG_MGR} -y install git

# Build generators and build utilities.
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install epel-release
    ${PKG_MGR} -y install cmake3
    /usr/sbin/alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 50
    git clone --depth 1 --branch v1.7.1 https://github.com/ninja-build/ninja.git
    pushd ninja; ./configure.py --bootstrap; cp ninja /usr/local/bin; popd; rm -r ninja
else
    ${PKG_MGR} -y install cmake ninja-build
fi

# Formatting tools.
if [ ! -z ${BINARY_URL:+x} ]; then
    curl -k ${BINARY_URL}/clang-format -o /usr/local/bin/clang-format
    chmod 755 /usr/local/bin/clang-format
fi

# Crap for Jenkins
${PKG_MGR} -y install java-1.8.0-openjdk-headless
useradd -m -g users jenkins
curl http://repo.jenkins-ci.org/releases/org/jenkins-ci/plugins/swarm-client/2.1/swarm-client-2.1-jar-with-dependencies.jar > /home/jenkins/swarm-client.jar
chown jenkins:users /home/jenkins/swarm-client.jar
mkdir /home/jenkins/.ssh
if [[ $DISTRO == 'fedora' ]]; then echo -e "HostkeyAlgorithms=+ssh-dss" >> /home/jenkins/.ssh/config; fi
echo -e "UserKnownHostsFile=/dev/null\nStrictHostKeyChecking=no" >> /home/jenkins/.ssh/config

# Cleanup dnf cache
${PKG_MGR} clean all
