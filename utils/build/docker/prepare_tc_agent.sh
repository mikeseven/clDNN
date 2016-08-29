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
${PKG_MGR} -y install tar xz python p7zip unzip

# Compilers and standard libraries.
${PKG_MGR} -y install gcc gcc-c++ clang make glibc-static glibc-devel libstdc++-static libstdc++-devel libstdc++ libgcc libX11-devel \
                      glibc-static.i686 glibc-devel.i686 libstdc++-static.i686 libstdc++.i686 libgcc.i686 libX11-devel.i686

# Development toolset (version 4).
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install centos-release-scl
    ${PKG_MGR} -y install devtoolset-4-toolchain
else
    ${PKG_MGR} -y install dnf-plugins-core
    ${PKG_MGR} -y copr enable mlampe/devtoolset-4
fi

# Repository utilities.
${PKG_MGR} -y install git

# Build generators and build utilities.
if [[ $DISTRO == 'centos' ]]; then
    ${PKG_MGR} -y install epel-release
    ${PKG_MGR} -y install cmake3 python34
    /usr/sbin/alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 50
    git clone --depth 1 --branch v1.7.1 https://github.com/ninja-build/ninja.git
    pushd ninja; ./configure.py --bootstrap; cp ninja /usr/local/bin; popd; rm -r ninja
else
    ${PKG_MGR} -y install cmake ninja-build
fi

# Formatting tools (optional).
if [ ! -z ${CF_BIN_URL:+x} ]; then
    curl -k ${CF_BIN_URL}/clang-format -o /usr/local/bin/clang-format
    chmod 755 /usr/local/bin/clang-format
fi

# TeamCity agent elements.
if [[ $USE_ORACLE_JAVA == 'y' ]]; then
    rpm -i http://javadl.oracle.com/webapps/download/AutoDL?BundleId=207764
else
    ${PKG_MGR} -y install java-1.8.0-openjdk-headless
fi

useradd -m -g users $TC_AGENT_USER
curl -k $TC_SERVER_URL/update/buildAgent.zip > /home/$TC_AGENT_USER/buildAgent.zip
unzip /home/$TC_AGENT_USER/buildAgent.zip -d /home/$TC_AGENT_USER/buildAgent
cat /home/$TC_AGENT_USER/buildAgent/conf/buildAgent.dist.properties | sed '
/^\s*serverUrl\s*=/ { c\
serverUrl='"$TC_SERVER_URL"'
}
/^\s*name\s*=/ { c\
name='"$TC_AGENT_NAME"'
}
/^\s*ownPort\s*=\s*[0-9]*/ { c\
ownPort='"$TC_AGENT_PORT"'
}' > /home/$TC_AGENT_USER/buildAgent/conf/buildAgent.properties
chown $TC_AGENT_USER:users /home/$TC_AGENT_USER/buildAgent.zip
chown -R $TC_AGENT_USER:users /home/$TC_AGENT_USER/buildAgent
chmod 755 /home/$TC_AGENT_USER/buildAgent/bin/*.sh


mkdir /home/$TC_AGENT_USER/.ssh
if [[ $DISTRO == 'fedora' ]]; then echo -e "HostkeyAlgorithms=+ssh-dss" >> /home/$TC_AGENT_USER/.ssh/config; fi
echo -e "UserKnownHostsFile=/dev/null\nStrictHostKeyChecking=no" >> /home/$TC_AGENT_USER/.ssh/config

# Cleanup dnf cache
${PKG_MGR} clean all
