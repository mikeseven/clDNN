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

# Add build agent user, service and set agent properties.
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

# Prepare Git/SSH configuration.
mkdir /home/$TC_AGENT_USER/.ssh
if [[ ! -e /home/$TC_AGENT_USER/.ssh/config ]]; then
    echo -e "Host *" > /home/$TC_AGENT_USER/.ssh/config
fi
if [[ $DISTRO == 'fedora' ]]; then echo -e "   HostkeyAlgorithms=+ssh-dss" >> /home/$TC_AGENT_USER/.ssh/config; fi
echo -e "   UserKnownHostsFile=/dev/null\n   StrictHostKeyChecking=no" >> /home/$TC_AGENT_USER/.ssh/config
