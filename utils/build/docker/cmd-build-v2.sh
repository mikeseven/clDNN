#!/bin/bash

docker build --build-arg TC_AGENT_USER=sys_gpudnna1 \
             --build-arg TC_AGENT_NAME=gklvbm-gpu-dnn-lnx-01 \
             --build-arg TC_SERVER_URL=https://teamcity01-igk.devtools.intel.com \
             --build-arg TC_AGENT_PROPS="my.req.linux.distro=Centos\nmy.req.linux.version=7.0.1406\nmy.req.linux.docker=True\nmy.req.can_run_command.git=True\nmy.req.can_run_command.python=True\nmy.req.can_run_command.python3=True" \
             $1 -f $1/Dockerfile.centos7.0-devtoolset4-6-7-clang-tc-agent --tag centos7.0-dt467-j8-tca:A-1.0

docker build --build-arg TC_AGENT_USER=sys_gpudnna1 \
             --build-arg TC_AGENT_NAME=gklvbm-gpu-dnn-lnx-02 \
             --build-arg TC_SERVER_URL=https://teamcity01-igk.devtools.intel.com \
             --build-arg TC_AGENT_PORT=9091 \
             --build-arg TC_AGENT_PROPS="my.req.linux.distro=Centos\nmy.req.linux.version=7.0.1406\nmy.req.linux.docker=True\nmy.req.can_run_command.git=True\nmy.req.can_run_command.python=True\nmy.req.can_run_command.python3=True" \
             $1 -f $1/Dockerfile.centos7.0-devtoolset4-6-7-clang-tc-agent --tag centos7.0-dt467-j8-tca:B-1.0