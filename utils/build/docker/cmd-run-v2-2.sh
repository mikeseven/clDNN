#!/bin/bash

docker run -p 9092:9092 -d --restart=unless-stopped \
    -e "http_proxy=http://proxy-chain.intel.com:911" \
    -e "https_proxy=http://proxy-chain.intel.com:912" \
    -e "ftp_proxy=http://proxy-chain.intel.com:911" \
    -e "no_proxy=localhost,127.0.0.0/8,::1" \
    --name agent-lnx-03-centos7-dt467 \
    centos7.0-dt467-j8-tca:C-1.0

docker run -p 9093:9093 -d --restart=unless-stopped \
    -e "http_proxy=http://proxy-chain.intel.com:911" \
    -e "https_proxy=http://proxy-chain.intel.com:912" \
    -e "ftp_proxy=http://proxy-chain.intel.com:911" \
    -e "no_proxy=localhost,127.0.0.0/8,::1" \
    --name agent-lnx-04-centos7-dt467 \
    centos7.0-dt467-j8-tca:D-1.0
