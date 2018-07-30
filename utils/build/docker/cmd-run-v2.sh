#!/bin/bash

docker run -p 9090:9090 -d --restart=unless-stopped \
    -e "http_proxy=http://proxy-chain.intel.com:911" \
    -e "https_proxy=http://proxy-chain.intel.com:912" \
    -e "ftp_proxy=http://proxy-chain.intel.com:911" \
    -e "no_proxy=localhost,127.0.0.0/8,::1" \
    --name agent-lnx-01-centos7-dt467 \
    centos7.0-dt467-j8-tca:A-1.0

docker run -p 9091:9091 -d --restart=unless-stopped \
    -e "http_proxy=http://proxy-chain.intel.com:911" \
    -e "https_proxy=http://proxy-chain.intel.com:912" \
    -e "ftp_proxy=http://proxy-chain.intel.com:911" \
    -e "no_proxy=localhost,127.0.0.0/8,::1" \
    --name agent-lnx-02-centos7-dt467 \
    centos7.0-dt467-j8-tca:B-1.0
