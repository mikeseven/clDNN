#!/bin/bash

docker run -p 9090:9090 -d --restart=unless-stopped \
    -e "http_proxy=http://proxy-chain.intel.com:911" \
    -e "https_proxy=http://proxy-chain.intel.com:912" \
    -e "ftp_proxy=http://proxy-chain.intel.com:911" \
    -e "no_proxy=localhost,127.0.0.0/8,::1" \
    centos7.0-dt478-j8-tca
