#!/bin/bash

docker run -p 9090:9090 -d --restart=unless-stopped centos7.0-dt4-j8-tca
