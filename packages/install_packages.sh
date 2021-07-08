#!/bin/bash
echo "### PACKAGE INSTALLATION ###"
for repo in ./*
  do
    if [ "$repo" != "./${0}" ];then  # do not consider the .sh
      if [ -e "${repo}/*.egg-info" ]; then
        rm -i -rf "${repo}/*.egg-info"  2> /dev/null  # remove previous installation repo
      fi
      pip3 install --user -e ./$repo
    fi
  done