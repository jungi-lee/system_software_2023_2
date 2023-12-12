#!/bin/bash

for ((i=50; i<2000; i=i+50));do
  echo ""
  echo "k_block_factor $i m_block_factor [5, 10, 15, ... 145]"
  echo "spmm (ms) | blocking (ms) "
for ((j=5; j<150; j=j+5));do
  export K_BLOCKING_PARAM=$i
  export M_BLOCKING_PARAM=$j
  python reddit_vendor.py

done
done
