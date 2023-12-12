#!/bin/bash

for ((i=100; i<2000; i=i+200));do
  echo ""
  echo "k_block_factor $i m_block_factor [5,200,20]"
  echo "spmm (ms) | blocking (ms) "
for ((j=205; j<400; j=j+20));do
  K_BLOCKING_PARAM=$i M_BLOCKING_PARAM=$j python amazon_vendor.py

done
done
