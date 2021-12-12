#!/bin/bash

pip3 install --upgrade gdown
mkdir engine
cd engine
mkdir onnx
mkdir trt_poly

gdown --id 1uJWUz1qLggPmluFSGstIPg7h0IySgwwc --folder

mv *.onnx onnx/
mv *.engine trt_poly/

cd onnx
mkdir nano
mkdir small
mkdir tiny
mkdir reg

mv nano.onnx nano/cwpssb_dyn.onnx
mv small.onnx small/cwpssb_dyn.onnx
mv tiny.onnx tiny/cwpssb_dyn.onnx
mv reg.onnx reg/cwpssb_dyn.onnx

cd ../trt_poly
mkdir nano
mkdir small
mkdir tiny
mkdir reg

mv nano.engine nano/cwpssb_dyn.engine
mv small.engine small/cwpssb_dyn.engine
mv tiny.engine tiny/cwpssb_dyn.engine
mv reg.engine reg/cwpssb_dyn.engine

cd ../..
