# traffic-violation-detection

## Installation
* Clone or download the repo
* Install the required packages with pip (Note: For CPU ONNX inference, comment the `onnxruntime-gpu` and uncomment the `onnxruntime`)
```
pip install -r requirements.txt
```
* For TensorRT inference, install the additional requirements
```
pip install -r requirements.txt
```
(Note: Make sure TensorRT is installed on your system. [Installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).)
* Run with
```
./run_voila.sh
```
or
```
run_voila.bat
```
* Open the link in a web browser (http://localhost:8866)
