SDK for weightslab that includes the operations on weights tensors.
cd .
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./weightslab/proto/experiment_service.proto

