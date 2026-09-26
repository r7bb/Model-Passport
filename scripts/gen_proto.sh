#!/usr/bin/env bash
# Regenerate the gRPC code for the control plane from controlplane/proto.
#   Go:     controlplane/gen/...          (protoc, protoc-gen-go, protoc-gen-go-grpc)
#   Python: src/model_passport/platform/rpc/   (grpcio-tools)
set -euo pipefail
cd "$(dirname "$0")/.."
proto=controlplane/proto
file=mp/controlplane/v1/controlplane.proto

(cd controlplane && protoc -I proto --go_out=gen --go_opt=paths=source_relative \
  --go-grpc_out=gen --go-grpc_opt=paths=source_relative "$file")

out=src/model_passport/platform/rpc
mkdir -p "$out"
python -m grpc_tools.protoc -I "$proto/mp/controlplane/v1" --python_out="$out" --pyi_out="$out" \
  --grpc_python_out="$out" "$proto/$file"
# grpcio-tools writes an absolute import; make it relative to the package.
sed -i.bak 's/^import controlplane_pb2 as/from . import controlplane_pb2 as/' "$out/controlplane_pb2_grpc.py"
rm -f "$out/controlplane_pb2_grpc.py.bak"
printf '"""Generated gRPC code for the control plane (scripts/gen_proto.sh); do not edit."""\n' \
  > "$out/__init__.py"
echo "generated Go and Python code for $file"
