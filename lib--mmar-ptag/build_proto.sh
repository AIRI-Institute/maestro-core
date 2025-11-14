set -e
module_name=$(grep -oP '(?<=^module-name = ").*(?=")' pyproject.toml)
init_path="src/$module_name/__init__.py"
version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' $init_path)

echo "Building protos for $module_name==$version"
# https://github.com/grpc/grpc/issues/9575
uv run python -m grpc_tools.protoc \
  --proto_path "src/proto" \
  --python_out="src" \
  --grpc_python_out="src" \
  --pyi_out="src" \
  "src/proto/$module_name"/*.proto
echo "Build success!"
