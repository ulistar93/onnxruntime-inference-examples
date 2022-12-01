source ~/torch1.8/bin/activate

out=`echo $1 | rev | cut -d'.' -f2- | rev`-infer.onnx
if [[ -f "$1" ]]
then
  echo python -m onnxruntime.quantization.preprocess --input $1 --output $out
  python -m onnxruntime.quantization.preprocess --input $1 --output $out
else
  echo no onnx file
fi
