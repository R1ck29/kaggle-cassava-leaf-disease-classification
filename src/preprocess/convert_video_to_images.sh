cd `dirname $0`;
input_path=$1; #../../data/street_demo/raw/videos/;
output_path=$2; #../../data/street_demo/raw/images/;

for fp in $(find $input_path -name '*.mp4'); do
  sleep 3;
  fn=${fp##*/};
  save_fn=${fn%%.*}_%05d.png;
  save_dir=$output_path; #${fn%%.*};
  save_fp=$save_dir/$save_fn;
  echo $fp;
  echo $save_dir;
  echo $save_fn;
  echo $save_fp;
  mkdir -p  $save_dir;
  ffmpeg -i $fp -r 30 -f image2 -q:v 0 $save_fp;
done;