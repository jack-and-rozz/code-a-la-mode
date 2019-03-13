

usage() {
    echo "Usage:$0 model_dir epoch"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi
model_dir=$1
epoch=$2

script_dir=$(cd $(dirname $0); pwd)
#game_id=$(openssl rand -base64 12 | fold -w 10 | head -1)

game_id=$(export LC_ALL=C; cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 10 | head -n 1 | sort | uniq)
result_path=$model_dir/games/$epoch/$game_id.score.txt
detail_path=$model_dir/games/$epoch/$game_id.detail.json
echo "java -jar $script_dir/judge/code-a-la-mode.jar $agent1 $agent2 $agent3 $result_path $detail_path"

java -jar \
     $script_dir/judge/code-a-la-mode.jar \
     python\ play.py\ -mrp\ $model_dir\ -e\ $epoch\ -eps\ 0.2 \
     python\ play.py\ -mrp\ $model_dir\ -e\ $epoch\ -eps\ 0.2 \
     others_agents/oda1/agent \
     $result_path \
     $detail_path 


     #others_agents/oda1/agent  \
