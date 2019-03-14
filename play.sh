#!/bin/bash

usage() {
    echo "Usage:$0 model_dir"
    exit 1
}
if [ $# -lt 1 ];then
    usage;
fi
model_dir=$1
script_dir=$(cd $(dirname $0); pwd)
games_dir=$model_dir/games
parameters_dir=$model_dir/parameters
others_agent=others_agents/oda1/agent

play(){
    epoch=$1
    if [ ! -e $games_dir/$epoch ]; then
	mkdir -p $games_dir/$epoch
    fi
    game_id=$(export LC_ALL=C; cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 10 | head -n 1 | sort | uniq)
    #result_path=$games_dir/$epoch/$game_id.score.txt
    #detail_path=$games_dir/$epoch/$game_id.detail.json
    result_path=$games_dir/$epoch/$game_id.score
    detail_path=$games_dir/$epoch/$game_id.json

    java -jar \
	 $script_dir/judge/code-a-la-mode.jar \
	 python\ play.py\ -mrp\ $model_dir\ -e\ $epoch\ -eps\ 0.2 \
	 python\ play.py\ -mrp\ $model_dir\ -e\ $epoch\ -eps\ 0.2 \
	 $others_agent \
	 $result_path \
	 $detail_path 2>/dev/null > /dev/null
}

play 000
exit 1

runx() {
  for ((n=0;n<$1;n++))
    do ${*:2} &
  done
  wait
}
play_multi(){
    epoch=$1
    #n_loop=$(( $n_play_per_epoch / $n_parallel )) 
    n_parallel=5
    total=100
    for i in {1..20}; do
	printf "<Epoch $latest_epoch> sampling: (%s/%s)\r" $(( i * $n_parallel )) $total
	runx $n_parallel play $epoch 
    done;
}

_latest_epoch='none'
while true; do
    epochs=$(ls $parameters_dir)
    for latest_epoch in ${epochs[@]}; do 
	: 
    done
    if [ $_latest_epoch != $latest_epoch ]; then
	play_multi $latest_epoch
    fi
    _latest_epoch=$latest_epoch
    sleep 1s
done;
#play 000
