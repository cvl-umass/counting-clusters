DATASET=${1:-"MacaqueFaces"}
MODEL=${2:-"L-384"} 
PRETRAINING=${3:-"megad"} 
RUNS=${5:-100}


python src/run_nested_is.py --dataset $DATASET --model $MODEL --pretraining $PRETRAINING --runs $RUNS 

