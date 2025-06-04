echo "Please input parenet directory, will count all .bin files..."
echo "Example: bash ./count_tokens.sh /workspace/dataset/music"

PARENT_DIR=${1:-/workspace/dataset/music}
LOG_DIR=./count_token_logs/
mkdir -p $LOG_DIR

# find all .bin files
BINS=$(find $PARENT_DIR -name "*.bin" -type f)

for bin in $BINS; do
    echo Checking mmap file: $bin
    
    mmap_path=$bin

    # mmap size in human readable format (e.g. 1.2G)
    mmap_size=$(du -h $mmap_path | awk '{print $1}')
    echo "Counting largest mmap file: $mmap_path, size: $mmap_size"

    # remove PARENT_DIR, replace / with _
    subdir=$(echo $mmap_path | sed "s|$PARENT_DIR/||g" | sed 's/\//_/g')

    cmd="nohup python tools/count_mmap_token.py --mmap_path $mmap_path > $LOG_DIR/count.$subdir.log 2>&1 &"
    echo $cmd
    
    eval $cmd


    echo "Finished!"
done


