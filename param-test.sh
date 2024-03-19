#!/bin/bash
# python3 chunk-size-test.py
python3 bot-object.py
# tmux new-session -d -s chunk-size-test ;
# tmux send-keys -t chunk-size-test "conda activate chunk-size-test" C-m
# tmux send-keys -t chunk-size-test "streamlit run chunk-size-test.py --server.port 8511" C-m
