#!/bin/bash

# 脚本路径
SCRIPT1="/home/DM24/workspace/Time_Series_Forecasting/HARP_FDN/scripts/HARP_FDN_optimized/exchange.sh"
SCRIPT2="/home/DM24/workspace/Time_Series_Forecasting/HARP_FDN/scripts/HARP_FDN_optimized/weather.sh"
SCRIPT3="/home/DM24/workspace/Time_Series_Forecasting/HARP_FDN/scripts/HARP_FDN_optimized/ECL.sh"

# 会话名
SESSION_NAME="task_run"

# 不存在则新建后台会话
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux new-session -d -s $SESSION_NAME
    echo "已创建 tmux 会话: $SESSION_NAME"
fi

# 新建窗口（不指定编号，自动顺延）
tmux new-window -t $SESSION_NAME -n "run_task"
echo "已在会话 $SESSION_NAME 中新建窗口"

# 发送命令串行执行脚本
tmux send-keys -t $SESSION_NAME: "
echo ===== 开始执行 exchange.sh =====
bash '$SCRIPT1'
if [ \$? -ne 0 ]; then
    echo ERROR: exchange.sh 执行失败
    exit 1
fi

echo -e \\n===== 开始执行 weather.sh =====
bash '$SCRIPT2'
if [ \$? -ne 0 ]; then
    echo ERROR: weather.sh 执行失败
    exit 1
fi

echo -e \\n===== 开始执行 ECL.sh =====
bash '$SCRIPT3'
if [ \$? -ne 0 ]; then
    echo ERROR: ECL.sh 执行失败
    exit 1
fi

echo -e \\n✅ 所有脚本执行完毕
" C-m

echo "任务已在 tmux 后台启动"
echo "查看日志：tmux a  $SESSION_NAME"