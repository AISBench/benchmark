#!/usr/bin/env bash

func() {
    # 定义颜色变量
    YELLOW='\033[0;33m'
    GREEN='\033[0;32m'
    BLUE='\033[0;36m'
    MAGENTA='\033[0;35m'
    ORANGE='\033[0;33m'
    RED='\033[0;31m'
    BOLD='\033[1m'
    RESET='\033[0m'

    echo
    echo -e "${BOLD}${YELLOW}使用:${RESET}"
    echo -e "  ${GREEN}run.sh [选项]${RESET}"
    echo

    echo -e "${BOLD}${YELLOW}选项:${RESET}"
    echo -e "  ${BOLD}${GREEN}-p${RESET} ${BOLD}TEST_CASE_PATH${RESET}    测试用例的根目录"
    echo -e "  ${BOLD}${GREEN}-t${RESET} ${BOLD}TEST_CASE_TAGS${RESET}    选择待测用例的标签（标签用空格分隔）"
    echo -e "                        ${BLUE}输入格式：${RESET}"
    echo -e "                          - 只要被包含在用例的 ${MAGENTA}case_type${RESET}、${MAGENTA}case_group${RESET} 或 ${MAGENTA}case_name${RESET} 中即被选中"
    echo -e "                          - ${RED}示例:${RESET} ${GREEN}-t group1 group2 casename anytags${RESET}"
    echo
    echo -e "  ${BOLD}${GREEN}-l${RESET} ${BOLD}TEST_CASE_LABEL${RESET}   选择待测用例的特殊值（键值对用空格分隔）"
    echo -e "                        ${BLUE}键值格式：${RESET}"
    echo -e "                          - 键值用等号分隔（${BOLD}key${RESET}=${BOLD}value${RESET}）"
    echo -e "                          - 值为可迭代元素时，内部元素用逗号分隔"
    echo -e "                        ${RED}示例:${RESET}"
    echo -e "                          ${GREEN}-l rank_size=3 case_group=group1,group2 script=start:run.sh,end:clean.sh${RESET}"
    echo
    echo -e "  ${BOLD}${GREEN}-d${RESET}                    ${BLUE}启用 debug 模式${RESET}"
    echo -e "  ${BOLD}${GREEN}-r${RESET}                    ${BLUE}重跑错误用例${RESET}"
    echo

    echo -e "${BOLD}${YELLOW}详细说明:${RESET}"
    echo -e "  ${BOLD}${GREEN}-t${RESET} 选项："
    echo -e "    选中所有包含指定标签的用例（${RED}逻辑 OR 关系${RESET}）"
    echo -e "    标签可出现在 ${MAGENTA}case_type${RESET}、${MAGENTA}case_group${RESET} 或 ${MAGENTA}case_name${RESET} 中"
    echo
    echo -e "  ${BOLD}${GREEN}-l${RESET} 选项："
    echo -e "    仅选中满足${ORANGE}所有指定键值对${RESET}的用例（${RED}逻辑 AND 关系${RESET}）"
    echo -e "    当值为可迭代元素时（如列表），需要${RED}全等${RESET}才满足条件"
    echo

    exit 0
}

export PROJECT_PATH=$(realpath `dirname $0`/..)                                      # 工程路径
export PROJECT_RESOURCE_PATH=$PROJECT_PATH/resource/                                 # 资源路径
export PROJECT_TEST_CASE_PATH=$PROJECT_PATH/test-case/                               # 用例路径
export PROJECT_OUTPUT_PATH=$PROJECT_PATH/output/                                     # 产物路径
export PROJECT_TEST_PY_SCRIPT=$PROJECT_PATH/framework/run.py                         # 运行python脚本
: ${PROJECT_WORKSPACE_PATH:="${PROJECT_PATH}/RunWorkspace/$(date '+%Y-%m-%d_%H-%M-%S')"}; export PROJECT_WORKSPACE_PATH # 运行日志路径

TEST_CASE_PATH=$PROJECT_TEST_CASE_PATH
TEST_CASE_TAGS=()
TEST_CASE_LABEL=()
RERUN_MODE=0
DEV_MODE=0
while getopts 'p:t:l:rhd' OPT; do
    case $OPT in
        p) TEST_CASE_PATH=`realpath "$OPTARG"`;;
		t)
           TEST_CASE_TAGS+=("$OPTARG")  # 首先添加OPTARG

           # 处理后续非选项参数
           while [[ "${@:$OPTIND:1}" != -* ]] && [[ $OPTIND -le $# ]]; do
               TEST_CASE_TAGS+=("${@:$OPTIND:1}")
               ((OPTIND++))
           done
           TEST_CASE_TAGS=${TEST_CASE_TAGS[@]}
           ;;
        l)
           TEST_CASE_LABEL+=("$OPTARG")  # 首先添加OPTARG

           # 处理后续非选项参数
           while [[ "${@:$OPTIND:1}" != -* ]] && [[ $OPTIND -le $# ]]; do
               TEST_CASE_LABEL+=("${@:$OPTIND:1}")
               ((OPTIND++))
           done
           TEST_CASE_LABEL=${TEST_CASE_LABEL[@]}
           ;;
        r) RERUN_MODE=1;;
        d) DEV_MODE=1;;
        h) func;;
    esac
done

# 设置环境变量
PYTHONPATH=$PROJECT_PATH:$PYTHONPATH
WORKSPACE_PATH=$PROJECT_WORKSPACE_PATH
python $PROJECT_TEST_PY_SCRIPT \
          --project_path $PROJECT_PATH \
          --test-case $TEST_CASE_PATH \
          --case-tags $TEST_CASE_TAGS \
          --case-label $TEST_CASE_LABEL \
          --rerun-mode $RERUN_MODE \
          --dev-mode $DEV_MODE \
          --workspace-path $PROJECT_WORKSPACE_PATH \
          --resource-path $PROJECT_RESOURCE_PATH