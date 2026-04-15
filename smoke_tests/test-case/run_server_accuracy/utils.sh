#!bin/bash


# 函数功能：统计匹配指定文件名模式的文件中包含给定字符串的文件数量
# 使用方式：count_files_with_string "文件模式" "搜索字符串"
# 返回值：包含指定字符串的文件数量（0-255），通过退出状态码返回
# 注意事项：
#   1. 使用Bash通配符进行文件名匹配（如*.json, file*.txt）
#   2. 搜索是区分大小写的
#   3. 只搜索普通文件，跳过目录和其他特殊文件
#   4. 由于Bash退出状态码限制，最多返回255个文件计数
#   5. 搜索字符串中的特殊字符会被视为字面值（非正则表达式）
count_files_with_string() {
    local pattern="$1"          # 第一个参数：文件名模式（例如 "*.json"）
    local search_string="$2"    # 第二个参数：要搜索的字符串
    local files                 # 存储匹配文件的数组
    local count=0               # 包含指定字符串的文件数量计数器

    # 启用nullglob选项：如果模式没有匹配文件，扩展为空数组
    shopt -s nullglob
    files=( $pattern )          # 将模式扩展为文件列表
    shopt -u nullglob           # 禁用nullglob选项

    # 检查是否有文件匹配模式
    if [ ${#files[@]} -eq 0 ]; then
        echo "Error: No files match pattern: $pattern" >&2
        return 0                # 返回0表示没有匹配文件
    fi

    # 遍历所有匹配的文件
    for file in "${files[@]}"; do
        # 检查是否为普通文件（跳过目录或其他类型）
        if [ ! -f "$file" ]; then
            echo "Warning: $file is not a regular file, skipping." >&2
            continue
        fi
        
        # 使用grep搜索字符串
        if grep -F -q "$search_string" "$file" 2>/dev/null; then
            ((count++))  # 增加包含字符串的文件计数
        fi
    done

    return $count  # # 返回包含字符串的文件数量
}

# 函数：检查是否存在匹配通配符模式的文件
# 返回值：0表示找到文件，非0表示未找到文件
check_files_exist() {
    local pattern="$1"
    local custom_message="${2:-}"  # 可选的自定义错误消息
    
    # 启用nullglob选项（如果无匹配文件，通配符扩展为空）
    local original_nullglob
    original_nullglob=$(shopt -p nullglob)
    shopt -s nullglob
    
    # 将匹配的文件存入数组（注意：不加引号以允许通配符扩展）
    local files=($pattern)
    
    # 恢复原来的nullglob设置
    eval "$original_nullglob"
    
    # 检查数组是否为空
    if [ ${#files[@]} -eq 0 ]; then
        if [ -n "$custom_message" ]; then
            echo "$custom_message" >&2  # 输出到标准错误
        else
            echo "Error: Cannot find any files matching pattern: $pattern" >&2
        fi
        return 1  # 返回非零退出码表示失败
    fi
    
    return 0  # 返回零退出码表示成功
}