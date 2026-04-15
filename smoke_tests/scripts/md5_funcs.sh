#!/bin/bash

# MD5处理函数库
save_md5_to_file() {
    local dir="$1"
    local output_file="$2"
    local base_dir="AISBench_Smoke"

    repo_name=$(basename `git rev-parse --show-toplevel` 2>/dev/null)

    if [ -z "$repo_name" ]; then
        base_dir=$repo_name
    fi

    if [[ "$dir" == *$base_dir* ]]; then
        display_dir="$base_dir${dir#*/$base_dir}"
    else
        display_dir="$dir"
    fi
    
    # 清空并写入头信息
    echo "# Auto-generated MD5 records (directory: ${display_dir})" > "$output_file"
    
    # 创建一个临时文件用于存储所有MD5记录
    temp_file=$(mktemp)
    
    # 遍历所有文件并保存相对路径和MD5到临时文件
    while IFS= read -r -d $'\0' file; do
        # 获取相对于目标目录的路径
        rel_path="${file#$dir}"
        md5_val=$(md5sum "$file" | awk '{print $1}')
        echo "${rel_path}:${md5_val}" >> "$temp_file"
    done < <(find "$dir" -type f ! -name "README.md" -print0)
    
    # 对临时文件按字母顺序排序并追加到输出文件
    sort "$temp_file" >> "$output_file"
    
    # 删除临时文件
    rm "$temp_file"
}

load_md5_from_file() {
    local input_file="$1"
    local -n md5_dict="$2"  # 传入数组引用
    
    md5_dict=()
    while IFS= read -r line; do
        # 跳过注释行和空行
        [[ "$line" == \#* || -z "$line" ]] && continue
        
        # 解析行数据 rel_path:md5value
        rel_path="${line%%:*}"
        md5_val="${line#*:}"
        md5_dict["$rel_path"]="$md5_val"
    done < "$input_file"
}

verify_md5() {
    local dir="$1"
    local -n ref_dict="$2"  # 传入数组引用
    local context="$3"      # 上下文信息
    
    errors_found=0
    # 遍历字典中的所有文件
    for rel_path in "${!ref_dict[@]}"; do
        file_path="${dir}/${rel_path}"
        
        # 检查文件是否存在
        if [ ! -f "$file_path" ]; then
            echo "  [${context}] File missing: ${file_path}"
            errors_found=1
            continue
        fi
        
        # 计算当前MD5
        current_md5=$(md5sum "$file_path" | awk '{print $1}')
        
        # 比较MD5值
        if [ "${ref_dict[$rel_path]}" != "$current_md5" ]; then
            echo "  [${context}] MD5 mismatch: ${rel_path}"
            echo "    Expected: ${ref_dict[$rel_path]}"
            echo "    Actual  : ${current_md5}"
            errors_found=1
        fi
    done
    
    return $errors_found
}