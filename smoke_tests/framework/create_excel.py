import os
import xlwt
from utils import print_json, get_max_deep, find_value_in_same_level
from typing import List, Dict, Any, Tuple, Optional, Union

def write_result_data(sheet, title_style, case_result, case_type, case_names, get_color_style):
    if case_type == "benchmark":
        # 表头数据填充
        sheet.write(0, 0, "用例名称", title_style)
        sheet.write(0, 1, "结果", title_style)
        max_deep = get_max_deep(case_result, case_names, match_key="case_name", target_key="metric_key")
        start_col = 2
        for i in range(max_deep):
            sheet.write(0, start_col + i, "用例场景第 {} 层".format(i+1), title_style)

        # 表格数据填充
        curr_row = 1
        for case_name in case_names:
            result = find_value_in_same_level(("case_name", case_name), case_result)
            metric_result = result.get("metric_result")

            metric_path = result.get("metric_key")
            sheet.write(curr_row, 0, case_name, get_color_style())
            color_index = 10 if not result.get("success") else 11
            content = "Failed" if not result.get("success") else "Success"
            sheet.write(curr_row, 1, result.get("success"), get_color_style(color_index))

            start_col = 2
            metric_path = result.get("metric_key")
            for i in range(max_deep):
                sheet.write(curr_row, start_col + i,
                            "{}".format(metric_path[i] if i < len(metric_path) else None),
                            get_color_style())
            curr_row += 1


def create_excel(case_result, excel_name, workspace_path, case_type, case_names):
    try:
        work_book = xlwt.Workbook(encoding='utf-8')
        sheet = work_book.add_sheet('benchmark_result')

        # 设置列宽和行高
        sheet.row(0).height_mismatch = True
        sheet.row(0).height = 20 * 30

        for i in range(10):
            sheet.col(i).width = 256 * 15

        title_style = xlwt.XFStyle()
        title_pattern = xlwt.Pattern()
        title_font = xlwt.Font()
        alignment = xlwt.Alignment()
        borders = xlwt.Borders()

        # 设置表头背景填充
        title_pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        title_pattern.pattern_fore_colour = 41

        # 设置表头字体
        title_font.bold = True

        # 设置表头对齐方式
        alignment.horz = 0x02
        alignment.vert = 0x01

        # 设置单元格格式
        borders.top = 1
        borders.bottom = 1
        borders.left = 1
        borders.right = 1

        title_style.pattern = title_pattern
        title_style.font = title_font
        title_style.alignment = alignment
        title_style.borders = borders

        # 添加自定义颜色管理
        custom_colors = {}  # 存储十六进制颜色到索引的映射
        next_color_index = 64  # xlwt中的自定义颜色起始索引

        # 设置颜色闭包
        def get_color_style(color_name:Union[str,int] = "white"):
            """创建基于calculated_xf_style并添加指定颜色的新样式"""
            nonlocal next_color_index, custom_colors

            # 复制基本样式
            calculated_xf_style = xlwt.XFStyle()
            # 设置计算列属性
            calculated_xf_style.alignment = alignment
            calculated_xf_style.borders = borders
            # 设置单元格格式
            calculated_xf_style.num_format_str = '0.00%'

            # 设置单元格颜色
            pattern = xlwt.Pattern()
            pattern.pattern = xlwt.Pattern.SOLID_PATTERN
            pattern.pattern_fore_colour = xlwt.Style.colour_map[color_name] if isinstance(color_name, str) else color_name

            calculated_xf_style.pattern = pattern
            return calculated_xf_style

        # excel表头
        if case_type == "benchmark":
            # 数据写入execl
            write_result_data(sheet, title_style, case_result, case_type, case_names, get_color_style)
            excel_path = os.path.abspath(os.path.join(workspace_path, excel_name))
            work_book.save(excel_path)
        print("create execl success")
    except Exception as e:
        print("create execl fail: {}".format(e))



def get_nearest_color(hex_color: str) -> int:
    """获取最接近的预定义颜色索引 """
    # 如果没有可用的自定义索引，使用最接近的预定义颜色
    predefined = {
        "#FF0000": "red", "#00FF00": "green", "#0000FF": "blue",
        "#FFFF00": "yellow", "#FF00FF": "magenta", "#00FFFF": "cyan",
        "#FFFFFF": "white", "#000000": "black", "#C0C0C0": "gray",
        "#808080": "dark_gray", "#008000": "dark_green", "#800000": "dark_red",
        "#92D050": "light_green"  # 特别处理这个颜色
    }

    # 特别处理您提到的颜色
    if hex_color == "#92D050":
        return xlwt.Style.colour_map["light_green"]

    # 查找最接近的预定义颜色
    color_name = predefined.get(
        hex_color,
        predefined.get(
            hex_color.upper(),
            "white"  # 默认值
        )
    )

    return xlwt.Style.colour_map.get(color_name, xlwt.Style.colour_map["white"])