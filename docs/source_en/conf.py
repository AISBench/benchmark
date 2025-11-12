import os
import sys
# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath('.'))

# 项目信息
sys.path.insert(0, os.path.abspath('.'))
project = 'AISBench Benchmark Tool'  # 这里设置你的 project name
copyright = '2025, AISBench AI System Performance Evaluation Benchmark Committee'  # 修改为实际的版权信息
author = 'AISBench AI System Performance Evaluation Benchmark Committee'  # 修改为实际的作者名称

html_theme = 'sphinx_rtd_theme'

# 添加主题相关配置，优化响应式布局
html_theme_options = {
    # 侧边栏自动折叠（在移动设备上会更友好）
    'collapse_navigation': True,
    # 隐藏导航栏中当前页面的父级
    'navigation_depth': 2,  # 将4改为1，只显示到文档层级，不显示文档内部的标题结构
    # 是否在顶部显示"返回顶部"按钮
    'sticky_navigation': True,
    # 是否在侧边栏显示搜索框
    'includehidden': True,
}

language = 'en'

# 3. 添加必要扩展（支持 Markdown、代码高亮、目录生成等）
extensions = [
    'sphinx.ext.autodoc',    # 自动生成代码文档
    'sphinx.ext.napoleon',   # 支持 Google/Numpy 风格的注释
    'myst_parser',           # 支持 Markdown 文件
    'sphinx.ext.todo',       # 支持 TODO 标记
    'sphinx.ext.intersphinx', # 支持跨文档引用
    'sphinx.ext.imgconverter', # 支持图片格式转换
    'sphinx.ext.mathjax',    # 支持数学公式
    'sphinx.ext.viewcode',   # 查看代码源文件
]

# 4. 若使用 Markdown，需指定源文件后缀
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# 添加源文件编码配置
source_encoding = 'utf-8'

# 5. 配置文档版本（与 Gitee 仓库分支对应，如 v1.0）
version = '1.0'
release = '1.0'

# 6. 添加 myst_parser 配置
myst_enable_extensions = [
    'linkify',
    'html_image',
    'smartquotes',
    'dollarmath',            # 支持 $ 分隔的数学公式
    'html_admonition',       # 支持 HTML 警告框
    'replacements',          # 支持文本替换
]

# （可选）配置 Mermaid 输出格式
myst_mermaid_output_format = 'svg'  # 或 'png'

# 添加以下配置来解决Pygments无法识别mermaid的问题
# 忽略Pygments无法识别mermaid的警告
suppress_warnings = [
    'myst.xref_missing',     # 忽略交叉引用丢失的警告
    'myst.header',           # 忽略标题格式警告
    'misc.highlighting_failure', # 忽略语法高亮失败的警告
]

# 添加交叉引用支持
default_role = 'any'
myst_all_links_external = False
myst_highlight_code_blocks = True
myst_heading_anchors = 3  # 为标题生成锚点的级别
myst_footnote_transition = True
myst_dmath_double_inline = True

# 忽略特定的警告
suppress_warnings = [
    'myst.xref_missing',     # 忽略交叉引用丢失的警告
    'myst.header',           # 忽略标题格式警告
    'misc.highlighting_failure', # 忽略语法高亮失败的警告
]

# 8. 添加静态文件路径配置
html_static_path = ['_static']

# 添加页面内目录配置
# 控制是否在页面侧边显示目录
html_sidebars = {
    '**': [
        'localtoc.html',  # 当前页面的目录
        'relations.html', # 上一篇/下一篇导航
        'searchbox.html', # 搜索框
    ]
}

# 9. 解决中文路径问题
# 确保路径正确编码
def setup(app):
    # 允许使用中文文件名
    app.config._raw_config['html_file_suffix'] = '.html'
    # 添加自定义 CSS 文件支持，用于进一步优化响应式布局
    app.add_css_file('custom.css')
    # 添加mermaid.js库和初始化脚本
    app.add_js_file('https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.min.js')
    app.add_js_file('mermaid_init.js')
    # 添加仓库链接脚本
    app.add_js_file('repo_links.js')
    # 添加自定义 JavaScript 以支持右侧目录
    app.add_js_file('right_toc.js')