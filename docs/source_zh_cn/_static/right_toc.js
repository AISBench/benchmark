// 等待页面加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 创建右侧目录容器
    const rightToc = document.createElement('div');
    rightToc.className = 'right-toc';

    // 创建目录标题
    const tocTitle = document.createElement('div');
    tocTitle.className = 'right-toc-title';
    tocTitle.textContent = '页面目录';
    rightToc.appendChild(tocTitle);

    // 创建目录列表
    const tocList = document.createElement('ul');
    rightToc.appendChild(tocList);

    // 创建显示/隐藏按钮
    const toggleButton = document.createElement('button');
    toggleButton.className = 'right-toc-toggle';
    toggleButton.innerHTML = '☰';
    toggleButton.title = '显示/隐藏目录';

    // 添加到页面
    document.body.appendChild(rightToc);
    document.body.appendChild(toggleButton);

    // 提取页面中的标题并构建目录
    function buildTableOfContents() {
        tocList.innerHTML = '';
        // 用于跟踪已使用的id，避免重复
        const usedIds = new Set();

        // 获取所有的 h1-h6 标题
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');

        headings.forEach(heading => {
            // 跳过一些不需要显示在目录中的标题
            if (heading.parentElement.classList.contains('wy-side-nav-search') ||
                heading.parentElement.classList.contains('wy-menu-vertical')) {
                return;
            }

            // 创建目录项
            const listItem = document.createElement('li');
            const link = document.createElement('a');

            // 设置链接文本和目标，使用更严格的字符过滤方法
            // 使用 innerText 代替 textContent，因为 innerText 会考虑样式和排版
            let cleanText = heading.innerText.trim();

            // 1. 移除所有不可见的控制字符（包括零宽字符等）
            cleanText = cleanText.replace(/[\x00-\x1F\x7F-\x9F\u00AD\u061C\u200B-\u200F\u2028-\u202F\u2060-\u206F]/g, '');

            // 2. 移除常见的标点符号和特殊字符
            cleanText = cleanText.replace(/[?？\s]+$/g, '');

            // 3. 只保留可打印的 ASCII 字符和中文字符
            cleanText = cleanText.replace(/[^\x20-\x7E\u4e00-\u9fa5]/g, '');

            // 4. 再次清理末尾的空白和标点
            cleanText = cleanText.trim().replace(/[?？\s.,;:]+$/g, '');

            // 确保标题有唯一的id
            let headingId = heading.id;
            if (!headingId) {
                // 如果没有id，基于标题文本创建一个
                headingId = generateIdFromText(cleanText);
                heading.id = headingId;
            }

            // 确保id唯一
            if (usedIds.has(headingId)) {
                let counter = 1;
                let uniqueId = `${headingId}-${counter}`;
                while (usedIds.has(uniqueId)) {
                    counter++;
                    uniqueId = `${headingId}-${counter}`;
                }
                headingId = uniqueId;
                heading.id = headingId;
            }
            usedIds.add(headingId);

            link.textContent = cleanText;
            link.href = '#' + headingId;

            // 根据标题级别设置缩进
            const level = parseInt(heading.tagName.charAt(1));
            listItem.style.paddingLeft = (level - 1) * 10 + 'px';
            listItem.style.fontSize = (16 - (level - 1) * 1) + 'px';

            // 添加点击事件
            link.addEventListener('click', function(e) {
                e.preventDefault();

                // 找到目标元素
                const targetElement = document.getElementById(headingId);
                if (targetElement) {
                    // 计算偏移量，避免被固定头部遮挡
                    const headerOffset = 80; // 可以根据实际页面调整这个值
                    const elementPosition = targetElement.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                    // 使用window.scrollTo实现平滑滚动
                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }

                // 在移动设备上点击链接后隐藏目录
                if (window.innerWidth <= 992) {
                    rightToc.classList.remove('visible');
                }
            });

            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });

        // 如果没有标题，显示提示信息
        if (tocList.children.length === 0) {
            const emptyItem = document.createElement('li');
            emptyItem.textContent = '本页面没有目录';
            emptyItem.style.color = '#777';
            tocList.appendChild(emptyItem);
        }
    }

    // 辅助函数：从文本生成有效的id
    function generateIdFromText(text) {
        // 将文本转换为小写
        let id = text.toLowerCase();
        // 将中文转换为拼音的简化版本（仅保留首字母）
        id = id.replace(/[\u4e00-\u9fa5]/g, function(char) {
            // 这里简化处理，如果需要更精确的拼音转换，可以引入专门的库
            return 'id';
        });
        // 用连字符替换空格和特殊字符
        id = id.replace(/[^a-z0-9]/g, '-');
        // 移除重复的连字符
        id = id.replace(/-+/g, '-');
        // 移除首尾的连字符
        id = id.replace(/^-+|-+$/g, '');
        // 如果id为空，返回默认值
        return id || 'section';
    }

    // 切换目录显示/隐藏
    toggleButton.addEventListener('click', function() {
        rightToc.classList.toggle('visible');
    });

    // 在桌面设备上默认显示目录
    if (window.innerWidth > 992) {
        rightToc.classList.add('visible');
    }

    // 监听窗口大小变化
    window.addEventListener('resize', function() {
        if (window.innerWidth > 992) {
            rightToc.classList.add('visible');
        } else {
            rightToc.classList.remove('visible');
        }
    });

    // 初始构建目录
    buildTableOfContents();
});