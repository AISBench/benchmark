// 等待DOM加载完成后初始化mermaid
 document.addEventListener('DOMContentLoaded', function() {
     // 配置mermaid
     mermaid.initialize({
         startOnLoad: true,
         theme: 'default',
         flowchart: {
             useMaxWidth: true,
             htmlLabels: true
         }
     });
     
     // 查找所有带有mermaid标记的<pre>或<code>元素并手动渲染
     const mermaidBlocks = document.querySelectorAll('pre code.language-mermaid, div[class*="mermaid"]');
     mermaidBlocks.forEach((block, index) => {
         try {
             // 获取mermaid图表代码
             let code = block.textContent || block.innerText;
             
             // 创建一个新的div元素来容纳渲染后的图表
             const newDiv = document.createElement('div');
             newDiv.className = 'mermaid';
             newDiv.id = 'mermaid-diagram-' + index;
             newDiv.textContent = code;
             
             // 替换原元素或在其旁边插入新元素
             block.parentNode.insertBefore(newDiv, block);
             
             // 隐藏原始代码块
             block.style.display = 'none';
             
         } catch (error) {
             console.error('Error rendering mermaid diagram:', error);
         }
     });
     
     // 确保所有mermaid图表都被渲染
     mermaid.run();
 });