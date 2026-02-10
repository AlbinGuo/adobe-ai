# Adobe AI - PDF Vector Line Converter

将 PDF 中的图片转换为平滑的矢量线条，支持 Adobe Illustrator 和印刷使用。

## 功能特点

- 📄 PDF 页面提取
- ✏️ 平滑线条描边
- 🎨 可调节线条粗细
- 🌈 多种线条颜色
- 💾 导出 SVG / AI / PDF

## 在线演示

部署后访问: `https://你的域名.vercel.app`

## 本地运行

```bash
# 使用 Python
python -m http.server 8080

# 或使用 Node.js
npm run dev
```

访问: http://localhost:8080

## 文件说明

- `index.html` - Web 展示页面
- `fast_smooth.py` - 快速平滑线条处理脚本
- `extracted/` - 生成的矢量文件

## 导出格式

| 格式 | 说明 | 用途 |
|------|------|------|
| SVG | 矢量图 | 网页、打印 |
| AI | Adobe Illustrator | AI 编辑 |
| PDF | 矢量 PDF | 印刷 |

## 技术栈

- Python (OpenCV) - 图像处理
- HTML/CSS/JavaScript - 前端展示
- Vercel - 静态部署

## License

MIT
