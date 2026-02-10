# Vercel 部署指南

## 方法一：通过 GitHub 自动部署（推荐）

1. 访问: https://vercel.com/new
2. 选择 "Import Git Repository"
3. 输入仓库地址: `https://github.com/AlbinGuo/adobe-ai`
4. 配置选项:
   - Framework Preset: `Other`
   - Build Command: `留空`
   - Output Directory: `.`
5. 点击 "Deploy"

## 方法二：Vercel CLI 部署

```bash
# 安装 Vercel CLI
npm install -g vercel

# 登录（如果还没登录）
vercel login

# 部署
npx vercel@latest --yes
```

## 部署成功

访问: **https://adobe-ai.vercel.app**

## 项目结构

```
├── index.html          # Web展示页面
├── extracted/         # 生成的矢量文件
├── *.py              # Python处理脚本
├── vercel.json       # Vercel配置
└── README.md         # 项目说明
```

## 功能

- 📄 PDF 转矢量线条
- 🎨 调整线条粗细/颜色
- 💾 导出 SVG / AI / PDF
