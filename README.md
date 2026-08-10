# Staccato Crop Workbench Cloud

公开协作版规范切图工作台。前端使用 Vite/React，后端使用 FastAPI/Pillow。

- 浏览器选择商品文件夹并上传。
- AI扩图和无AI留白可在同一页面切换。
- 云端临时处理，完成后下载ZIP。
- API Key只保存在Render环境变量，不进入前端或GitHub。

## Render

Render Blueprint：`render.yaml`。创建服务时需在控制台填写 `OPENAI_API_KEY`（Gemini Key）。

## Netlify

构建命令：`npm run build`，发布目录：`dist`。设置构建环境变量 `VITE_API_BASE` 为Render后端HTTPS地址。
