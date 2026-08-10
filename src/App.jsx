import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Check, CircleHelp, Crop, FolderOpen, GripVertical, ImagePlus, Maximize, Play, RotateCcw, Settings, Sparkles, Undo2, X } from 'lucide-react';

const DEFAULT_ROOT = '/Users/coralee/Desktop/EIL15DG6';
const API_BASE = import.meta.env.VITE_API_BASE || '';
const PUBLIC_MODE = import.meta.env.VITE_PUBLIC_MODE === 'true';
const AI_PRESETS = [
  { id: 'openai', provider: 'openai', name: 'OpenAI · 正式生产推荐', note: '质量稳定，按量付费', base_url: 'https://api.openai.com/v1', model: 'gpt-image-2' },
  { id: 'gemini-lite', provider: 'gemini', name: 'Google Gemini Flash Lite Image', note: '低价试用，需开通API计费', base_url: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-3.1-flash-lite-image' },
  { id: 'gemini', provider: 'gemini', name: 'Google Gemini Flash Image', note: '质量较高，需开通API计费', base_url: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-3.1-flash-image' },
  { id: 'gemini-pro', provider: 'gemini', name: 'Nano Banana Pro · Gemini 3 Pro Image', note: '专业商品图，4K高质量，成本较高', base_url: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-3-pro-image' },
  { id: 'local', provider: 'openai', name: '本地免费模型网关', note: '不收API费，需要本机显卡和兼容网关', base_url: 'http://127.0.0.1:8080/v1', model: '填写支持 Images Edit 的模型名' },
  { id: 'compatible', provider: 'openai', name: '其他 OpenAI 兼容服务', note: '自行确认支持图片编辑与蒙版', base_url: '', model: '' },
];

function apiUrl(path) {
  if (!path || /^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path}`;
}

async function api(path, options = {}) {
  const multipart = options.body instanceof FormData;
  const response = await fetch(apiUrl(path), { headers: { ...(multipart ? {} : { 'Content-Type': 'application/json' }), ...(options.headers || {}) }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || '请求失败');
  return payload;
}

export default function App() {
  const [completionMode, setCompletionMode] = useState(() => localStorage.getItem('completionMode') || 'ai');
  const [root, setRoot] = useState(PUBLIC_MODE ? '未选择文件夹' : DEFAULT_ROOT);
  const [job, setJob] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [activeFolder, setActiveFolder] = useState('');
  const [filter, setFilter] = useState('all');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [dragged, setDragged] = useState(null);
  const [health, setHealth] = useState(null);
  const [selectedBarcode, setSelectedBarcode] = useState('');
  const [draftCrop, setDraftCrop] = useState({ offset_x: 0, offset_y: 0, zoom: 100 });
  const [zoomInput, setZoomInput] = useState('100');
  const [sizeForm, setSizeForm] = useState({ width: '3000', height: '4000' });
  const [nameInput, setNameInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [aiForm, setAiForm] = useState({ enabled: true, provider: 'openai', base_url: 'https://api.openai.com/v1', model: 'gpt-image-2', api_key: '' });
  const [layout, setLayout] = useState(null);
  const gesture = useRef(null);
  const draftRef = useRef(draftCrop);
  const saveTimer = useRef(null);
  const frameTimer = useRef(null);
  const folderInput = useRef(null);
  const noAiMode = completionMode === 'blank';

  useEffect(() => { api('/api/health').then(setHealth).catch(() => setHealth({ ok: false, ai_configured: false })); }, []);
  useEffect(() => { localStorage.setItem('completionMode', completionMode); setResult(null); }, [completionMode]);
  const folders = useMemo(() => job ? [...new Set(job.items.map(item => item.folder))] : [], [job]);
  useEffect(() => { if (folders.length && !folders.includes(activeFolder)) setActiveFolder(folders[0]); }, [folders, activeFolder]);
  const folderItems = useMemo(() => (job?.items || []).filter(item => item.folder === activeFolder), [job, activeFolder]);
  const visibleItems = filter === 'review' ? folderItems.filter(item => item.needs_review) : folderItems;
  const active = job?.items.find(item => item.id === activeId) || visibleItems[0] || null;
  const previewBarcode = active?.barcodes?.includes(selectedBarcode) ? selectedBarcode : active?.barcodes?.[0];
  const defaultPixels = previewBarcode && ['45', '50'].includes(previewBarcode) ? [3000, 3000] : [3000, 4000];
  const targetPixels = active?.output_sizes?.[previewBarcode] || defaultPixels;
  const target = `${targetPixels[0]} × ${targetPixels[1]}`;
  const isSquare = Math.abs(targetPixels[0] / targetPixels[1] - 1) < .02;
  const safe = active?.image_type === '全身' ? '人物自适应' : isSquare ? `${Math.round(targetPixels[0] * 1664 / 3000)} × ${Math.round(targetPixels[1] * 1665 / 3000)}` : `${Math.round(targetPixels[0] * 2680 / 3000)} × ${Math.round(targetPixels[1] * 2400 / 4000)}`;
  const reviewCount = job?.items.filter(item => item.needs_review).length || 0;
  const outputCount = job?.items.reduce((sum, item) => sum + item.barcodes.length, 0) || 0;
  const detailLike = active?.image_type === '静物' && active.subject_box && (active.subject_box.w * active.subject_box.h) / (active.width * active.height) >= .88;

  useEffect(() => {
    if (!active) return;
    setSelectedBarcode(current => active.barcodes?.includes(current) ? current : (active.barcodes?.[0] || ''));
    setDraftCrop(active.crop);
    setZoomInput(String(active.crop.zoom));
    draftRef.current = active.crop;
  }, [active?.id]);

  useEffect(() => { setSizeForm({ width: String(targetPixels[0]), height: String(targetPixels[1]) }); }, [active?.id, previewBarcode, targetPixels[0], targetPixels[1]]);
  useEffect(() => { if (active && previewBarcode) setNameInput(active.output_names?.[previewBarcode] || `${active.folder}_${previewBarcode}.jpg`); }, [active?.id, previewBarcode, active?.output_names]);
  useEffect(() => {
    if (!active || !previewBarcode) { setLayout(null); return; }
    let live = true;
    api(`/api/images/${active.id}/layout?barcode=${previewBarcode}`).then(next => { if (live) setLayout(next); }).catch(() => { if (live) setLayout(null); });
    return () => { live = false; };
  }, [active?.id, previewBarcode, active?.crop.offset_x, active?.crop.offset_y, active?.crop.zoom, active?.image_type]);

  const scan = async () => {
    setBusy('正在扫描与识别…'); setError(''); setResult(null);
    try { const next = await api('/api/jobs/scan', { method: 'POST', body: JSON.stringify({ root }) }); setJob(next); setActiveFolder(next.items[0]?.folder || ''); setActiveId(next.items[0]?.id || null); }
    catch (err) { setError(err.message.includes('fetch') ? '无法连接本地图像服务，请先运行后端启动命令。' : err.message); }
    finally { setBusy(''); }
  };
  const selectCloudFolder = () => folderInput.current?.click();
  const uploadCloudFolder = async event => {
    const selected = [...(event.target.files || [])].filter(file => /\.(jpe?g|png)$/i.test(file.name));
    if (!selected.length) return;
    setBusy(`正在上传 ${selected.length} 张…`); setError(''); setResult(null);
    try {
      const form = new FormData();
      selected.forEach(file => { form.append('files', file, file.name); form.append('paths', file.webkitRelativePath || file.name); });
      const next = await api('/api/cloud/jobs/scan', { method: 'POST', body: form });
      setRoot(selected[0].webkitRelativePath?.split('/')[0] || '已选商品文件夹');
      setJob(next); setActiveFolder(next.items[0]?.folder || ''); setActiveId(next.items[0]?.id || null);
    } catch (err) { setError(err.message.includes('fetch') ? '云端图像服务暂时不可用，请稍后重试。' : err.message); }
    finally { setBusy(''); event.target.value = ''; }
  };
  const refreshJob = next => { setJob(next); if (!next.items.some(i => i.id === activeId)) setActiveId(next.items[0]?.id); };
  const updateType = async imageType => { if (!active) return; setBusy('正在重排条码…'); try { refreshJob(await api(`/api/images/${active.id}/classification`, { method: 'PUT', body: JSON.stringify({ image_type: imageType }) })); } catch (err) { setError(err.message); } finally { setBusy(''); } };
  const updateCrop = async crop => { if (!active) return; const next = await api(`/api/images/${active.id}/crop`, { method: 'PUT', body: JSON.stringify(crop) }); setJob(old => ({ ...old, items: old.items.map(i => i.id === next.id ? next : i) })); setDraftCrop(next.crop); draftRef.current = next.crop; };
  const setDraft = crop => { setDraftCrop(crop); draftRef.current = crop; };
  const scheduleCropSave = crop => { clearTimeout(saveTimer.current); saveTimer.current = setTimeout(() => updateCrop(crop), 650); };
  const startCanvasDrag = e => { if (!active) return; e.currentTarget.setPointerCapture(e.pointerId); gesture.current = { x: e.clientX, y: e.clientY, crop: { ...draftRef.current }, width: e.currentTarget.clientWidth, height: e.currentTarget.clientHeight }; };
  const moveCanvasDrag = e => { if (!gesture.current) return; const g = gesture.current; const point = { x: e.clientX, y: e.clientY }; cancelAnimationFrame(frameTimer.current); frameTimer.current = requestAnimationFrame(() => { const next = { ...g.crop, offset_x: Math.round(g.crop.offset_x + (point.x - g.x) * targetPixels[0] / g.width), offset_y: Math.round(g.crop.offset_y + (point.y - g.y) * targetPixels[1] / g.height) }; setDraft(next); }); };
  const endCanvasDrag = () => { if (!gesture.current) return; gesture.current = null; clearTimeout(saveTimer.current); updateCrop(draftRef.current); };
  const wheelCanvas = e => { e.preventDefault(); const delta = e.deltaY < 0 ? 4 : -4; cancelAnimationFrame(frameTimer.current); frameTimer.current = requestAnimationFrame(() => { const next = { ...draftRef.current, zoom: Math.max(35, Math.min(240, draftRef.current.zoom + delta)) }; setZoomInput(String(next.zoom)); setDraft(next); scheduleCropSave(next); }); };
  const changeZoom = value => { const zoom = Math.max(35, Math.min(240, Number(value) || 100)); const next = { ...draftRef.current, zoom }; setZoomInput(String(zoom)); setDraft(next); scheduleCropSave(next); };
  const commitZoomInput = () => changeZoom(zoomInput);
  const saveOutputSize = async () => {
    if (!active || !previewBarcode) return;
    setBusy('正在更新画布…'); setError('');
    try {
      const next = await api(`/api/images/${active.id}/output-size`, { method: 'PUT', body: JSON.stringify({ barcode: previewBarcode, width: Number(sizeForm.width), height: Number(sizeForm.height) }) });
      setJob(old => ({ ...old, items: old.items.map(i => i.id === next.id ? next : i) }));
    } catch (err) { setError(err.message); } finally { setBusy(''); }
  };
  const saveOutputName = async () => {
    if (!active || !previewBarcode) return;
    setBusy('正在保存文件名…'); setError('');
    try {
      const next = await api(`/api/images/${active.id}/output-name`, { method: 'PUT', body: JSON.stringify({ barcode: previewBarcode, filename: nameInput }) });
      setJob(old => ({ ...old, items: old.items.map(i => i.id === next.id ? next : i) }));
      setNameInput(next.output_names?.[previewBarcode] || nameInput);
    } catch (err) { setError(err.message); } finally { setBusy(''); }
  };
  const visualTransform = active ? `translate(${(draftCrop.offset_x - active.crop.offset_x) / targetPixels[0] * 100}%, ${(draftCrop.offset_y - active.crop.offset_y) / targetPixels[1] * 100}%) scale(${draftCrop.zoom / active.crop.zoom})` : 'none';
  const subjectStyle = layout ? { left:`${layout.subject_box.x / layout.target[0] * 100}%`, top:`${layout.subject_box.y / layout.target[1] * 100}%`, width:`${layout.subject_box.w / layout.target[0] * 100}%`, height:`${layout.subject_box.h / layout.target[1] * 100}%` } : null;

  const openAiSettings = async () => { try { const current = await api('/api/settings/ai'); setAiForm({ ...current, api_key: '' }); } catch {} setShowSettings(true); };
  const saveAiSettings = async () => { setBusy('正在保存 AI 配置…'); try { const next = await api('/api/settings/ai', { method: 'PUT', body: JSON.stringify(aiForm) }); setHealth(old => ({ ...old, ai_configured: next.configured, ai: next })); setShowSettings(false); } catch (err) { setError(err.message); } finally { setBusy(''); } };
  const reorder = async targetId => {
    if (!dragged || dragged === targetId || !job) return;
    const ids = job.items.map(i => i.id); const from = ids.indexOf(dragged); const to = ids.indexOf(targetId);
    ids.splice(to, 0, ids.splice(from, 1)[0]); setDragged(null); setBusy('正在重排条码…');
    try { refreshJob(await api(`/api/jobs/${job.id}/order`, { method: 'PUT', body: JSON.stringify({ image_ids: ids }) })); } catch (err) { setError(err.message); } finally { setBusy(''); }
  };
  const process = async () => { if (!job) return; setBusy('正在检查补图需求…'); try { refreshJob(await api(`/api/jobs/${job.id}/process`, { method: 'POST' })); } catch (err) { setError(err.message); } finally { setBusy(''); } };
  const exportAll = async () => {
    if (!job) return;
    setBusy('正在导出原尺寸切图…'); setResult(null); setError('');
    try {
      const next = await api(`/api/jobs/${job.id}/export`, { method: 'POST', body: JSON.stringify({ completion_mode: completionMode }) });
      setResult(next);
      if (PUBLIC_MODE && next.download_url) {
        const link = document.createElement('a'); link.href = apiUrl(next.download_url); link.download = ''; document.body.appendChild(link); link.click(); link.remove();
      }
      if (!noAiMode && next.failed?.some(item => /API Key|无权限|认证/.test(item.error))) {
        setError('当前 API Key 无效或没有图片编辑权限，请重新填写真实 Key 后重试失败图片。');
        setShowSettings(true);
      }
    } catch (err) { setError(err.message); } finally { setBusy(''); }
  };
  const exportWarnings = result?.exported?.filter(item => item.warning) || [];

  return <div className="app">
    <header className="topbar">
      <div className="brand"><strong>CUT<span>/</span>规</strong><i/><b>规范切图工作台</b></div>
      <div className="path-input"><FolderOpen size={17}/><input aria-label="根目录路径" value={root} onChange={e => !PUBLIC_MODE && setRoot(e.target.value)} readOnly={PUBLIC_MODE}/><button onClick={PUBLIC_MODE ? selectCloudFolder : scan} disabled={!!busy}>{busy || (PUBLIC_MODE ? '选择文件夹' : '读取根目录')}</button><input ref={folderInput} className="hidden-folder-input" type="file" accept="image/jpeg,image/png" multiple webkitdirectory="" directory="" onChange={uploadCloudFolder}/></div>
      <div className="completion-switch" role="group" aria-label="扩图模式"><button className={!noAiMode ? 'active' : ''} onClick={() => setCompletionMode('ai')}><Sparkles size={14}/>AI扩图</button><button className={noAiMode ? 'active' : ''} onClick={() => setCompletionMode('blank')}><Crop size={14}/>无AI留白</button></div>
      <span className={`service ${health?.ok ? 'online' : ''}`}><i/>{health?.ok ? (noAiMode ? '图像服务在线 · 不调用AI' : `图像服务在线 · AI${health.ai_configured ? '已配置' : '未配置'}`) : '图像服务未启动'}</span>
      {!noAiMode && <button className="settings-button" onClick={openAiSettings} aria-label="AI 设置"><Settings size={17}/></button>}
    </header>

    <aside className="queue">
      <div className="panel-title"><strong>商品文件夹</strong><span>{folders.length} 个</span></div>
      <div className="folder-list">{folders.map(folder => <button key={folder} className={activeFolder === folder ? 'active' : ''} onClick={() => { setActiveFolder(folder); setActiveId(null); }}><FolderOpen size={15}/><span>{folder}</span><b>{job.items.filter(i => i.folder === folder).length}</b></button>)}</div>
      <div className="queue-tabs"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>全部 {folderItems.length}</button><button className={filter === 'review' ? 'active' : ''} onClick={() => setFilter('review')}>待复核 {folderItems.filter(i => i.needs_review).length}</button></div>
      <div className="queue-list">{visibleItems.map((item, index) => <button draggable key={item.id} onDragStart={() => setDragged(item.id)} onDragOver={e => e.preventDefault()} onDrop={() => reorder(item.id)} className={`queue-item ${active?.id === item.id ? 'selected' : ''}`} onClick={() => setActiveId(item.id)}>
        <GripVertical className="grip" size={15}/><div className="thumb"><img loading="lazy" src={`${apiUrl(item.preview_url)}?barcode=${item.barcodes[0] || '43'}&size=220&completion_mode=${completionMode}&v=${item.crop.zoom}-${item.crop.offset_x}-${item.crop.offset_y}`}/><span>{index + 1}</span></div>
        <div className="queue-copy"><div><strong>{item.image_type}</strong>{item.needs_review && <AlertTriangle size={13}/>}</div><em className={item.confidence < .8 ? 'warn' : ''}>{Math.round(item.confidence * 100)}%</em><small>{item.barcodes.length ? item.barcodes.map(b => `_${b}`).join(' · ') : '编号已用尽'}</small><p>{item.filename}</p></div>
      </button>)}</div>
    </aside>

    <main className="workspace">
      {!job && <section className="empty-state"><div className="empty-mark"><Crop size={38}/></div><h1>{PUBLIC_MODE ? '选择商品文件夹，联机完成整批切图' : '选择根目录，自动完成整批切图'}</h1><p>系统会按商品文件夹扫描、识别图片类型、定位主体并分配输出名称。仅异常图片需要人工复核。</p><button onClick={PUBLIC_MODE ? selectCloudFolder : scan} disabled={!!busy}><Play size={17}/>{busy || (PUBLIC_MODE ? '选择文件夹并上传' : '开始自动处理')}</button><small>支持 JPG、JPEG、PNG · 原始图片不会被修改</small>{error && <div className="error"><AlertTriangle size={15}/>{error}</div>}</section>}
      {job && active && <>
        <div className="toolbar"><div><button onClick={() => updateCrop({ offset_x: 0, offset_y: 0, zoom: 100 })}><Maximize size={16}/>适应画布</button><button onClick={() => updateCrop({ ...draftCrop, offset_x: 0, offset_y: 0 })}><Crop size={16}/>主体居中</button><button onClick={() => updateCrop({ offset_x: 0, offset_y: 0, zoom: 100 })}><Undo2 size={16}/>重置</button></div><span>{active.filename}</span><div className="ratio"><button className={isSquare ? 'active' : ''}>1:1</button><button className={!isSquare ? 'active' : ''}>{targetPixels[0]}:{targetPixels[1]}</button></div></div>
        <section className="stage"><div className="canvas interactive" style={{aspectRatio:`${targetPixels[0]} / ${targetPixels[1]}`}} onPointerDown={startCanvasDrag} onPointerMove={moveCanvasDrag} onPointerUp={endCanvasDrag} onPointerCancel={endCanvasDrag} onWheel={wheelCanvas}><div className="canvas-media" style={{transform:visualTransform}}><img draggable="false" src={`${apiUrl(active.preview_url)}?barcode=${previewBarcode || '43'}&size=680&completion_mode=${completionMode}&v=${active.crop.zoom}-${active.crop.offset_x}-${active.crop.offset_y}-${target}`}/>{subjectStyle && <div className="subject-box" style={subjectStyle}><span>{active.image_type === '全身' ? '人物主体' : '鞋子主体'}</span></div>}</div>{active.image_type !== '全身' && <div className={`safe-zone ${isSquare ? 'safe-square' : ''}`}><span>鞋子主体规范区 · {safe}</span></div>}</div><div className="canvas-meta"><span>输出画布 {target}</span><span>{active.image_type === '腿模' ? '腿模靠顶 · 鞋子完整' : active.image_type === '全身' ? '人物躯体完整自适应' : `规范区 ${safe}`}</span><span>{detailLike ? '完整画布：产品细节铺满' : noAiMode ? '完整画布：不足区域留白' : '完整画布：不足时 AI 整图扩展'}</span></div></section>
        <section className="output-strip"><div><strong>本商品生成输出</strong><span>拖动左侧图片可重新分配</span></div><div className="barcode-list">{folderItems.flatMap(i => i.barcodes.map(code => <button key={`${i.id}-${code}`} className={i.id === active.id && code === previewBarcode ? 'active' : ''} onClick={() => { setActiveId(i.id); setSelectedBarcode(code); }}><img loading="lazy" src={`${apiUrl(i.preview_url)}?barcode=${code}&size=180&completion_mode=${completionMode}&v=${(i.output_sizes?.[code] || []).join('x')}`}/><strong>{i.output_names?.[code] || `${i.folder}_${code}.jpg`}</strong><small>{(i.output_sizes?.[code] || (['45','50'].includes(code) ? [3000,3000] : [3000,4000])).join(' × ')}</small></button>))}</div></section>
      </>}
    </main>

    <aside className="inspector">
      <div className="panel-title"><strong>图像属性</strong><CircleHelp size={14}/></div>
      {active ? <div className="inspector-body">
        <Field title="图片分类"><div className="segments">{['静物','腿模','全身'].map(v => <button key={v} className={active.image_type === v ? 'active' : ''} onClick={() => updateType(v)}>{v}</button>)}</div><p className="reason">{active.reason}</p></Field>
        <Field title="输出映射"><div className="mapping">{active.barcodes.length ? active.barcodes.map(b => <b key={b}>{active.output_names?.[b] || `${active.folder}_${b}.jpg`}</b>) : <b className="warning-text">无可用编号</b>}</div>{previewBarcode && <><div className="output-name"><input aria-label="当前输出文件名" value={nameInput} onChange={e => setNameInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') saveOutputName(); }}/><button onClick={saveOutputName} disabled={!!busy}>保存命名</button></div><small className="size-help">仅修改当前编号 _{previewBarcode} 的导出文件名；未填写扩展名时自动补 .jpg。</small></>}</Field>
        <Field title="画面定位"><label>画布尺寸<span>{target}</span></label><div className="custom-size"><input aria-label="自定义宽度" inputMode="numeric" value={sizeForm.width} onChange={e => setSizeForm({...sizeForm,width:e.target.value.replace(/\D/g,'').slice(0,4)})}/><i>×</i><input aria-label="自定义高度" inputMode="numeric" value={sizeForm.height} onChange={e => setSizeForm({...sizeForm,height:e.target.value.replace(/\D/g,'').slice(0,4)})}/><button onClick={saveOutputSize} disabled={!!busy}>应用尺寸</button></div><small className="size-help">当前编号独立设置，支持 320–8000 px；比例随宽高自动更新。</small><label>安全区<span>{safe}</span></label><p className="direct-help">直接拖动画面定位；滚轮、滑杆或百分比输入框均可缩放。</p><label className="slider">缩放<input type="range" min="35" max="240" value={draftCrop.zoom} onChange={e => changeZoom(e.target.value)}/><span className="zoom-input"><input aria-label="缩放百分比" inputMode="numeric" value={zoomInput} onChange={e => setZoomInput(e.target.value.replace(/\D/g,'').slice(0,3))} onBlur={commitZoomInput} onKeyDown={e => { if (e.key === 'Enter') { commitZoomInput(); e.currentTarget.blur(); } }}/><i>%</i></span></label></Field>
        <Field title="完整画布"><div className="fill-method">{noAiMode ? <Crop size={16}/> : <Sparkles size={16}/>}<span><b>{detailLike ? '产品细节自动铺满' : noAiMode ? '不足区域纯白留空' : 'AI 完整画布扩图'}</b><small>{detailLike ? '优先放大裁切，保持产品细节' : noAiMode ? '不连接AI，不生成或改动缺失区域' : '画布未铺满时整张AI扩图，不虚化、不拼接'}</small></span></div></Field>
        <Field title="自动处理"><ul className="checks"><li className="ok"><Check/>原始图片保持不变</li><li className="ok"><Check/>支持逐个输出自定义命名</li><li className="ok"><Check/>JPEG 自动控制在 2MB 内</li><li className="ok"><Check/>{noAiMode ? '画布不足区域输出纯白' : '缺失边缘自动识别并自然扩图'}</li>{noAiMode ? <li className="ok"><Check/>不连接任何AI服务</li> : <li className={active.background === 'complex' && !health?.ai_configured ? 'warn' : 'ok'}>{active.background === 'complex' && !health?.ai_configured ? <AlertTriangle/> : <Check/>}{active.background === 'complex' && !health?.ai_configured ? '复杂背景：请配置可用的图片编辑模型' : '最多 3 张并发处理，可直接整批导出'}</li>}</ul></Field>
      </div> : <div className="inspector-empty">读取根目录后显示图像属性</div>}
      <div className="export-area"><button className="secondary" onClick={process} disabled={!job || !!busy}><RotateCcw size={16}/>检查补图条件</button><button className="export" onClick={exportAll} disabled={!job || !!busy}><ImagePlus size={17}/>{busy || `一键导出全部 ${outputCount} 张`}</button><div className="export-meta"><span>{reviewCount ? `分类待确认 ${reviewCount}` : '无需逐张复核'}</span><b>全部写回商品文件夹</b></div>{result && <div className={result.failed.length || exportWarnings.length ? 'result warning result-detail' : 'result success'}><b>{result.failed.length ? `已导出 ${result.exported.length} 张，${result.failed.length} 张最终失败` : exportWarnings.length ? `已导出全部 ${result.exported.length} 张；${exportWarnings.length} 张使用未补图构图兜底` : `已成功导出全部 ${result.exported.length} 张`}</b>{exportWarnings.slice(0,3).map(f => <small key={`${f.path}-warning`}>{f.filename}（_{f.barcode}）：AI失败，已按当前主体位置和缩放正常导出未补图版本。原因：{f.warning}</small>)}{result.failed.slice(0,3).map(f => <small key={`${f.image_id}-${f.barcode}`}>{f.filename}（_{f.barcode}）：{f.error}</small>)}{exportWarnings.length > 3 && <small>另有 {exportWarnings.length - 3} 张已使用未补图构图兜底。</small>}{result.failed.length > 3 && <small>另有 {result.failed.length - 3} 项最终失败。</small>}</div>}{error && <div className="error"><AlertTriangle size={14}/>{error}</div>}</div>
    </aside>
    {!noAiMode && showSettings && <div className="modal-backdrop" onMouseDown={() => setShowSettings(false)}><section className="settings-modal" onMouseDown={e => e.stopPropagation()}><header><div><b>AI 识别与扩图配置</b><small>{PUBLIC_MODE ? '公开版由服务端统一管理，所有同事共用' : '选择可直接使用的图片编辑服务'}</small></div><button onClick={() => setShowSettings(false)}><X size={18}/></button></header><label className="toggle"><input type="checkbox" checked={aiForm.enabled} disabled={PUBLIC_MODE} onChange={e => setAiForm({...aiForm,enabled:e.target.checked})}/>启用 AI 主体识别与完整扩图</label><label>服务类型<select disabled={PUBLIC_MODE} value={AI_PRESETS.find(p => p.provider === aiForm.provider && p.base_url === aiForm.base_url && p.model === aiForm.model)?.id || 'compatible'} onChange={e => { const preset = AI_PRESETS.find(p => p.id === e.target.value); setAiForm({...aiForm,provider:preset.provider,base_url:preset.base_url,model:preset.model}); }}>{AI_PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}｜{p.note}</option>)}</select></label><label>API 站点链接<input disabled={PUBLIC_MODE} value={aiForm.base_url} onChange={e => setAiForm({...aiForm,base_url:e.target.value})} placeholder="https://服务地址/v1"/></label><label>图片编辑模型<input disabled={PUBLIC_MODE} value={aiForm.model} onChange={e => setAiForm({...aiForm,model:e.target.value})} placeholder="必须支持图片编辑/完整扩图"/></label>{!PUBLIC_MODE && <label>API Key<input type="password" value={aiForm.api_key} onChange={e => setAiForm({...aiForm,api_key:e.target.value})} placeholder={health?.ai_configured ? '已安全保存；留空保持不变' : '输入所选服务的 API Key'}/></label>}<p>{PUBLIC_MODE ? `当前云端配置：${health?.ai?.provider || '未配置'} / ${health?.ai?.model || '未配置'}。API Key 仅保存在服务端，不会发送到浏览器。` : 'Key 仅保存在当前 Mac 系统钥匙串；切换供应商时请填写对应的新 Key。'}</p>{PUBLIC_MODE ? <button className="save-settings" onClick={() => setShowSettings(false)}>关闭</button> : <button className="save-settings" onClick={saveAiSettings}>{busy || '保存配置'}</button>}</section></div>}
  </div>;
}

function Field({ title, children }) { return <section className="field"><h3>{title}</h3>{children}</section>; }
