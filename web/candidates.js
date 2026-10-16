'use strict';
// Configurable grid candidates. Backend is the authoritative defaults/validation source.
let candidateDefaults=null;
const candidateStorageKey='oxidation-v1-candidate-grid-v1';
const candidateLabels={...processing,Temperature:'氧化温度 / °C',E_Time:'暴露时间 / h'};
function candidateRows(names,cfg){
  return `<table class="candidate-ranges"><thead><tr><th>元素 / at.%</th><th>下限</th><th>上限</th><th>步长</th></tr></thead><tbody>${names.map(e=>`<tr><th scope="row">${esc(e)}</th>${['min','max','step'].map(k=>`<td><input type="number" id="range-${e}-${k}" data-range-element="${e}" data-range-field="${k}" aria-label="${esc(e)} ${k==='min'?'下限':k==='max'?'上限':'步长'}" value="${esc(cfg.elements[e][k])}" min="${k==='step'?'0.000001':'0'}" max="100" step="any" required></td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderCandidateSettings(cfg){
  const primary=['Al','Cr','Ni','W','Ti','Mo','Ta','Si'];
  const additional=elements.filter(e=>e!=='Co'&&!primary.includes(e));
  $('#candidate-range-main').innerHTML=candidateRows(primary,cfg);
  $('#candidate-range-extra').innerHTML=candidateRows(additional,cfg);
  $('#candidate-co-min').value=cfg.co.min;
  $('#candidate-co-max').value=cfg.co.max;
  $('#candidate-conditions').innerHTML=Object.entries(candidateLabels).map(([k,label])=>`<label>${esc(label)}<input id="candidate-condition-${k}" type="number" value="${esc(cfg.conditions[k])}" min="0" step="any" required></label>`).join('');
}
function candidateSourceChanged(){
  const generated=$('#candidate-source').value==='generated';
  $('#candidate-generation-settings').hidden=!generated;
  $('#candidate-csv-note').hidden=generated;
  $('#candidate-count').disabled=!generated;
  $('#seed').disabled=!generated;
}
function candidateConfigFromUI(){
  const cfg={elements:{},co:{},conditions:{}};
  for(const e of elements.filter(e=>e!=='Co')){
    cfg.elements[e]={};
    for(const k of ['min','max','step'])cfg.elements[e][k]=Number($(`#range-${e}-${k}`).value);
  }
  cfg.co={min:Number($('#candidate-co-min').value),max:Number($('#candidate-co-max').value)};
  for(const k of Object.keys(candidateLabels))cfg.conditions[k]=Number($(`#candidate-condition-${k}`).value);
  return cfg;
}
function screenRequest(){
  const source=$('#candidate-source').value;
  const request={mode:mode(),candidate_source:source,kinetic:Number($('#kinetic').value),divergence:Number($('#divergence').value),prescreen:Number($('#prescreen').value),cross_source:$('#cross-source').checked};
  if(source==='csv'){
    if(!$('#csv-text').value.trim())throw Error('请先在数据管理页导入 CSV，再运行候选筛选。');
    request.csv=$('#csv-text').value;
    return request;
  }
  if(!candidateDefaults)throw Error('候选范围配置尚未加载，请刷新页面后重试。');
  for(const input of [...$$('#candidate-generation-settings input'),$('#candidate-count'),$('#seed')]){
    if(!input.checkValidity()){
      const details=input.closest('details');if(details)details.open=true;
      input.reportValidity();
      throw Error('请填写有效的候选范围、步长、数量及条件参数。');
    }
  }
  request.n=Number($('#candidate-count').value);
  request.seed=Number($('#seed').value);
  request.generation_config=candidateConfigFromUI();
  return request;
}
function candidateGenerationSummary(info){
  if(!info)return '';
  if(info.source==='csv')return '<article class="card"><h3>候选输入记录</h3><p class="muted">本次使用导入 CSV 中的状态，生成范围、步长及固定条件不参与计算。</p></article>';
  // Store only successful, backend-normalized settings for the next local visit.
  try{localStorage.setItem(candidateStorageKey,JSON.stringify({settings:info.settings,n:info.generated_count,seed:info.seed}));}catch(e){}
  const cfg=info.settings;
  const ranges=Object.entries(cfg.elements).filter(([e,s])=>s.min!==0||s.max!==0).map(([e,s])=>s.min===s.max?`${e} = ${s.min}`:`${e} ${s.min}–${s.max}，步长 ${s.step}，网格末值 ${info.effective_grid_max[e]}`).join('；');
  const conditions=Object.entries(candidateLabels).map(([k,label])=>`${label}：${cfg.conditions[k]}`).join('；');
  return `<article class="card"><h3>本次候选生成参数</h3><p class="muted">${esc(ranges||'所有非 Co 元素均固定为 0')} at.%；Co 为余量，范围 ${cfg.co.min}–${cfg.co.max} at.%。</p><p class="muted">${esc(conditions)}</p><p class="subnote">生成 ${info.generated_count} 个不同成分状态，随机种子 ${info.seed}。按配置网格随机抽样，非全网格扫描。完整参数可下载 candidate_generation.json；prediction.csv 保留实际生成状态。合成来源编号仅用于占位。</p></article>`;
}
async function initCandidateSettings(){
  $('#run-screen').disabled=true;
  try{
    const d=await api('/api/candidates/config');candidateDefaults=d.defaults;
    renderCandidateSettings(candidateDefaults);
    try{
      const saved=JSON.parse(localStorage.getItem(candidateStorageKey)||'null');
      if(saved&&saved.settings){
        const c=structuredClone(candidateDefaults);
        for(const e of Object.keys(c.elements))Object.assign(c.elements[e],saved.settings.elements?.[e]||{});
        Object.assign(c.co,saved.settings.co||{});
        Object.assign(c.conditions,saved.settings.conditions||{});
        renderCandidateSettings(c);
        $('#candidate-count').value=saved.n;$('#seed').value=saved.seed;
      }
    }catch(e){}
    $('#candidate-config-status').textContent='范围与条件已加载；成功运行后的设置会保存在当前浏览器。';
  }catch(e){$('#candidate-config-status').textContent='配置加载失败，请确认后端文件已同步替换。';error(e.message);}
  finally{$('#run-screen').disabled=false;candidateSourceChanged();}
}
$('#candidate-source').onchange=candidateSourceChanged;
$('#candidate-apply-step').onclick=()=>{
  const input=$('#candidate-common-step');
  if(!input.reportValidity())return;
  $$('[data-range-field="step"]').forEach(x=>x.value=input.value);
};
$('#candidate-reset').onclick=()=>{
  if(!candidateDefaults)return;
  renderCandidateSettings(candidateDefaults);
  $('#candidate-count').value=180;$('#seed').value=42;$('#candidate-common-step').value=.5;
  try{localStorage.removeItem(candidateStorageKey);}catch(e){}
  $('#candidate-config-status').textContent='已恢复默认生成范围、固定条件、数量和随机种子。';
};
initCandidateSettings();
