'use strict';
let vizSettings = null;
let atlasJob = null;
let pairJob = null;
const vizDisplayed = {};
const vizDownloads = {};
const clusterNames = {consistent:'参考基本一致',mixed:'参考存在分歧',insufficient:'参考不足'};
const comparisonNames = {different:'机制邻域不同',similar:'机制邻域相近',mixed:'差异与分歧并存',insufficient:'比较证据不足'};
const palette = ['#168b89','#e19535','#4078b3','#8b70ad','#c46570','#839b42','#56a6bc','#b27e59'];

function graphStyle() {
 return {size:Number($('#viz-point-size').value)||3,alpha:Number($('#viz-alpha').value)||.55,font:Number($('#viz-font').value)||12};
}
function svgSpace(points, states=[], groupKey='cluster_id', highlighted=null) {
 if(!points.length)return '<p class="muted">没有可用坐标。</p>';
 const st=graphStyle(),all=[...points,...states],w=620,h=410,L=50,R=20,T=20,B=45;
 let xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x)),ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));
 const xp=.07*(xmax-xmin||1),yp=.07*(ymax-ymin||1);xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
 const x=v=>L+(v-xmin)/(xmax-xmin)*(w-L-R),y=v=>h-B-(v-ymin)/(ymax-ymin)*(h-T-B);
 let s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="UMAP mechanism space"><rect width="100%" height="100%" fill="white"/>';
 for(let i=0;i<5;i++){const v=i/4,xx=L+v*(w-L-R),yy=h-B-v*(h-T-B);s+='<line x1="'+xx+'" y1="'+T+'" x2="'+xx+'" y2="'+(h-B)+'" stroke="#edf2f4"/><line x1="'+L+'" y1="'+yy+'" x2="'+(w-R)+'" y2="'+yy+'" stroke="#edf2f4"/><text x="'+xx+'" y="'+(h-B+16)+'" text-anchor="middle" style="font-size:10px">'+(xmin+v*(xmax-xmin)).toFixed(1)+'</text><text x="'+(L-7)+'" y="'+(yy+3)+'" text-anchor="end" style="font-size:10px">'+(ymin+v*(ymax-ymin)).toFixed(1)+'</text>';}
 s+='<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(h-B)+'" stroke="#a9bec8"/><line x1="'+L+'" y1="'+(h-B)+'" x2="'+(w-R)+'" y2="'+(h-B)+'" stroke="#a9bec8"/>';
 points.forEach((p,index)=>{const g=Number(p[groupKey]??0),color=g<0?'#a7b2b8':palette[(g+(groupKey==='channel'?-1:0))%palette.length],opacity=highlighted!==null&&g!==highlighted?.12:st.alpha;
 s+='<circle class="umap-reference" data-point="'+index+'" cx="'+x(p.x)+'" cy="'+y(p.y)+'" r="'+st.size+'" fill="'+color+'" opacity="'+opacity+'" tabindex="0"><title>'+esc((p.sample_id||'')+' · '+(p.Ref_source_num||'')+' · '+(p.text||''))+'</title></circle>';});
 states.forEach((state,index)=>{const color=index===0?'#d7792a':'#286ec4';state.references?.forEach(r=>{s+='<circle cx="'+x(r.x)+'" cy="'+y(r.y)+'" r="7" stroke="'+color+'" stroke-width="2" fill="none"><title>'+esc(r.reference_id+' · '+r.sample_id)+'</title></circle>';});
 const xx=x(state.x),yy=y(state.y);s+=index===0?'<path d="M '+xx+' '+(yy-12)+' L '+(xx+3)+' '+(yy-3)+' L '+(xx+12)+' '+(yy-3)+' L '+(xx+5)+' '+(yy+3)+' L '+(xx+8)+' '+(yy+12)+' L '+xx+' '+(yy+6)+' L '+(xx-8)+' '+(yy+12)+' L '+(xx-5)+' '+(yy+3)+' L '+(xx-12)+' '+(yy-3)+' L '+(xx-3)+' '+(yy-3)+' Z" fill="'+color+'" stroke="#183747"/>':'<polygon points="'+xx+','+(yy-10)+' '+(xx+10)+','+yy+' '+xx+','+(yy+10)+' '+(xx-10)+','+yy+'" fill="'+color+'" stroke="#183747"/>';
 s+='<text x="'+(xx+12)+'" y="'+(yy-10)+'" style="font-size:'+st.font+'px;fill:'+color+'">'+esc(state.sample_id)+'</text>';});
 s+='<text x="'+w/2+'" y="'+(h-9)+'" text-anchor="middle" style="font-size:'+st.font+'px">UMAP 1</text><text x="17" y="'+h/2+'" transform="rotate(-90 17 '+h/2+')" text-anchor="middle" style="font-size:'+st.font+'px">UMAP 2</text></svg>';
 return s;
}
function referenceList(refs){
 return refs.map(r=>'<div class="analysis-source" id="viz-ref-'+esc(r.reference_id)+'"><b>'+esc(r.reference_id)+'</b> · 来源 '+esc(r.Ref_source_num)+' · 样本 '+esc(r.sample_id)+' · '+esc(r.condition_id)+'<p>'+esc(r.text)+'</p></div>').join('');
}
function bindGraph(container,points){
 const info=container.querySelector('.umap-point-info');container.querySelectorAll('[data-point]').forEach(el=>{
  const action=()=>{const p=points[Number(el.dataset.point)];info.innerHTML='<b>'+esc(p.sample_id)+'</b> · 来源 '+esc(p.Ref_source_num)+' · '+esc(p.condition_id)+'<p>'+esc(p.text)+'</p>';};
  el.onclick=action;el.onkeydown=e=>{if(e.key==='Enter')action();};
 });
}
function drawSpaces(result,target){
 vizDisplayed[target]=result;
 const dest=$(target);let html='<article class="card"><span class="badge '+(result.mode==='demo'?'amber':'')+'">'+(result.mode==='demo'?'合成演示':'本地研究数据')+'</span><p class="subnote">'+esc(result.note||'')+'</p></article>';
 if(result.global){html+='<article class="card"><h3>整体六通道语义空间</h3>'+svgSpace(result.global.points||[],[],'channel')+'<p class="subnote">'+palette.slice(0,6).map((color,i)=>'<span style="color:'+color+'">● CH '+(i+1)+'</span>').join(' · ')+'</p><p class="subnote">'+esc(result.global.note||'')+'</p><div class="umap-point-info"></div></article>';}
 html+='<div class="umap-grid">'+result.channels.map(c=>'<article class="card umap-card" data-channel="'+c.channel+'"><h3>CH '+c.channel+' · '+esc(c.name)+'</h3><div class="umap-plot">'+svgSpace(c.points,c.states||[])+'</div><div class="umap-clusters">'+(c.clusters||[]).map(cluster=>'<button data-cluster="'+cluster.cluster_id+'">'+esc(cluster.label)+' · '+cluster.size+'</button>').join('')+'</div><div class="umap-point-info"></div><p class="subnote">'+(c.states?.map((s,i)=>'<span style="color:'+(i===0?'#d7792a':'#286ec4')+'">'+esc(s.sample_id)+(i===0?' ★':' ◆')+'</span>').join(' · ')||'噪声 '+(c.noise_count||0)+' · 聚类范围 '+result.scope)+'<br>'+esc(c.coordinate_source||'训练参考 UMAP，预测点 transform')+' · 拟合范围 '+esc(c.fit_scope||'train_only')+'</p></article>').join('')+'</div>';
 if(result.states)html+='<article class="card"><h3>投影样本与 Top-5 证据</h3>'+result.channels.map(c=>'<details><summary>CH '+c.channel+' · '+esc(c.name)+'</summary>'+c.states.map(s=>'<h4>'+esc(s.sample_id)+'</h4>'+referenceList(s.references)).join('')+'</details>').join('')+'</article>';
 dest.innerHTML=html;
 if(result.global?.points.length)bindGraph(dest.firstElementChild.nextElementSibling,result.global.points);
 dest.querySelectorAll('[data-channel]').forEach(card=>{const c=result.channels.find(x=>x.channel===Number(card.dataset.channel));bindGraph(card,c.points);let active=null;card.querySelectorAll('[data-cluster]').forEach(button=>button.onclick=()=>{const g=Number(button.dataset.cluster);active=active===g?null:g;card.querySelector('.umap-plot').innerHTML=svgSpace(c.points,c.states||[],'cluster_id',active);bindGraph(card,c.points);});});
 if(vizDownloads[target])dest.insertAdjacentHTML('afterbegin',vizDownloads[target]);
}
function renderClusterSummary(result){
 $('#cluster-output').innerHTML='<article class="card"><h3>六通道簇机制解释</h3><p class="subnote">数值聚类后的代表证据解释；导入的旧总结与新 GLM 结果标明来源。</p>'+result.clusters.map(c=>'<details open><summary>CH '+c.channel+' · Cluster '+c.cluster_id+' · '+esc(c.short_label||c.full_label||'')+' · n='+c.size+'</summary><p class="analysis-claim">'+esc(c.mechanism_summary||'')+'</p><p class="subnote">来源 '+esc(c.source)+' · '+esc(clusterNames[c.heterogeneity]||c.heterogeneity||'导入总结')+'</p>'+uncertaintyBox(c.uncertainties||[])+'<div class="evidence-links">'+(c.supporting_reference_ids||[]).map(id=>'<a href="#viz-ref-'+esc(id)+'">'+esc(id)+'</a>').join('')+'</div>'+referenceList(c.representatives)+'</details>').join('')+'</article>';
}
function renderPairSummary(result){
 $('#pair-summary-output').innerHTML='<article class="card analysis-overall"><h3>整体机制差异</h3><p class="analysis-claim">'+esc(result.overall.summary)+'</p>'+uncertaintyBox(result.overall.uncertainties)+'</article>'+result.channels.map(c=>'<article class="card"><div class="card-head"><h3>CH '+c.channel+' · '+esc(c.name)+'</h3><span class="badge">'+esc(comparisonNames[c.comparison_status])+'</span></div><p class="analysis-claim">'+esc(c.summary)+'</p><p><b>共同特征：</b>'+esc(c.common_features.join('；'))+'</p><p><b>差异：</b>'+esc(c.differences.join('；'))+'</p><div class="evidence-links">'+c.supporting_reference_ids.map(id=>'<a href="#viz-ref-'+esc(id)+'">'+esc(id)+'</a>').join('')+'</div>'+uncertaintyBox(c.uncertainties)+'</article>').join('');
}
async function waitViz(ident,progressTarget,target){
 const el=$(progressTarget);
 while(true){
  const j=await api('/api/visualization/job/'+ident);el.textContent=j.stage;el.classList.toggle('failed',j.status==='failed'||j.status==='interrupted');
  if(j.status==='failed'||j.status==='interrupted')throw Error(j.error||j.stage);
  if(j.status==='completed'){
   const r=j.result;
   delete vizDownloads[target];
   if(r.kind==='cluster_summary')renderClusterSummary(r);
   else if(r.kind==='pair_summary')renderPairSummary(r);
   else drawSpaces(r,target);
   const downloadBlock='<article class="card" data-viz-downloads><h3>坐标、图表与报告下载</h3>'+downloads(j.downloads)+'</article>';
   vizDownloads[target]=downloadBlock;
   $(target).querySelector('[data-viz-downloads]')?.remove();
   $(target).insertAdjacentHTML('afterbegin',downloadBlock);
   if(r.kind==='atlas'){atlasJob=ident;$('#viz-summarize').disabled=false;}
   if(r.kind==='pair_projection'){pairJob=ident;$('#pair-summarize').disabled=false;}
   await refreshVizHistory();return j;
  }
  await new Promise(resolve=>setTimeout(resolve,1500));
 }
}
async function startViz(request,progressTarget,target){
 $(progressTarget).textContent='正在提交任务…';
 const j=await api('/api/visualization/start',request);
 return waitViz(j.id,progressTarget,target);
}
async function projectCurrentSample(){
 if(!lastPrediction?.run_id)throw Error('请先完成带 Top-5 的预测。');
 const index=$('#case-select')?Number($('#case-select').value):0;
 return startViz({kind:'sample_projection',run_id:lastPrediction.run_id,sample_index:index},'#sample-viz-progress','#sample-viz-output');
}
async function projectCurrentPair(){
 if(!screenResult?.pairs?.length)throw Error('当前没有可分析候选对。');
 const index=window.selectedPairIndex??0,p=screenResult.pairs[index];
 $('#pair-context').textContent=p.sample_1+' ↔ '+p.sample_2+' · |Δlgkp|='+p.abs_delta_lgkp.toFixed(5)+' · D_mech='+p.D_mech.toFixed(3);
 $('#pair-summary-output').innerHTML='';$('#pair-summarize').disabled=true;page('pair-analysis');
 return startViz({kind:'pair_projection',run_id:screenResult.run_id,pair_index:index},'#pair-progress','#pair-projection-output');
}
async function loadVizSettings(){
 const d=await api('/api/visualization/config');vizSettings=d.config;
 $('#viz-config').value=JSON.stringify(d.config,null,2);
 const c=d.config;$('#viz-neighbors').value=c.n_neighbors;$('#viz-min-dist').value=c.min_dist;$('#viz-method').value=c.cluster_method;$('#viz-clusters').value=c.n_clusters;$('#viz-representatives').value=c.representatives;
}
async function saveVizSettings(){
 const c=JSON.parse($('#viz-config').value);
 Object.assign(c,{n_neighbors:Number($('#viz-neighbors').value),min_dist:Number($('#viz-min-dist').value),cluster_method:$('#viz-method').value,n_clusters:Number($('#viz-clusters').value),representatives:Number($('#viz-representatives').value)});
 await api('/api/visualization/config',c);vizSettings=c;$('#viz-config').value=JSON.stringify(c,null,2);
}
async function refreshVizHistory(){
 const jobs=await api('/api/visualization/jobs');
 $('#viz-history').innerHTML=jobs.length?'<div class="scroll-table"><table><thead><tr><th>任务</th><th>状态</th><th>阶段</th><th>创建时间</th><th>查看</th></tr></thead><tbody>'+jobs.map(j=>'<tr><td>'+esc(j.kind)+'</td><td>'+esc(j.status)+'</td><td>'+esc(j.stage)+'</td><td>'+esc(j.created)+'</td><td><button data-viz-job="'+j.id+'">打开结果</button></td></tr>').join('')+'</tbody></table></div>':'<p class="muted">暂无空间分析任务。</p>';
 $$('[data-viz-job]').forEach(b=>b.onclick=()=>run(b,async()=>{const j=await api('/api/visualization/job/'+b.dataset.vizJob);if(j.status!=='completed')throw Error(j.error||j.stage);const pair=j.kind.startsWith('pair'),sample=j.kind==='sample_projection';page(pair?'pair-analysis':sample?'predict':'spaces');const target=pair?(j.kind==='pair_summary'?'#pair-summary-output':'#pair-projection-output'):sample?'#sample-viz-output':j.kind==='cluster_summary'?'#cluster-output':'#viz-output';await waitViz(j.id,pair?'#pair-progress':sample?'#sample-viz-progress':'#viz-progress',target);}));
}
$('#viz-save').onclick=e=>run(e.target,saveVizSettings);
$('#viz-build').onclick=e=>run(e.target,async()=>{await saveVizSettings();$('#cluster-output').innerHTML='';$('#viz-summarize').disabled=true;return startViz({kind:'atlas',mode:mode(),scope:$('#viz-scope').value},'#viz-progress','#viz-output');});
$('#viz-summarize').onclick=e=>run(e.target,()=>startViz({kind:'cluster_summary',source_job:atlasJob},'#viz-progress','#cluster-output'));
$('#pair-summarize').onclick=e=>run(e.target,()=>startViz({kind:'pair_summary',source_job:pairJob},'#pair-progress','#pair-summary-output'));
$('#viz-history-refresh').onclick=e=>run(e.target,refreshVizHistory);
$('#viz-redraw').onclick=()=>Object.entries(vizDisplayed).forEach(([target,result])=>drawSpaces(result,target));
$('#viz-config').onchange=()=>{try{const c=JSON.parse($('#viz-config').value);$('#viz-neighbors').value=c.n_neighbors;$('#viz-min-dist').value=c.min_dist;$('#viz-method').value=c.cluster_method;$('#viz-clusters').value=c.n_clusters;$('#viz-representatives').value=c.representatives;}catch(e){error('配置 JSON 无效。');}};
function trialPlot(trials,direction){
 const ok=trials.filter(t=>t.state==='COMPLETE'&&Number.isFinite(t.value));if(!ok.length)return '<p>无已完成试验。</p>';
 const w=600,h=240,L=65,T=20,B=35,maxN=Math.max(...ok.map(t=>t.number),1),lo=Math.min(...ok.map(t=>t.value)),hi=Math.max(...ok.map(t=>t.value)),span=hi-lo||1;
 const x=t=>L+t.number/maxN*(w-L-20),y=t=>h-B-(t.value-lo)/span*(h-B-T);
 const ticks=Array.from({length:5},(_,i)=>{const yy=h-B-i/4*(h-B-T),xx=L+i/4*(w-L-20);return '<line x1="'+L+'" y1="'+yy+'" x2="'+(w-20)+'" y2="'+yy+'" stroke="#edf2f4"/><text x="'+(L-8)+'" y="'+(yy+3)+'" text-anchor="end" font-size="10">'+(lo+i/4*span).toFixed(4)+'</text><text x="'+xx+'" y="'+(h-B+15)+'" text-anchor="middle" font-size="10">'+(i/4*maxN).toFixed(maxN<4?1:0)+'</text>';}).join('');
 return '<svg viewBox="0 0 '+w+' '+h+'">'+ticks+'<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(h-B)+'" stroke="#a9bec8"/><line x1="'+L+'" y1="'+(h-B)+'" x2="'+(w-20)+'" y2="'+(h-B)+'" stroke="#a9bec8"/>'+ok.map(t=>'<circle cx="'+x(t)+'" cy="'+y(t)+'" r="5" fill="#168b89"><title>Trial '+t.number+' · '+t.value+'</title></circle>').join('')+'<text x="'+w/2+'" y="'+(h-8)+'" text-anchor="middle">Trial · '+direction+'</text></svg>';
}
async function refreshOptimization(){
 const d=await api('/api/optimization/config');$('#opt-config').value=JSON.stringify(d.config,null,2);
 if(!d.last){$('#opt-result').innerHTML='<p class="muted">尚无完成的优化运行。试验进度见任务日志。</p>';return;}
 $('#opt-result').innerHTML='<p class="subnote">'+esc(d.last.run_dir)+'</p>'+Object.entries(d.last.summary.studies).map(([stage,s])=>'<article><h4>'+esc(stage)+' · 最佳 Trial '+s.best_trial+' · '+s.best_value.toFixed(5)+'</h4><p class="subnote">'+esc(s.objective)+'</p>'+trialPlot(s.trials||[],s.direction)+'<div class="scroll-table">'+table(s.trials||[])+'</div></article>').join('')+downloads(d.last.downloads);
}
async function saveOptimization(){await api('/api/optimization/config',JSON.parse($('#opt-config').value));}
$('#opt-save').onclick=e=>run(e.target,saveOptimization);
$('#opt-refresh').onclick=e=>run(e.target,refreshOptimization);
$('#opt-start').onclick=e=>run(e.target,async()=>{await saveTrainingForm();await saveOptimization();await api('/api/job',{kind:'optimize'});await jobs();});
loadVizSettings().catch(e=>error(e.message));refreshVizHistory().catch(e=>error(e.message));refreshOptimization().catch(e=>error(e.message));

