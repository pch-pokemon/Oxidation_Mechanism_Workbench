// Execute the actual analysis-page code with a minimal DOM fixture.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
class Element{constructor(){this.value='';this.textContent='';this.classList={toggle(){}};}set innerHTML(s){this._html=s;const m=s.match(/<option value="([^"]*)"/);if(m)this.value=m[1];}get innerHTML(){return this._html;}}
const elements={};const $=selector=>elements[selector]||(elements[selector]=new Element());
let response={runs:[],issues:[]};
const last={run_id:'run_20261013_090000_abcdef',metadata:{mode:'real',label:'本地研究模型'},reference_scope:'train',retrieval:[{sample_id:'s001',channels:[]},{sample_id:'s002',channels:[]}]};
const context={$,esc:String,api:async p=>p==='/api/analysis/runs'?response:{config:{model:'glm-4-plus',language:'zh',env_file:'',api_key_env_candidates:['API_KEY']},dependencies:{},api_key_present:false,temperature:0},lastPrediction:last,run:()=>{},error:()=>{},setTimeout:()=>0,clearTimeout:()=>{},downloads:()=>'',console};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/analysis.js'),'utf8'),context);
(async()=>{
 await vm.runInContext('refreshAnalysisRuns()',context);
 assert.equal($('#analysis-run').value,last.run_id);assert.equal($('#analysis-sample').value,'0');assert($('#analysis-sample').innerHTML.includes('s002'));
 const selected=vm.runInContext('analysisSelection()',context);assert.equal(selected.run_id,last.run_id);assert.equal(selected.sample_index,0);
 $('#analysis-sample').value='1';assert.equal(vm.runInContext('analysisSelection()',context).sample_index,1);
 response={runs:[],issues:[{run_id:'bad-run',error:'读取错误'}]};await vm.runInContext('refreshAnalysisRuns()',context);assert($('#analysis-progress').textContent.includes('读取错误'));
 console.log('PASS: current prediction populated even when historical list is empty; sample selection and diagnostics work.');
})().catch(e=>{console.error(e);process.exit(1)});
