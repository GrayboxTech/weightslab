/* Run with Playwright installed; writes actual screenshots and request evidence. */
const {chromium} = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const output = path.resolve(process.argv[2]);
  await fs.mkdir(path.join(output, 'screenshots'), {recursive:true});
  const browser = await chromium.launch({headless:true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {})});
  const page = await browser.newPage({viewport:{width:1440,height:1200},deviceScaleFactor:1.5});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  const screenshots=[], requests=[];
  async function shot(name,caption,selector) {
    const file='screenshots/'+name+'.png';
    if(selector) await page.locator(selector).screenshot({path:path.join(output,file)});
    else await page.screenshot({path:path.join(output,file),fullPage:true});
    screenshots.push({file,caption,url:page.url(),captured_at:new Date().toISOString()});
  }
  const demoUrl=process.env.RECORDED_DEMO_URL||'http://127.0.0.1:8877/';
  const labUrl=process.env.LIVE_LAB_URL||'http://127.0.0.1:8878/';
  await page.goto(demoUrl);
  await page.waitForFunction(()=>window.DEMO_READY&&window.ATTRIBUTION_READY);
  await page.screenshot({path:path.join(output,'screenshots/01-recorded-overview.png'),fullPage:false});
  screenshots.push({file:'screenshots/01-recorded-overview.png',caption:'Recorded three-seed pilot: balanced exposure improves the rare group but regresses common cases. This screen is offline replay, not a live edit.',url:demoUrl});
  for(const name of ['examples','details','attribution','results']) {
    await page.click('#tab-'+name);
    assert.equal(await page.locator('#panel-'+name).isVisible(),true);
    assert.equal(await page.locator('.tab-panel:visible').count(),1);
  }
  await page.click('#tab-attribution');
  assert.equal(await page.inputValue('#attr-mode'),'overlay');
  await page.selectOption('#arm','widen_resample');
  await page.selectOption('#attribution-case','075.Green_Jay/Green_Jay_0102_65813.jpg');
  await page.waitForFunction(()=>window.ATTRIBUTION_READY);
  await page.locator('#control-map').hover({position:{x:180,y:180}});
  assert.match(await page.textContent('#attr-readout'),/target-logit units/);
  assert.equal(await page.locator('.attr-cursor:visible').count(),2);
  await shot('02-attribution','Shared-scale Integrated Gradients over the actual validation crop. Hover links the same image location and reports both contributions and their difference. Both predictions remain incorrect.','#attribution-section');
  const attributionCases=await page.locator('#attribution-case option').evaluateAll(options=>options.map(o=>o.value));
  let attributionViews=0;
  for(const arm of ['resample','widen','widen_resample']) for(const id of attributionCases) for(const mode of ['overlay','heatmap']) for(const detail of ['16','1']) {
    await page.selectOption('#arm',arm);await page.selectOption('#attribution-case',id);await page.selectOption('#attr-mode',mode);await page.selectOption('#attr-detail',detail);
    await page.waitForFunction(()=>window.ATTRIBUTION_READY);
    assert.equal(await page.locator('#control-map').isVisible(),true);
    assert.equal(await page.locator('#attr-opacity').isDisabled(),mode!=='overlay');
    attributionViews++;
  }
  for(const width of [390,320]) {
    await page.setViewportSize({width,height:900});
    for(const name of ['results','examples','attribution','details']) {await page.click('#tab-'+name);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}
  }
  await page.setViewportSize({width:1440,height:1200});
  await page.goto(labUrl);
  await page.waitForFunction(()=>window.LAB_READY&&window.LAST_RESULT,{},{timeout:90000});
  async function apply(values) {
    for(const [key,value] of Object.entries(values)) await page.selectOption('#'+key,String(value));
    const previous=await page.evaluate(()=>window.LAST_RESULT.request_id);
    await page.click('#apply');
    await page.waitForFunction(id=>window.LAST_RESULT&&window.LAST_RESULT.request_id!==id,previous,{timeout:90000});
    const result=await page.evaluate(()=>{const {examples,...rest}=window.LAST_RESULT;return rest;});
    for(const [key,value] of Object.entries(values)) assert.equal(String(result.request[key]),String(value));
    requests.push(result);
    return result;
  }
  let result=await apply({split:'test',initialization:'random',stage:'immediate',sampling:'original',seed:17,prune:0});
  assert.ok(result.max_reference_logit_delta>1);
  await shot('03-live-random-growth','Live request: add 16 neurons with the default random initialization before any training. Logit changes and prediction regressions expose initialization shock.');
  result=await apply({initialization:'zero_outgoing'});
  assert.ok(result.max_reference_logit_delta<=1e-5);
  assert.equal(result.comparison.overall.corrected_count,0);assert.equal(result.comparison.overall.regressed_count,0);
  await shot('04-live-preserved-growth','Live request on the same seed and test images: zero only the new outgoing columns after growth. Hidden/classifier dimensions expand while initial predictions are preserved. This is an experiment-local policy.');
  result=await apply({stage:'trained',sampling:'group_balanced'});
  assert.equal(result.shape.hidden[0],80);
  await shot('05-live-trained-added-units','Live inference from the output-preserving-growth checkpoint after 250 balanced steps. Compare against equally trained, equally sampled no-growth control.');
  result=await apply({prune:16});
  assert.equal(result.shape.hidden[0],64);assert.equal(result.shape.classifier[1],64);
  assert.ok(result.events.some(e=>e.operation==='model.prune_neurons'&&e.count===16));
  await shot('06-live-remove-added-units','Live causal removal: prune all 16 added units through WeightsLab and reevaluate the same population. Shapes contract together. This is ablation, not rollback of retained weights.');
  for(const seed of [17,29,43]) for(const prune of [0,4,8,16]) {
    const r=await apply({seed,prune});
    assert.equal(r.shape.hidden[0],80-prune);
    assert.equal(r.shape.classifier[1],80-prune);
    assert.equal(r.comparison.overall.count,5794);
  }
  const invalid=await page.evaluate(async()=>{
    const response=await fetch('/api/edit',{method:'POST',headers:{'Content-Type':'application/json','X-Lab-Token':token},body:JSON.stringify({...selection(),prune:17})});return response.status;
  });assert.equal(invalid,400);
  const unauthorized=await page.evaluate(async()=>{const response=await fetch('/api/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});return response.status;});assert.equal(unauthorized,403);
  await apply({seed:17,prune:0,stage:'immediate',sampling:'original'});
  for(const width of [390,320]) {await page.setViewportSize({width,height:900});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}
  await shot('07-live-mobile','The live controls and results remain usable at narrow mobile width.');
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(output,'screenshots.json'),JSON.stringify({screenshots},null,2)+'\n');
  await fs.writeFile(path.join(output,'browser-evidence.json'),JSON.stringify({passed:true,page_errors:errors,attribution_views:attributionViews,live_requests:requests,invalid_request_status:invalid,missing_token_status:unauthorized,mobile_widths:[390,320]},null,2)+'\n');
  console.log(JSON.stringify({screenshots:screenshots.length,live_requests:requests.length,attribution_views:attributionViews,page_errors:errors}));
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
