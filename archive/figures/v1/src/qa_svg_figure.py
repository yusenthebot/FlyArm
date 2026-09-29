"""Render-truth QA gate for paper-style SVG figures built with figkit.

The SVG is rendered in headless Chrome/Chromium/Edge and measured with the
browser's own text layout, so width estimates in the build script never decide
acceptance. Reported checks:

- overflow:   text escaping its container (text[data-in] -> [data-box]) or the canvas
- collide:    text/text bounding-box collisions
- boxOverlap: sibling cards/chips/thumbnails that partially overlap
- textCovered: a label hidden under a card or chip drawn after it
- lineText:   connector or curve strokes crossing a label that no opaque,
              later-drawn container covers
- smallText:  labels that would print below --min-pt (6 pt) at the figure's print
              width (the SVG's data-print-width-pt, else 516 pt text width for
              canvases at least 1000 px wide and 252 pt column width below);
              math scripts inside a label are exempt
- longText:   labels longer than --max-words words; labels boxed in an `example`
              card or a speech bubble (prompts, code, reasoning traces) are exempt
- words:      label words outside example content, which must stay within
              --words-per-10k words per 10,000 px^2 of canvas
- coverage:   share of the canvas covered by cards, chips, thumbnails and text
- framed:     share covered by stage panels or content
- tidy:       geometry report, failing only with --strict-tidy: connector ends that
              stop short of a box edge or run inside one (edgeGap), wires crossing a card
              they do not attach to (edgeThroughBox), connectors that overlap collinearly
              (edgeOverlap), peer boxes whose edges or centers nearly line up but miss
              (misalign), centered chip labels that sit off-center (offCenter), labels
              pressed within 3 px of a shape or stroke they do not sit in (crowded), and
              drawings or icons that belong to a card but cross its edge (sketchOverflow), and
              cards, chips or wire labels that sit half inside a panel (straddle).
              Reported but never failing: connector crossings, uneven gaps in a run of peers
              (gapUneven), the median text share of cards, and cards that are almost empty.
              Peers are boxes of the same kind in the same container, so nested groups and
              separate columns are not compared with each other

Sizes and word budgets come from measured ICRA / IROS / RSS method figures; see
references/paper-figure-study.md. Exit code 1 when any check fails.

Usage:
    python qa_svg_figure.py figure.svg [--png] [--json] [--min-coverage 0.4] [--min-pt 6]
                            [--max-words 6] [--words-per-10k 1.0]
Set CHROME_PATH when the browser is not auto-detected.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKS = ("overflow", "collide", "boxOverlap", "textCovered", "lineText", "smallText", "longText")

_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge")

JS = r"""
const svg=document.querySelector('svg');
const W=svg.viewBox.baseVal.width,H=svg.viewBox.baseVal.height;
const R=b=>({x:b.x,y:b.y,w:b.width,h:b.height});
const boxes={};
svg.querySelectorAll('[data-box]').forEach(e=>{boxes[e.dataset.box]=Object.assign(R(e.getBBox()),
  {kind:e.dataset.kind,id:e.dataset.box,qa:e.dataset.qa||''});});
const overflow=[],texts=[],textEls=[],smallText=[],longText=[];
const PRINT=parseFloat(svg.dataset.printWidthPt)||__PRINT_PT__||(W>=1000?516:252);
const MIN_FONT=__MIN_PX__>0?__MIN_PX__*W/1400:__MIN_PT__*W/PRINT;
const wordCount=s=>s.split(/\s+/).reduce((n,tok)=>{const cjk=(tok.match(/[\u3040-\u30ff\u3400-\u9fff]/g)||[]).length;
  return n+(cjk?Math.ceil(cjk/2):0)+(/[A-Za-z0-9]/.test(tok.replace(/[\u3040-\u30ff\u3400-\u9fff]/g,''))?1:0);},0);
let words=0;
svg.querySelectorAll('text').forEach(t=>{
  const b=R(t.getBBox()); if(!b.w) return; b.s=t.textContent.slice(0,40); texts.push(b); textEls.push(t);
  const fs=parseFloat(getComputedStyle(t).fontSize); if(fs<MIN_FONT-0.01) smallText.push({s:b.s,size:fs});
  const home=t.dataset.in?svg.querySelector('[data-box="'+t.dataset.in+'"]'):null;
  if(!(home&&home.dataset.kind==='example')&&!t.closest('symbol,defs')){const n=wordCount(t.textContent); words+=n;
    if(n>__MAX_WORDS__) longText.push({s:t.textContent.slice(0,60),words:n});}
  if(b.x<0||b.y<0||b.x+b.w>W||b.y+b.h>H) overflow.push({s:b.s,box:'canvas'});
  const B=boxes[t.dataset.in]; if(!B) return;
  const l=b.x-B.x,r=B.x+B.w-(b.x+b.w),tp=b.y-B.y,bt=B.y+B.h-(b.y+b.h);
  if(l<1.5||r<1.5||tp<-1||bt<-2) overflow.push({s:b.s,box:B.kind,left:Math.round(l),right:Math.round(r),top:Math.round(tp),bottom:Math.round(bt)});
});
const collide=[];
for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
  const a=texts[i],c=texts[j];
  const ix=Math.min(a.x+a.w,c.x+c.w)-Math.max(a.x,c.x), iy=Math.min(a.y+a.h,c.y+c.h)-Math.max(a.y,c.y);
  if(ix>1.5&&iy>4) collide.push([a.s,c.s,Math.round(ix),Math.round(iy)]);
}
const solid=Object.values(boxes).filter(b=>b.kind!=='panel');
const inside=(a,b)=>a.x>=b.x-0.5&&a.y>=b.y-0.5&&a.x+a.w<=b.x+b.w+0.5&&a.y+a.h<=b.y+b.h+0.5;
const boxOverlap=[];
for(let i=0;i<solid.length;i++)for(let j=i+1;j<solid.length;j++){
  const a=solid[i],c=solid[j]; if(inside(a,c)||inside(c,a)) continue;
  const ix=Math.min(a.x+a.w,c.x+c.w)-Math.max(a.x,c.x), iy=Math.min(a.y+a.h,c.y+c.h)-Math.max(a.y,c.y);
  if(ix>0.5&&iy>0.5) boxOverlap.push([a.kind+':'+[a.x,a.y].map(Math.round),c.kind+':'+[c.x,c.y].map(Math.round)]);
}
// a label hidden under a later opaque card or chip: the gate must catch what the eye catches
const textCovered=[];
textEls.forEach((t,i)=>{const b=texts[i]; if(!b.w) return;
  Object.values(boxes).forEach(B=>{ if(B.kind==='panel'||B.id===t.dataset.in) return;
    const el=svg.querySelector('[data-box="'+B.id+'"]');
    if(!el||getComputedStyle(el).fill==='none') return;
    if(!(t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING)) return;
    if(el.contains(t)) return;
    const ix=Math.min(b.x+b.w,B.x+B.w)-Math.max(b.x,B.x), iy=Math.min(b.y+b.h,B.y+B.h)-Math.max(b.y,B.y);
    if(ix>1.5&&iy>b.h*0.45)
      textCovered.push({s:b.s,box:B.kind+':'+Math.round(B.x)+','+Math.round(B.y),
                        share:+((ix*iy)/(b.w*b.h)).toFixed(2)});});});
const lineText=[];
const strokes=[...svg.querySelectorAll('path,line,polyline')].filter(e=>!e.closest('symbol,marker,clipPath,defs,[data-qa=ignore]')&&getComputedStyle(e).stroke!=='none');
strokes.forEach(e=>{
  let L; try{L=e.getTotalLength();}catch(_){return;} if(!L) return;
  const hits=new Set();
  for(let d=0;d<=L;d+=2){const p=e.getPointAtLength(d);
    texts.forEach((b,i)=>{if(p.x>b.x+1&&p.x<b.x+b.w-1&&p.y>b.y+b.h*0.22&&p.y<b.y+b.h*0.86) hits.add(i);});}
  hits.forEach(i=>{
    const t=textEls[i]; const cont=t.dataset.in?svg.querySelector('[data-box="'+t.dataset.in+'"]'):null;
    if(cont&&getComputedStyle(cont).fill!=='none'&&(e.compareDocumentPosition(cont)&Node.DOCUMENT_POSITION_FOLLOWING)) return;
    lineText.push([texts[i].s,(e.getAttribute('d')||e.tagName).slice(0,34)]);
  });
});
let cov=0,pcov=0,n=0; const panels=Object.values(boxes).filter(b=>b.kind==='panel');
const hit=(arr,x,y)=>arr.some(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);
for(let y=2;y<H;y+=4)for(let x=2;x<W;x+=4){n++; if(hit(solid,x,y)||hit(texts,x,y)) cov++; if(hit(panels,x,y)||hit(solid,x,y)) pcov++;}
// ---- tidiness: connector anchoring, connector overlap and crossings, alignment, gaps, centering, text fill
const conns=[...svg.querySelectorAll('path,line,polyline')].filter(e=>!e.closest('symbol,marker,clipPath,defs,[data-qa=ignore]')
  &&getComputedStyle(e).stroke!=='none'&&(e.getAttribute('marker-end')||e.getAttribute('marker-start')));
const polys=conns.map(e=>{let L=0; try{L=e.getTotalLength();}catch(_){return null;} if(!L) return null;
  const pts=[],step=Math.max(2,L/80); for(let d=0;d<L;d+=step){const p=e.getPointAtLength(d); pts.push([p.x,p.y]);}
  const pe=e.getPointAtLength(L); pts.push([pe.x,pe.y]);
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const bb={x:Math.min(...xs),y:Math.min(...ys),w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)};
  return {el:e,pts,bb,d:(e.getAttribute('d')||e.tagName).slice(0,30)};}).filter(Boolean);
const outDist=(b,x,y)=>Math.hypot(Math.max(b.x-x,0,x-(b.x+b.w)),Math.max(b.y-y,0,y-(b.y+b.h)));
const pen=(b,x,y)=>(x<b.x||x>b.x+b.w||y<b.y||y>b.y+b.h)?0:Math.min(x-b.x,b.x+b.w-x,y-b.y,b.y+b.h-y);
// the smallest box that encloses another is its container; a wire drawn inside its own container is normal
const encloses=(o,b)=>b.x>=o.x-0.5&&b.y>=o.y-0.5&&b.x+b.w<=o.x+o.w+0.5&&b.y+b.h<=o.y+o.h+0.5&&o.w*o.h>b.w*b.h;
Object.values(boxes).forEach(b=>{let p=null;
  Object.values(boxes).forEach(o=>{if(o!==b&&encloses(o,b)&&(!p||o.w*o.h<p.w*p.h)) p=o;});
  b.parent=p?p.id:'canvas';});
const wires=strokes.map(e=>{let L=0; try{L=e.getTotalLength();}catch(_){return null;} if(!L) return null;
  const pts=[],step=3; for(let d=0;d<=L;d+=step){const q=e.getPointAtLength(d); pts.push([q.x,q.y]);}
  return {el:e,pts};}).filter(Boolean);
const unit=(a,b)=>{const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1; return [dx/L,dy/L];};
// a T junction on another wire is a fork (a bus stub, a merge stem); lying along one is not
const forkOnWire=(p,dir,own)=>wires.some(w=>{ if(w.el===own) return false;
  for(let i=0;i<w.pts.length;i++){ const q=w.pts[i]; if(Math.hypot(q[0]-p[0],q[1]-p[1])>2.5) continue;
    const j=i?i-1:Math.min(1,w.pts.length-1), k=i?Math.min(i+1,w.pts.length-1):0;
    const wd=unit(w.pts[j],w.pts[k]);
    if(Math.abs(wd[0]*dir[0]+wd[1]*dir[1])<0.87) return true;}
  return false;});
const edgeGap=[],edgeThroughBox=[];
polys.forEach(c=>{
  const last=c.pts.length-1;
  const ends=[[c.pts[0],'start',unit(c.pts[0],c.pts[Math.min(1,last)])],
              [c.pts[last],'end',unit(c.pts[Math.max(0,last-1)],c.pts[last])]];
  const obst=solid.filter(b=>!encloses(b,c.bb));
  ends.forEach(([p,which,dir])=>{
    let near=1e9,deep=0,anchored=false;
    solid.forEach(b=>{const o=outDist(b,p[0],p[1]),d=pen(b,p[0],p[1]); if(o<=3.5&&d<=3.5) anchored=true;});
    if(!anchored&&forkOnWire(p,dir,c.el)) anchored=true;
    obst.forEach(b=>{near=Math.min(near,outDist(b,p[0],p[1])); deep=Math.max(deep,pen(b,p[0],p[1]));});
    if(deep>3.5&&!anchored) edgeGap.push({end:which,issue:'inside a box',by:+deep.toFixed(1),path:c.d});
    else if(!anchored&&near>2.5&&near<=14) edgeGap.push({end:which,issue:'short of the edge',gap:+near.toFixed(1),path:c.d});
  });
  const attached=b=>ends.some(([p])=>outDist(b,p[0],p[1])<=3.5||pen(b,p[0],p[1])>0);
  const covers=b=>{const el=svg.querySelector('[data-box="'+b.id+'"]');
    return el&&getComputedStyle(el).fill!=='none'&&(c.el.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING);};
  obst.filter(b=>!attached(b)&&!covers(b)).forEach(b=>{let worst=0;
    c.pts.forEach(p=>{worst=Math.max(worst,pen(b,p[0],p[1]));});
    if(worst>2.5) edgeThroughBox.push({box:b.kind+':'+[Math.round(b.x),Math.round(b.y)],by:+worst.toFixed(1),path:c.d});});
});
const segsOf=c=>{const out=[]; for(let i=1;i<c.pts.length;i++){const a=c.pts[i-1],b=c.pts[i]; if(Math.hypot(b[0]-a[0],b[1]-a[1])>0.4) out.push([a,b]);} return out;};
const allSegs=polys.map(segsOf);
const cross=(p,q,r,s2)=>{const d=(q[0]-p[0])*(s2[1]-r[1])-(q[1]-p[1])*(s2[0]-r[0]); if(Math.abs(d)<1e-9) return false;
  const t=((r[0]-p[0])*(s2[1]-r[1])-(r[1]-p[1])*(s2[0]-r[0]))/d, u=((r[0]-p[0])*(q[1]-p[1])-(r[1]-p[1])*(q[0]-p[0]))/d;
  return t>0.02&&t<0.98&&u>0.02&&u<0.98;};
let crossings=0; const edgeOverlap=[];
for(let i=0;i<allSegs.length;i++)for(let j=i+1;j<allSegs.length;j++){
  let hit=false,ov=0;
  allSegs[i].forEach(a=>allSegs[j].forEach(b=>{
    if(cross(a[0],a[1],b[0],b[1])) hit=true;
    const ah=Math.abs(a[0][1]-a[1][1])<0.6, bh=Math.abs(b[0][1]-b[1][1])<0.6;
    const av=Math.abs(a[0][0]-a[1][0])<0.6, bv=Math.abs(b[0][0]-b[1][0])<0.6;
    if(ah&&bh&&Math.abs(a[0][1]-b[0][1])<1.5) ov+=Math.max(0,Math.min(Math.max(a[0][0],a[1][0]),Math.max(b[0][0],b[1][0]))-Math.max(Math.min(a[0][0],a[1][0]),Math.min(b[0][0],b[1][0])));
    if(av&&bv&&Math.abs(a[0][0]-b[0][0])<1.5) ov+=Math.max(0,Math.min(Math.max(a[0][1],a[1][1]),Math.max(b[0][1],b[1][1]))-Math.max(Math.min(a[0][1],a[1][1]),Math.min(b[0][1],b[1][1])));
  }));
  if(hit) crossings++;
  if(ov>8) edgeOverlap.push({a:polys[i].d,b:polys[j].d,overlap:Math.round(ov)});
}
const clusters=(vals,tol)=>{const v=[...vals].sort((a,b)=>a-b),out=[]; let cur=[v[0]];
  for(let i=1;i<v.length;i++){ if(v[i]-cur[cur.length-1]<=tol) cur.push(v[i]); else {out.push(cur); cur=[v[i]];}} if(v.length) out.push(cur); return out;};
// peers are boxes of the same kind in the same container; only peers are expected to line up or share gaps
const peers={}; solid.forEach(b=>{const k=b.parent+'|'+b.kind; (peers[k]=peers[k]||[]).push(b);});
const misalign=[];
const AXES={x:[['left',b=>b.x],['right',b=>b.x+b.w],['center x',b=>b.x+b.w/2]],
            y:[['top',b=>b.y],['bottom',b=>b.y+b.h],['center y',b=>b.y+b.h/2]]};
Object.values(peers).filter(bs=>bs.length>1).forEach(bs=>{
  Object.values(AXES).forEach(dir=>dir.forEach(([axis,fn])=>clusters(bs.map(fn),4).forEach(c=>{
     const spread=c[c.length-1]-c[0];
     if(c.length<2||spread<=0.6) return;
     const group=bs.filter(b=>fn(b)>=c[0]-0.01&&fn(b)<=c[c.length-1]+0.01);
     // boxes already aligned on another edge of the same direction (objects of different heights resting on
     // one line, chips of different widths sharing a left edge) are aligned, not nearly aligned
     if(dir.some(([other,g])=>other!==axis&&Math.max(...group.map(g))-Math.min(...group.map(g))<=0.6)) return;
     misalign.push({axis,at:+c[0].toFixed(1),spread:+spread.toFixed(1),boxes:c.length,
       which:group.map(b=>b.kind+':'+Math.round(b.x)+','+Math.round(b.y)).slice(0,4)});})));});
const gapUneven=[];
[['row',b=>b.y+b.h/2,b=>b.x,b=>b.x+b.w],['column',b=>b.x+b.w/2,b=>b.y,b=>b.y+b.h]].forEach(([kind,key,lo,hi])=>{
  const groups={}; solid.forEach(b=>{const k=b.parent+'|'+b.kind+'|'+Math.round(key(b)/4)*4; (groups[k]=groups[k]||[]).push(b);});
  Object.values(groups).forEach(g=>{ if(g.length<3) return; const bs=g.slice().sort((a,b)=>lo(a)-lo(b));
    const gaps=[]; for(let i=1;i<bs.length;i++) gaps.push(+(lo(bs[i])-hi(bs[i-1])).toFixed(1));
    if(gaps.some(v=>v<0)) return;
    const med=[...gaps].sort((a,b)=>a-b)[Math.floor(gaps.length/2)];
    let run=[gaps[0]]; const runs=[run];                      // a gap far above the median ends the run
    for(let i=1;i<gaps.length;i++){ if(gaps[i]>2.5*med||gaps[i]*2.5<med){run=[gaps[i]]; runs.push(run);} else run.push(gaps[i]); }
    runs.filter(r=>r.length>1).forEach(r=>{ const span=Math.max(...r)-Math.min(...r);
      if(span>2) gapUneven.push({kind,at:Math.round(key(bs[0])),gaps:r});});});
});
// a label pressed against a drawing: every shape or stroke keeps clear space around text it does not hold
// measured on the ink, not the line box: a serif line box is about 1.5 em tall, its lowercase ink about half that
const CLEAR=3, crowded=[];
const ink2d=document.createElement('canvas').getContext('2d');
const inkBox=(t,b)=>{const cs=getComputedStyle(t), y=parseFloat(t.getAttribute('y'));
  if(!isFinite(y)) return b;
  ink2d.font=`${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const m=ink2d.measureText(t.textContent);
  const top=Math.max(b.y,y-m.actualBoundingBoxAscent), bot=Math.min(b.y+b.h,y+m.actualBoundingBoxDescent);
  return bot>top?{x:b.x,y:top,w:b.w,h:bot-top}:b;};
const gapOf=(a,b)=>Math.hypot(Math.max(0,a.x-(b.x+b.w),b.x-(a.x+a.w)),Math.max(0,a.y-(b.y+b.h),b.y-(a.y+a.h)));
const holds=(o,t)=>t.x>=o.x-0.5&&t.y>=o.y-0.5&&t.x+t.w<=o.x+o.w+0.5&&t.y+t.h<=o.y+o.h+0.5;
const shapes=[...svg.querySelectorAll('rect,circle,ellipse,polygon')]
  .filter(e=>!e.closest('symbol,marker,clipPath,defs,[data-qa=ignore]'))
  .map(e=>({el:e,b:R(e.getBBox())})).filter(q=>q.b.w>0&&q.b.h>0&&q.b.w<W*0.95);
textEls.forEach((t,i)=>{const line=texts[i]; if(!line.w) return; let worst=null; const b=inkBox(t,line);
  shapes.forEach(q=>{ if(q.el.dataset.box&&q.el.dataset.box===t.dataset.in) return; if(holds(q.b,b)) return;
    const d=gapOf(b,q.b); if(d<CLEAR&&(!worst||d<worst.d))
      worst={d,what:q.el.tagName+':'+[q.b.x,q.b.y,q.b.w,q.b.h].map(Math.round).join(',')};});
  wires.forEach(w=>{ let d=1e9; w.pts.forEach(p=>{d=Math.min(d,gapOf(b,{x:p[0],y:p[1],w:0,h:0}));});
    if(d>0&&d<CLEAR&&(!worst||d<worst.d)) worst={d,what:w.el.tagName+':'+(w.el.getAttribute('d')||'').slice(0,24)};});
  if(worst) crowded.push({s:line.s,at:[b.x,b.y,b.w,b.h].map(Math.round),against:worst.what,gap:+worst.d.toFixed(1)});});
// a drawing that belongs to a card but crosses its edge (curves, glyphs, icons, bars hanging out of their card)
const sketchOverflow=[];
const holders=Object.values(boxes).filter(b=>b.kind!=='icon').sort((a,b)=>a.w*a.h-b.w*b.h);
const drawings=[...svg.querySelectorAll('path,line,polyline,rect,circle,ellipse,polygon,[data-kind=icon]')]
  .filter(e=>!e.closest('symbol,marker,clipPath,defs,[data-qa=ignore]')&&!(e.dataset.box&&e.dataset.kind!=='icon')
    &&!e.getAttribute('marker-end')&&!e.getAttribute('marker-start'))
  .map(e=>({el:e,b:R(e.getBBox())})).filter(q=>q.b.w*q.b.h>4&&q.b.w<W*0.95);
drawings.forEach(q=>{const a=q.b.w*q.b.h||1;
  const home=holders.find(o=>{const ix=Math.min(o.x+o.w,q.b.x+q.b.w)-Math.max(o.x,q.b.x),
    iy=Math.min(o.y+o.h,q.b.y+q.b.h)-Math.max(o.y,q.b.y); return ix>0&&iy>0&&ix*iy>=0.5*a&&o.w*o.h>=a;});
  if(!home) return;
  const by=Math.max(home.x-q.b.x,q.b.x+q.b.w-home.x-home.w,home.y-q.b.y,q.b.y+q.b.h-home.y-home.h);
  if(by>1.5) sketchOverflow.push({what:q.el.tagName+':'+[q.b.x,q.b.y,q.b.w,q.b.h].map(Math.round).join(','),
    card:home.kind+':'+Math.round(home.x)+','+Math.round(home.y),by:+by.toFixed(1)});});
// a card, chip or wire label that sits half in a panel and half outside it
const straddle=[];
const panelsAll=Object.values(boxes).filter(b=>b.kind==='panel');
solid.filter(b=>b.qa!=='ignore').forEach(b=>{panelsAll.forEach(P=>{
  const ix=Math.min(b.x+b.w,P.x+P.w)-Math.max(b.x,P.x), iy=Math.min(b.y+b.h,P.y+P.h)-Math.max(b.y,P.y);
  if(ix>1.5&&iy>1.5&&!inside(b,P)) straddle.push({box:b.kind+':'+[b.x,b.y,b.w,b.h].map(Math.round).join(','),
    panel:[P.x,P.y].map(Math.round).join(','),out:+Math.max(P.x-b.x,b.x+b.w-P.x-P.w,P.y-b.y,b.y+b.h-P.y-P.h).toFixed(1)});});});
const offCenter=[];
textEls.forEach((t,i)=>{ if(t.getAttribute('text-anchor')!=='middle') return; const B=boxes[t.dataset.in];
  if(!B||B.kind!=='chip'||texts.filter(x=>x!==texts[i]).length===0) return;
  const off=(texts[i].x+texts[i].w/2)-(B.x+B.w/2); if(Math.abs(off)>3) offCenter.push({s:texts[i].s,off:+off.toFixed(1)});});
const fills=[]; const emptyBoxes=[];
Object.values(boxes).filter(b=>['card','chip','example'].includes(b.kind)&&b.qa!=='ignore').forEach(b=>{
  const area=b.w*b.h; if(area<400) return;
  const inked=texts.filter(t=>t.x>=b.x-1&&t.y>=b.y-1&&t.x+t.w<=b.x+b.w+1&&t.y+t.h<=b.y+b.h+1).reduce((s2,t)=>s2+t.w*t.h,0);
  const others=Object.values(boxes).filter(o=>o!==b&&o.x>=b.x-1&&o.y>=b.y-1&&o.x+o.w<=b.x+b.w+1&&o.y+o.h<=b.y+b.h+1);
  const ratio=inked/area; fills.push(+ratio.toFixed(3));
  if(ratio<0.08&&!others.length) emptyBoxes.push({kind:b.kind,at:[Math.round(b.x),Math.round(b.y)],fill:+ratio.toFixed(3)});});
fills.sort((a,b)=>a-b);
const tidy={connectors:polys.length,edgeGap,edgeThroughBox,edgeOverlap,crossings,misalign,gapUneven,offCenter,crowded,sketchOverflow,straddle,
  textFillMedian:fills.length?fills[Math.floor(fills.length/2)]:null,emptyBoxes};
const wordBudget=Math.round(__WORDS_PER_10K__*W*H/10000);
const out={size:[W,H],texts:texts.length,printWidthPt:PRINT,minFontPx:+MIN_FONT.toFixed(1),overflow,collide,boxOverlap,textCovered,lineText,smallText,longText,
  words,wordBudget,tidy,coverage:+(cov/n).toFixed(3),framed:+(pcov/n).toFixed(3)};
const pre=document.createElement('pre'); pre.id='qa'; pre.textContent=JSON.stringify(out); document.body.appendChild(pre);
"""


def find_browser() -> str:
    env = os.environ.get("CHROME_PATH")
    if env:
        return env
    for candidate in _CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Chrome, Chromium or Edge was not found; set CHROME_PATH to the browser executable.")


def measure(svg_path: Path, browser: str, min_font: float | None = None, min_pt: float = 6.0,
            print_width_pt: float | None = None, max_words: int = 6, words_per_10k: float = 1.0) -> dict:
    """Render the SVG in headless Chrome and return the QA report.

    `min_font` is a legacy pixel minimum at 1400 px width; when given it replaces the print-size rule.
    """
    svg = svg_path.read_text(encoding="utf-8")
    script = (JS.replace("__MIN_PX__", f"{float(min_font or 0):g}").replace("__MIN_PT__", f"{float(min_pt):g}")
              .replace("__PRINT_PT__", f"{float(print_width_pt or 0):g}").replace("__MAX_WORDS__", str(int(max_words)))
              .replace("__WORDS_PER_10K__", f"{float(words_per_10k):g}"))
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "qa.html"
        page.write_text(f"<!doctype html><meta charset=utf-8><body style='margin:0'>{svg}<script>{script}</script></body>", encoding="utf-8")
        result = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--virtual-time-budget=3000", "--dump-dom", page.as_uri()],
            capture_output=True, text=True, encoding="utf-8", timeout=180, check=True,
        )
    match = re.search(r'<pre id="qa">(.*?)</pre>', result.stdout, re.DOTALL)
    if not match:
        raise RuntimeError("the QA script produced no report; check that the SVG parses in the browser")
    return json.loads(html.unescape(match.group(1)))


def render_png(svg_path: Path, size: list[float], browser: str, scale: int = 2) -> Path:
    out = svg_path.with_suffix(".png")
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
         f"--window-size={int(size[0])},{int(size[1])}", f"--screenshot={out}", svg_path.resolve().as_uri()],
        capture_output=True, timeout=180, check=True,
    )
    return out


TIDY_CHECKS = ("edgeGap", "edgeThroughBox", "edgeOverlap", "misalign", "offCenter", "crowded", "sketchOverflow",
               "straddle")
TIDY_REPORTS = ("gapUneven", "crossings", "emptyBoxes")  # judgement calls, printed but never failing


def failures(report: dict, min_coverage: float, strict_tidy: bool = False) -> list[str]:
    failed = [f"{key}={len(report[key])}" for key in CHECKS if report[key]]
    if strict_tidy:
        failed += [f"tidy.{key}={len(report['tidy'][key])}" for key in TIDY_CHECKS if report.get("tidy", {}).get(key)]
    if "words" in report and report["words"] > report["wordBudget"]:
        failed.append(f"words={report['words']} > budget {report['wordBudget']}")
    if report["coverage"] < min_coverage:
        failed.append(f"coverage={report['coverage']} < {min_coverage}")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("svg", type=Path)
    parser.add_argument("--png", action="store_true", help="also write a 2x PNG next to the SVG")
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    parser.add_argument("--min-coverage", type=float, default=0.40)
    parser.add_argument("--min-pt", type=float, default=6.0, help="smallest printed label size in points")
    parser.add_argument("--print-width-pt", type=float, default=None,
                        help="printed width of the canvas; default from the SVG, else 516 (>= 1000 px) or 252")
    parser.add_argument("--max-words", type=int, default=6, help="longest label outside example content")
    parser.add_argument("--words-per-10k", type=float, default=1.0,
                        help="label word budget per 10,000 px^2 of canvas, outside example content")
    parser.add_argument("--min-font", type=float, default=None,
                        help="legacy: smallest label size in px at 1400 px width; replaces --min-pt when given")
    parser.add_argument("--strict-tidy", action="store_true",
                        help="also fail on the tidy geometry checks (connector anchoring, alignment, gaps, centering)")
    args = parser.parse_args(argv)

    browser = find_browser()
    report = measure(args.svg, browser, args.min_font, args.min_pt, args.print_width_pt, args.max_words,
                     args.words_per_10k)
    if args.png:
        report["png"] = str(render_png(args.svg, report["size"], browser))
    failed = failures(report, args.min_coverage, args.strict_tidy)
    report["pass"] = not failed

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{args.svg.name}: size={report['size']} print={report['printWidthPt']:g}pt min={report['minFontPx']}px "
              f"texts={report['texts']} words={report['words']}/{report['wordBudget']} "
              f"coverage={report['coverage']} framed={report['framed']}")
        for key in CHECKS:
            print(f"  {key}: {len(report[key])}")
            for item in report[key][:40]:
                print("    " + json.dumps(item, ensure_ascii=False))
        tidy = report.get("tidy", {})
        if tidy:
            print(f"  tidy: {tidy['connectors']} connectors, {tidy['crossings']} crossings, "
                  f"text fill {tidy['textFillMedian']}")
            for key in TIDY_CHECKS + ("gapUneven", "emptyBoxes"):
                items = tidy.get(key) or []
                if items:
                    print(f"    {key}: {len(items)}")
                    for item in items[:10]:
                        print("      " + json.dumps(item, ensure_ascii=False))
        print("PASS" if not failed else "FAIL: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
