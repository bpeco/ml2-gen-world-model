/* Primitivas SVG compartidas por arquitecturas.html y presentacion.html (las inserta paper/presentacion/armar.py).
   Requiere una constante global D = {ctx, tokens, frames} (paper/presentacion/frames.json). */
const C = {ink:'#1f1e1c', mut:'#6b6a66', line:'#cfcdc6', soft:'#f1efe9', mask:'#d6d3ca'};
const FLAT = D.tokens.flat();
const tc = id => `hsl(${Math.round((id*137.508)%360)} 42% ${48+(id%4)*8}%)`;

/* ---------- primitivas SVG ---------- */
const T=(x,y,s,o={})=>`<text x="${x}" y="${y}" font-size="${o.s||13}" text-anchor="${o.a||'middle'}" font-weight="${o.w||400}" fill="${o.c||C.ink}"${o.it?' font-style="italic"':''}>${s}</text>`;
const R=(x,y,w,h,o={})=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx??7}" fill="${o.f||'#fff'}" stroke="${o.st||C.line}" stroke-width="${o.sw||1.4}"${o.d?' stroke-dasharray="5 4"':''}/>`;
function B(x,y,w,h,lines,o={}){
  let s=R(x,y,w,h,o); const lh=o.lh||16, n=lines.length; const y0=y+h/2-(n-1)*lh/2+4.5;
  lines.forEach((L,i)=>{s+=T(x+w/2,y0+i*lh,L,{s:i?(o.s2||11.5):(o.s||13.5),w:i?400:600,c:i?C.mut:(o.tc||C.ink)})});
  return s;
}
const A=(d,o={})=>`<path d="${d}" fill="none" stroke="${o.c||C.mut}" stroke-width="${o.w||1.6}"${o.d?' stroke-dasharray="4 4"':''} marker-end="url(#${o.m||'ah'})"/>`;
const IMG=(x,y,w,h,src)=>`<image href="${src}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid slice"/>`+R(x,y,w,h,{f:'none',st:'#00000030',rx:2,sw:1});
function STACK(x,y,w,h,d=10){ // tres frames de contexto, el más reciente adelante
  return D.ctx.map((s,i)=>IMG(x+(2-i)*d,y+i*d,w,h,s)).join('');
}
function G(x,y,cs,state,acc){ // grilla 8x16 de tokens
  let s='';
  for(let i=0;i<128;i++){
    const X=x+(i%16)*cs, Y=y+Math.floor(i/16)*cs, st=state?state(i):'tok';
    const f = st==='tok'||st==='hl' ? tc(FLAT[i]) : st==='mask' ? C.mask : st==='next' ? '#fff' : C.soft;
    s+=`<rect x="${X}" y="${Y}" width="${cs-.7}" height="${cs-.7}" fill="${f}"${st==='next'?` stroke="${acc}" stroke-width="2"`:''}/>`;
  }
  return s+R(x-1,y-1,16*cs+1.3,8*cs+1.3,{f:'none',rx:2,st:C.mut,sw:1});
}
function MM(x,y,n,cs,on,acc){ // matriz de atención (fila = query)
  let s='';
  for(let r=0;r<n;r++)for(let c=0;c<n;c++) s+=`<rect x="${x+c*cs}" y="${y+r*cs}" width="${cs-1}" height="${cs-1}" rx="1" fill="${on(r,c)?acc:C.soft}" ${on(r,c)?'fill-opacity=".75"':''}/>`;
  return s;
}
let UID=0;
function uniq(html){ // ids de markers únicos por SVG (si no, un SVG oculto los anula)
  const u=++UID;
  return html.replace(/id="(ah|aa[0-9a-f]+)"/g,`id="$1_${u}"`).replace(/url\(#(ah|aa[0-9a-f]+)\)/g,`url(#$1_${u})`);
}
function svg(body,acc){
  return uniq(`<svg class="dia" viewBox="0 0 940 500"><defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.mut}"/></marker>
  <marker id="aa${acc.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${acc}"/></marker>
  </defs>${body}</svg>`);
}
const AA=(d,acc,o={})=>A(d,{...o,c:acc,m:'aa'+acc.slice(1)});


function diaPortada(){
  const acc='#3d3d3a'; let s='';
  s+=STACK(10,60,170,85,12)+T(105,192,'cámara · 20 FPS · 128 × 256',{s:12,c:C.mut});
  s+=A('M216 112 H246')+B(248,82,110,60,['Encoder VQ','3,8 ms']);
  s+=A('M358 112 H380')+G(382,84,7)+T(438,158,'8 × 16 tokens',{s:12,c:C.mut});
  s+=A('M496 112 H520');
  s+=R(522,22,250,180,{f:'#faf9f5',st:C.ink,sw:1.6,rx:12})+T(647,46,'World model',{s:15,w:700});
  [['gpt2m (referencia)','#3d3d3a'],['GPT-S','#2a78d6'],['MaskGIT-S','#eb6834'],['Genie-S / M','#1baf7a'],['ConvLSTM (píxeles)','#6250d6'],['Baselines','#eda100']].forEach(([n,c],k)=>{
    s+=`<rect x="${540}" y="${60+k*22}" width="12" height="12" rx="3" fill="${c}"/>`+T(560,71+k*22,n,{a:'start',s:13});});
  s+=A('M772 112 H794')+G(796,84,7,i=>'tok')+T(852,158,'frame t + 1',{s:12,c:C.mut});
  s+=B(796,192,112,52,['Decoder VQ','6,1 ms']);
  s+=A('M852 168 V190');
  s+=AA('M796 218 H200 V196','#b3261e',{d:true});
  s+=T(470,236,'el frame generado se suma al contexto: 60 frames (3 s) en lazo cerrado',{s:12.5,c:'#b3261e',w:600});
  return `<svg class="dia" viewBox="0 0 940 250"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.mut}"/></marker><marker id="aab3261e" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#b3261e"/></marker></defs>${s}</svg>`.replace(/id="(ah|aa[0-9a-f]+)"/g,'id="$1_p"').replace(/url\(#(ah|aa[0-9a-f]+)\)/g,'url(#$1_p)');
}

