const byId = id => document.getElementById(id);
const paint = byId('paint');
const ctx = paint.getContext('2d');
const guide = byId('guide').getContext('2d');
const overlay = byId('overlay');
const video = byId('video');
const notice = byId('notice');
const camera = byId('camera');
const history = [];
let mask;
let brush = 'hard', stroke, dirty = true, hasPaint = false;
const W = paint.width, H = paint.height;
function snapshot() {
  if (history.length === 20) history.shift();
  history.push({ pixels: ctx.getImageData(0,0,W,H), hasPaint });
  byId('undo').disabled = false;
}
function changed() {
  dirty = true;
  byId('clear').disabled = !hasPaint;
}
byId('undo').onclick = () => {
  if (stroke || !history.length) return;
  const saved = history.pop();
  ctx.putImageData(saved.pixels,0,0);
  hasPaint = saved.hasPaint;
  byId('undo').disabled = !history.length;
  changed();
};
byId('clear').onclick = () => {
  if (stroke || !hasPaint) return;
  snapshot();
  ctx.clearRect(0,0,W,H);
  hasPaint = false;
  changed();
};
for (const button of document.querySelectorAll('[data-brush]')) {
  button.onclick = () => {
    brush = button.dataset.brush;
    for (const choice of document.querySelectorAll('[data-brush]'))
      choice.setAttribute('aria-pressed', String(choice === button));
  };
}
function point(event) {
  const bounds = paint.getBoundingClientRect();
  return [(event.clientX-bounds.left)*W/bounds.width,(event.clientY-bounds.top)*H/bounds.height];
}
function stamp(x,y) {
  const { radius, color, mode } = stroke;
  ctx.save();
  ctx.clip(mask);
  ctx.globalCompositeOperation = mode === 'erase' ? 'destination-out' : 'source-over';
  ctx.fillStyle = color;
  if (mode === 'soft') {
    const gradient = ctx.createRadialGradient(x,y,0,x,y,radius);
    gradient.addColorStop(0,color+'80');
    gradient.addColorStop(1,color+'00');
    ctx.fillStyle = gradient;
  }
  ctx.beginPath();
  ctx.arc(x,y,radius,0,Math.PI*2);
  ctx.fill();
  ctx.restore();
}
paint.onpointerdown = event => {
  if (stroke || event.button !== 0 || !mask) return;
  const [x,y] = point(event);
  if (!ctx.isPointInPath(mask,x,y)) return;
  snapshot();
  paint.setPointerCapture(event.pointerId);
  stroke = { id: event.pointerId, x, y, mode: brush, radius: Number(byId('size').value)/2, color: byId('color').value };
  stamp(x,y);
  if (brush !== 'erase') hasPaint = true;
  changed();
};
paint.onpointermove = event => {
  if (stroke?.id !== event.pointerId) return;
  const [x,y] = point(event);
  const dx = x-stroke.x, dy = y-stroke.y;
  const steps = Math.ceil(Math.hypot(dx,dy)/Math.max(0.75,stroke.radius/5));
  for (let i=1;i<=steps;i++) stamp(stroke.x+dx*i/steps,stroke.y+dy*i/steps);
  stroke.x=x;
  stroke.y=y;
  changed();
};
for (const name of ['pointerup','pointercancel','lostpointercapture'])
  paint.addEventListener(name,event => { if (stroke?.id === event.pointerId) stroke = undefined; });

const outline = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
let mesh, detector, renderer, stream, frame, request = 0;
const setNotice = message => { if (notice.textContent !== message) notice.textContent = message; };
const cdn = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
function path(points, indices, close = true) {
  guide.beginPath();
  for (const [order,index] of indices.entries()) {
    const [x,y] = points[index];
    if (order === 0) guide.moveTo(x,y); else guide.lineTo(x,y);
  }
  if (close) guide.closePath();
}
function drawGuide(points) {
  guide.clearRect(0,0,W,H);
  guide.lineWidth = 2;
  guide.strokeStyle = '#886550';
  guide.fillStyle = '#fae9d4';
  path(points,outline);
  guide.fill();
  guide.stroke();
  for (const contour of [
    [33,160,158,133,153,144],[362,385,387,263,373,380],
    [61,40,37,0,267,270,291,321,314,17,84,91],
    [168,6,1,2],[98,97,2,326,327],
    [70,63,105,66,107],[336,296,334,293,300]
  ]) {
    path(points,contour,contour.length>6 || contour.length===6);
    guide.stroke();
  }
}
guide.fillStyle='#fae9d4';
guide.beginPath();
guide.ellipse(W/2,H/2,149,188,0,0,Math.PI*2);
guide.fill();
let meshReady;
function loadMesh() {
  meshReady ??= (async () => {
  const response = await fetch('https://cdn.jsdelivr.net/gh/google-ai-edge/mediapipe@v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj');
  if (!response.ok) throw Error('臉部底圖載入失敗');
  const points=[], triangles=[];
  for (const line of (await response.text()).split(/\r?\n/)) {
    const fields=line.trim().split(/\s+/);
    if (fields[0]==='v') points.push([W/2-Number(fields[1])*20, H/2-(Number(fields[2])+0.8)*20]);
    if (fields[0]==='f') triangles.push(fields.slice(1).map(field=>Number(field.split('/')[0])-1));
  }
  if (points.length!==468 || !triangles.length || triangles.some(face=>face.length!==3 || face.some(i=>!points[i]))) throw Error('臉部網格資料不完整');
  const shape = new Path2D();
  for (const [order,index] of outline.entries()) {
    if (order === 0) shape.moveTo(...points[index]); else shape.lineTo(...points[index]);
  }
  shape.closePath();
  mask = shape;
  drawGuide(points);
  return { points, triangles };
  })().catch(error => { meshReady = undefined; throw error; });
  return meshReady;
}
loadMesh().catch(() => setNotice('臉部底圖載入失敗。點「開啟鏡頭」可重試。'));

function createRenderer() {
  const gl=overlay.getContext('webgl',{alpha:true,premultipliedAlpha:false,preserveDrawingBuffer:true});
  if (!gl) throw Error('此瀏覽器無法啟用 WebGL');
  function shader(type,source) {
    const id=gl.createShader(type);
    gl.shaderSource(id,source);
    gl.compileShader(id);
    if (!gl.getShaderParameter(id,gl.COMPILE_STATUS)) throw Error('臉部貼圖初始化失敗');
    return id;
  }
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec2 p;attribute vec2 t;varying vec2 uv;void main(){gl_Position=vec4(p,0.,1.);uv=t;}'));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec2 uv;uniform sampler2D ink;void main(){gl_FragColor=texture2D(ink,uv);}'));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error('臉部貼圖初始化失敗');
  gl.useProgram(program);
  const buffer=gl.createBuffer(), texture=gl.createTexture();
  gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  for (const [attribute,offset] of [['p',0],['t',8]]) {
    const location=gl.getAttribLocation(program,attribute);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location,2,gl.FLOAT,false,16,offset);
  }
  gl.bindTexture(gl.TEXTURE_2D,texture);
  for (const parameter of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,parameter,gl.CLAMP_TO_EDGE);
  for (const parameter of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D,parameter,gl.LINEAR);
  const vertices = new Float32Array(mesh.triangles.length*12);
  const area = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const signedArea = mesh.triangles.reduce((sum,[a,b,c])=>sum+area(mesh.points[a],mesh.points[b],mesh.points[c]),0);
  const winding = Math.sign(signedArea);
  const fronts = mesh.triangles.filter(([a,b,c])=>area(mesh.points[a],mesh.points[b],mesh.points[c])*winding>0);
  let previous, lastWidth, lastHeight, count = 0, visible = false;
  return landmarks => {
    const ratio=Math.min(devicePixelRatio,2);
    const width=Math.round(overlay.clientWidth*ratio), height=Math.round(overlay.clientHeight*ratio);
    if (!width || !height) return;
    if (overlay.width!==width) overlay.width=width;
    if (overlay.height!==height) overlay.height=height;
    const resized = width !== lastWidth || height !== lastHeight;
    lastWidth = width; lastHeight = height;
    if (!landmarks || !hasPaint) {
      if (visible || resized) gl.clear(gl.COLOR_BUFFER_BIT);
      visible = false;
      return;
    }
    const geometryChanged = resized || landmarks !== previous;
    if (visible && !geometryChanged && !dirty) return;
    gl.viewport(0,0,width,height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (dirty) {
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,paint);
      dirty=false;
    }
    if (geometryChanged) {
      previous = landmarks;
      const scale=Math.min(width/video.videoWidth,height/video.videoHeight);
      const sx=video.videoWidth*scale/width, sy=video.videoHeight*scale/height;
      const coords=landmarks.map(p=>[1-p.x,p.y]);
      let offset=0;
      for (const [a,b,c] of fronts) {
        if (area(coords[a],coords[b],coords[c])*winding<=0) continue;
        for (const index of [a,b,c]) {
          const [x,y]=coords[index], [u,v]=mesh.points[index];
          vertices[offset++]=(2*x-1)*sx;
          vertices[offset++]=(1-2*y)*sy;
          vertices[offset++]=u/W;
          vertices[offset++]=v/H;
        }
      }
      gl.bufferData(gl.ARRAY_BUFFER,vertices.subarray(0,offset),gl.DYNAMIC_DRAW);
      count = offset/4;
    }
    gl.drawArrays(gl.TRIANGLES,0,count);
    visible = true;
  };
}
function stop(message='鏡頭已關閉。') {
  request++;
  cancelAnimationFrame(frame);
  stream?.getTracks().forEach(track=>track.stop());
  stream=undefined;
  video.srcObject=null;
  renderer?.();
  camera.disabled=false;
  camera.textContent='開啟鏡頭';
  setNotice(message);
}
camera.onclick=async () => {
  if (stream) { stop(); return; }
  const active = ++request;
  camera.disabled=true;
  setNotice('正在載入臉部追蹤模型…');
  let stage='模型';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('請在 HTTPS 網頁或 localhost 開啟鏡頭。');
    mesh=await loadMesh();
    if (active !== request) return;
    renderer??=createRenderer();
    if (!detector) {
      const {FaceLandmarker,FilesetResolver}=await import(cdn+'/vision_bundle.mjs');
      if (active !== request) return;
      const files=await FilesetResolver.forVisionTasks(cdn+'/wasm');
      if (active !== request) return;
      detector=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'},runningMode:'VIDEO',numFaces:1});
    }
    if (active !== request) return;
    stage='相機';
    setNotice('等待相機權限，請允許瀏覽器使用鏡頭。');
    const granted=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:640},height:{ideal:480}}});
    if (active !== request) { granted.getTracks().forEach(track=>track.stop()); return; }
    stream=granted;
    for (const track of stream.getVideoTracks()) track.addEventListener('ended',()=>{
      if (active === request) stop('相機已中斷，請重新開啟。');
    });
    video.srcObject=stream;
    await video.play();
    if (active !== request) return;
    camera.textContent='關閉鏡頭';
    let lastTime=-1, landmarks;
    function tick(time) {
      try {
        if (video.readyState>=2 && video.currentTime!==lastTime) {
          lastTime=video.currentTime;
          landmarks=detector.detectForVideo(video,time).faceLandmarks[0];
          setNotice(landmarks ? '' : '請讓臉部完整出現在鏡頭中。');
        }
        renderer(landmarks);
        frame=requestAnimationFrame(tick);
      } catch { stop('臉部追蹤中斷，請重新開啟鏡頭。'); }
    }
    frame=requestAnimationFrame(tick);
  } catch(error) {
    if (active !== request) return;
    stop(error.name==='NotAllowedError' ? '請在瀏覽器允許相機權限，再重新開啟。' : error.name==='NotFoundError' ? '找不到可用的相機。' : stage === '模型' ? '臉部追蹤載入失敗，請確認網路後重試。' : '無法啟動相機，請確認相機未被其他程式使用後重試。');
  } finally { if (active === request) camera.disabled=false; }
};
overlay.addEventListener('webglcontextlost',event=>{event.preventDefault();renderer=undefined;stop('繪圖暫時中斷，請重新開啟鏡頭。');});
overlay.addEventListener('webglcontextrestored',()=>{dirty=true;});
addEventListener('pagehide',()=>{stop();camera.disabled=false;});


