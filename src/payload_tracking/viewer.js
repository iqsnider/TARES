const socket = io();
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const status = document.getElementById('status');
let camW = 1920, camH = 1080;

socket.on('connect', () => status.textContent = 'connected');
socket.on('disconnect', () => status.textContent = 'disconnected');

socket.on('markers', (data) => {
  camW = data.w || camW; camH = data.h || camH;
  const dispW = 960, dispH = Math.round(dispW * camH / camW);
  if (cv.width !== dispW || cv.height !== dispH){ cv.width = dispW; cv.height = dispH; }
  const sx = dispW / camW, sy = dispH / camH;

  ctx.clearRect(0, 0, cv.width, cv.height);
  status.textContent = `connected — ${data.markers.length} marker(s)`;

  for (const m of data.markers){
    ctx.beginPath();
    m.corners.forEach(([x,y], k) => {
      const px = x*sx, py = y*sy;
      k === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    });
    ctx.closePath();
    ctx.lineWidth = 2; ctx.strokeStyle = '#3cf'; ctx.stroke();

    const [x0,y0] = m.corners[0];              // first corner = orientation
    ctx.fillStyle = '#f33';
    ctx.beginPath(); ctx.arc(x0*sx, y0*sy, 4, 0, 7); ctx.fill();

    const [cx,cy] = m.center;
    ctx.fillStyle = '#ff0'; ctx.font = '14px monospace';
    ctx.fillText('id ' + m.id, cx*sx + 6, cy*sy);
  }
});
