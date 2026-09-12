/**
 * sphere.js — render a panorama the way an eye would see it.
 *
 * Panning a flat equirectangular image is wrong in a way people feel before they can
 * name it: the horizon bows, verticals lean, and looking up or down smears. The image is
 * a map of a sphere, so it has to be re-projected for whatever direction you are facing.
 *
 * This is a fullscreen quad and one fragment shader. For each pixel it builds a ray in
 * camera space, rotates it by yaw and pitch, converts to longitude and latitude, and
 * samples the panorama. About a hundred lines, no dependency, and it handles all three
 * projections from the same maths — a flat photo is simply a sphere you can only see a
 * small window of.
 *
 *   const view = new SphereView(canvas);
 *   await view.setImage(img, { projection: 'equirect', hfov: 360, vfov: 180 });
 *   view.render({ yaw: 120, pitch: -5, fovH: 75, fovV: 47 });
 */

const VERT = `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform sampler2D uTex;
uniform vec2  uResolution;
uniform float uYaw, uPitch;        // radians
uniform float uTanH, uTanV;        // tan(fov/2) horizontally and vertically
uniform float uHfov, uVfov;        // coverage of the image, radians
uniform float uWrap;               // 1.0 for a full 360
uniform vec3  uBackground;

void main() {
  // Normalised device coords, -1..1, y up.
  vec2 ndc = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;

  // Ray in camera space. Forward is -z, matching the audio engine's convention where
  // azimuth 0 is straight ahead.
  vec3 d = normalize(vec3(ndc.x * uTanH, ndc.y * uTanV, -1.0));

  // Pitch about x, then yaw about y.
  float cp = cos(uPitch), sp = sin(uPitch);
  d = vec3(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
  float cy = cos(uYaw), sy = sin(uYaw);
  d = vec3(d.x * cy - d.z * sy, d.y, d.x * sy + d.z * cy);

  float lon = atan(d.x, -d.z);     // 0 straight ahead, + to the right
  float lat = asin(clamp(d.y, -1.0, 1.0));

  vec2 uv = vec2(lon / uHfov + 0.5, 0.5 - lat / uVfov);

  if (uWrap > 0.5) {
    uv.x = fract(uv.x);            // longitude wraps all the way round
  } else if (uv.x < 0.0 || uv.x > 1.0) {
    gl_FragColor = vec4(uBackground, 1.0); return;   // outside a partial panorama
  }
  if (uv.y < 0.0 || uv.y > 1.0) { gl_FragColor = vec4(uBackground, 1.0); return; }

  gl_FragColor = texture2D(uTex, uv);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

export class SphereView {
  constructor(canvas) {
    this.canvas = canvas;
    // WebGL2 first: it allows REPEAT on non-power-of-two textures, which a 6144x3072
    // panorama is. On WebGL1 we wrap with fract() in the shader instead.
    this.gl = canvas.getContext('webgl2', { antialias: true, alpha: false })
           || canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!this.gl) throw new Error('WebGL unavailable');
    this.isGL2 = typeof WebGL2RenderingContext !== 'undefined'
              && this.gl instanceof WebGL2RenderingContext;

    const gl = this.gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog));
    }
    gl.useProgram(prog);
    this.prog = prog;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.u = {};
    for (const n of ['uTex', 'uResolution', 'uYaw', 'uPitch', 'uTanH', 'uTanV',
                     'uHfov', 'uVfov', 'uWrap', 'uBackground']) {
      this.u[n] = gl.getUniformLocation(prog, n);
    }
    this.tex = gl.createTexture();
    this.geometry = { hfov: 360, vfov: 180, wrap: true };
  }

  /** Upload a panorama, downscaling if the GPU cannot hold it at full size. */
  setImage(img, geometry) {
    const gl = this.gl;
    const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    let source = img;
    if (img.naturalWidth > max || img.naturalHeight > max) {
      const k = Math.min(max / img.naturalWidth, max / img.naturalHeight);
      const c = document.createElement('canvas');
      c.width = Math.floor(img.naturalWidth * k);
      c.height = Math.floor(img.naturalHeight * k);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      source = c;
      console.info(`[soundscapes] panorama downscaled to ${c.width}x${c.height} ` +
                   `(GPU limit ${max})`);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    const wrapMode = (this.isGL2 && geometry.wrap) ? gl.REPEAT : gl.CLAMP_TO_EDGE;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapMode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.geometry = geometry;
    this.size = { w: source.width || source.naturalWidth,
                  h: source.height || source.naturalHeight };
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    return { w, h };
  }

  render({ yaw, pitch, fovH, fovV }) {
    const gl = this.gl;
    const { w, h } = this.resize();
    const R = Math.PI / 180;
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.u.uTex, 0);
    gl.uniform2f(this.u.uResolution, w, h);
    gl.uniform1f(this.u.uYaw, yaw * R);
    gl.uniform1f(this.u.uPitch, pitch * R);
    gl.uniform1f(this.u.uTanH, Math.tan((fovH * R) / 2));
    gl.uniform1f(this.u.uTanV, Math.tan((fovV * R) / 2));
    gl.uniform1f(this.u.uHfov, this.geometry.hfov * R);
    gl.uniform1f(this.u.uVfov, this.geometry.vfov * R);
    gl.uniform1f(this.u.uWrap, this.geometry.wrap ? 1 : 0);
    gl.uniform3f(this.u.uBackground, 0.043, 0.063, 0.059);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

export default SphereView;
