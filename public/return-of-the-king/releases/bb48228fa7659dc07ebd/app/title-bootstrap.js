
    (function () {
      var q = new URLSearchParams(window.location.search);
      var wantTitle = (q.get('film') !== '1' || q.get('title') === '1' || q.get('title') === 'true') && q.get('mode') !== 'play' && q.get('play') !== '1';
      if (wantTitle) document.body.classList.add('title-screen');
    })();
  
;// Runs at the title markup, before the renderer and its assets. The media element
// can start from an HTTP range; decoding the entire score would delay first sound.
(() => {
  const config = document.currentScript.dataset;
  const query = new URLSearchParams(location.search);
  if (!document.body.classList.contains('title-screen') || query.get('film') === '1' || query.get('score') === '0') return;
  const media = document.getElementById('arrival-score') || new Audio();
  media.preload = 'auto';
  media.loop = true;
  if (!media.getAttribute('src')) media.src = config.track;
  const context = new AudioContext();
  const output = context.createGain();
  const trackGain = context.createGain();
  output.gain.value = 0.72;
  trackGain.gain.value = 0.9;
  context.createMediaElementSource(media).connect(trackGain).connect(output);
  output.connect(context.destination);
  const loopStart = Number(config.loopStart);
  const loopEnd = Number(config.loopEnd);
  media.addEventListener('timeupdate', () => {
    if (loopEnd > loopStart && media.currentTime >= loopEnd) {
      media.currentTime = loopStart + (media.currentTime - loopEnd) % (loopEnd - loopStart);
    }
  });

  const play = document.getElementById('title-play');
  const sound = document.getElementById('title-sound');
  const hint = document.getElementById('title-audio-hint');
  const cues = new Map();
  let opened = false;
  let confirmed = false;
  let finished = false;
  let lastHover = -Infinity;
  const gains = { ui_open: 0.5, ui_click: 0.52, ui_hover: 0.28 };
  function bufferFor(name) {
    if (!cues.has(name)) {
      const loaded = fetch('./assets/sfx/ui/' + name + '_01.mp3')
        .then(response => {
          if (!response.ok) throw new Error(response.statusText);
          return response.arrayBuffer();
        })
        .then(bytes => context.decodeAudioData(bytes))
        .catch(() => { cues.delete(name); return null; });
      cues.set(name, loaded);
    }
    return cues.get(name);
  }
  for (const name of Object.keys(gains)) bufferFor(name);
  async function cue(name) {
    const buffer = await bufferFor(name);
    if (!buffer || context.state !== 'running' || (finished && name !== 'ui_click')) return;
    const source = context.createBufferSource();
    source.buffer = buffer;
    const gain = context.createGain();
    gain.gain.value = gains[name];
    source.connect(gain).connect(output);
    source.onended = () => { source.disconnect(); gain.disconnect(); };
    source.start();
    window.dispatchEvent(new CustomEvent('monolith:titlecue', { detail: name }));
  }
  function showSoundButton(visible) {
    if (sound) sound.hidden = !visible;
    if (hint) hint.hidden = !visible;
  }
  // Both permission-sensitive calls must run directly inside the input event,
  // even if an earlier automatic attempt still has an unresolved resume promise.
  function start() {
    if (finished) return;
    const unlocked = context.resume();
    const playing = media.play();
    Promise.all([unlocked, playing]).then(() => {
      if (finished || media.paused) return;
      showSoundButton(false);
      if (!opened) {
        opened = true;
        if (!confirmed) void cue('ui_open');
      }
    }).catch(() => { if (!finished && !opened) showSoundButton(true); });
    return unlocked;
  }
  function confirm() {
    if (confirmed || finished) return;
    confirmed = true;
    start()?.then(() => cue('ui_click')).catch(() => {});
  }
  function hover() {
    if (!opened || finished || confirmed || performance.now() - lastHover < 90) return;
    lastHover = performance.now();
    void cue('ui_hover');
  }
  const input = new AbortController();
  const options = { capture: true, signal: input.signal };
  play.addEventListener('pointerenter', hover, options);
  play.addEventListener('focus', hover, options);
  window.addEventListener('click', event => {
    if (play.contains(event.target)) confirm(); else start();
  }, options);
  window.addEventListener('keydown', event => {
    if (event.repeat) return;
    if (event.target !== sound && ['Enter', 'Space'].includes(event.code)) confirm();
    else start();
  }, options);

  window.monolithTitleAudio = {
    media,
    context,
    output,
    get active() { return !finished; },
    musicTime(path) {
      return path && new URL(path, document.baseURI).href === media.src && !finished && !media.paused && context.state === 'running' ? media.currentTime : null;
    },
    confirm,
    finish() {
      if (finished) return;
      finished = true;
      input.abort();
      showSoundButton(false);
      const now = context.currentTime;
      trackGain.gain.setValueAtTime(trackGain.gain.value, now);
      trackGain.gain.linearRampToValueAtTime(0, now + 0.28);
      setTimeout(() => {
        media.pause();
        media.removeAttribute('src');
        media.load();
        trackGain.disconnect();
      }, 320);
    },
  };
  showSoundButton(true);
  start();
})();

;// The title and courtyard can run while the WebGL engine is still downloading.
(() => {
  const body = document.body;
  const query = new URLSearchParams(location.search);
  if (!body.classList.contains('title-screen') || query.get('film') === '1') return;
  const title = document.getElementById('story-title');
  const play = document.getElementById('title-play');
  const status = document.getElementById('arrival-status');
  const clip = document.getElementById('arrival-film');
  let startGame;
  let queued = false;
  let failed = false;
  let stage = status.textContent;
  let movement = '';
  function describe() {
    if (!failed) status.textContent = movement || stage;
  }

  function enter() {
    if (failed) return location.reload();
    if (startGame) {
      const start = startGame;
      startGame = null;
      start();
      return;
    }
    if (queued) return;
    queued = true;
    body.classList.add('arrival-queued');
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches && !navigator.connection?.saveData) {
      clip.addEventListener('playing', () => clip.classList.add('playing'), { once: true });
      clip.src = './assets/arrival/approach.mp4';
      clip.play().catch(() => {}); // The courtyard still is the reduced-motion and low-data fallback.
    }
  }
  function retry() {
    failed = true;
    body.classList.remove('arrival-queued');
    body.classList.add('arrival-loading');
    title.hidden = false;
    play.textContent = 'Try again';
    status.textContent = 'The courtyard couldn’t load. Check your connection and try again.';
  }

  body.classList.add('arrival-loading');
  title.hidden = false;
  play.onclick = enter;
  window.addEventListener('keydown', event => {
    if (title.hidden || event.target?.id === 'title-sound' || event.repeat) return;
    if (['Space', 'Enter'].includes(event.code)) {
      event.preventDefault();
      enter();
    }
  });
  window.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement && event.target.src.endsWith('/app/bundle.js')) retry();
  }, true);
  window.monolithArrival = {
    progress(message) { stage = message; describe(); },
    movementProgress(loaded, total) {
      const next = loaded < total ? `Downloading movement · ${(loaded / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB` : '';
      if (next !== movement) { movement = next; describe(); }
    },
    fail: retry,
    complete(start) {
      startGame = start;
      body.classList.remove('arrival-loading', 'arrival-queued');
      clip.pause();
      if (clip.hasAttribute('src')) { clip.removeAttribute('src'); clip.load(); }
      if (queued) enter();
    },
  };
})();
