
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
  let muted = false;
  let confirmed = false;
  let finished = false;
  let buffering;
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
    if (!buffer || muted || context.state !== 'running' || (finished && name !== 'ui_click')) return;
    const source = context.createBufferSource();
    source.buffer = buffer;
    const gain = context.createGain();
    gain.gain.value = gains[name];
    source.connect(gain).connect(output);
    source.onended = () => { source.disconnect(); gain.disconnect(); };
    source.start();
    window.dispatchEvent(new CustomEvent('monolith:titlecue', { detail: name }));
  }
  function updateSoundButton() {
    if (sound) {
      sound.hidden = finished;
      sound.textContent = muted ? 'Unmute sound' : opened ? 'Mute sound' : 'Enable sound';
    }
    if (hint) hint.hidden = finished || muted || opened;
  }
  // The stream may already be buffered while browser autoplay is blocked. Invoke
  // both activation-sensitive methods inside the gesture, before awaiting either.
  function start() {
    if (finished || muted) return;
    const unlocked = context.resume();
    const playing = media.play();
    Promise.all([unlocked, playing]).then(() => {
      if (finished || muted || media.paused) return;
      if (!opened) {
        opened = true;
        if (!confirmed) void cue('ui_open');
      }
      updateSoundButton();
    }).catch(updateSoundButton);
    return unlocked;
  }
  function toggleSound() {
    if (finished) return;
    if (!muted && opened && !media.paused) {
      muted = true;
      output.gain.value = 0;
      media.pause();
      updateSoundButton();
      return;
    }
    muted = false;
    output.gain.value = 0.72;
    start();
  }
  function confirm() {
    if (play.disabled || confirmed || finished) return;
    confirmed = true;
    start()?.then(() => cue('ui_click')).catch(() => {});
  }
  function hover() {
    if (play.disabled || muted || !opened || finished || confirmed || performance.now() - lastHover < 90) return;
    lastHover = performance.now();
    void cue('ui_hover');
  }
  const input = new AbortController();
  const options = { capture: true, signal: input.signal };
  play.addEventListener('pointerenter', hover, options);
  play.addEventListener('focus', hover, options);
  window.addEventListener('click', event => {
    if (sound?.contains(event.target)) toggleSound();
    else if (play.contains(event.target)) confirm();
    else if (!opened) start();
  }, options);
  window.addEventListener('keydown', event => {
    if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.code === 'KeyM') toggleSound();
    else if (event.target !== sound && ['Enter', 'Space'].includes(event.code)) confirm();
  }, options);

  window.monolithTitleAudio = {
    media,
    context,
    output,
    get active() { return !finished; },
    get muted() { return muted; },
    get whenBuffered() {
      return buffering ??= new Promise(resolve => {
        const hasData = () => media.readyState >= media.HAVE_CURRENT_DATA;
        const track = new URL(media.currentSrc || media.src);
        // A gesture-gated player can finish its HTTP range before it decodes
        // audio. Count the successful body from this element's own request.
        const isScoreTransfer = entry => entry.name === track.href && track.origin === location.origin
          && entry.initiatorType === 'audio' && entry.responseEnd > 0 && entry.encodedBodySize > 0
          && (entry.responseStatus === 200 || entry.responseStatus === 206
            || (!entry.responseStatus && entry.contentType?.startsWith('audio/')));
        const alreadyReady = hasData() || performance.getEntriesByName(track.href, 'resource').some(isScoreTransfer);
        if (alreadyReady || media.error || finished) {
          resolve(alreadyReady && !media.error && !finished);
          return;
        }
        const events = new AbortController();
        const observer = window.PerformanceObserver?.supportedEntryTypes?.includes('resource')
          ? new PerformanceObserver(list => {
            if (list.getEntries().some(isScoreTransfer)) settle(true);
          }) : null;
        observer?.observe({ type: 'resource', buffered: true });
        // Give an active download more time. A browser that declines preload
        // can show its sound button soon after the portrait's first frame.
        let deadline = setTimeout(() => {
          if (media.networkState === media.NETWORK_LOADING) deadline = setTimeout(settle, 6500);
          else settle();
        }, 1500);
        function settle(transferComplete) {
          clearTimeout(deadline);
          events.abort();
          observer?.disconnect();
          resolve(!finished && !media.error && (transferComplete === true || hasData()));
        }
        const options = { signal: events.signal };
        for (const event of ['loadeddata', 'canplay', 'error']) media.addEventListener(event, settle, options);
        input.signal.addEventListener('abort', settle, options);
      });
    },
    musicTime(path) {
      return path && new URL(path, document.baseURI).href === media.src && !finished && !media.paused && context.state === 'running' ? media.currentTime : null;
    },
    confirm,
    finish() {
      if (finished) return;
      finished = true;
      input.abort();
      updateSoundButton();
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
  updateSoundButton();
  start();
})();

;// The engine reveals the portrait after drawing its first frame and releases Start
// separately, once the campaign and its opening scene are ready.
(() => {
  const body = document.body;
  if (!body.classList.contains('title-screen') || new URLSearchParams(location.search).get('film') === '1') return;

  const veil = document.getElementById('loading-veil');
  const title = document.getElementById('story-title');
  const play = document.getElementById('title-play');
  const status = document.getElementById('arrival-status');
  const bytes = document.getElementById('movement-progress');
  const retry = document.getElementById('title-retry');
  let startGame;
  let failed = false;
  let portraitReady;

  const hero = document.currentScript?.dataset.hero;
  if (hero && !new URLSearchParams(location.search).get('hero')) {
    const preload = document.createElement('link');
    preload.rel = 'preload';
    preload.as = 'fetch';
    preload.crossOrigin = 'anonymous';
    preload.fetchPriority = 'high';
    preload.href = hero;
    document.head.append(preload);
  }

  function enter() {
    if (play.disabled || !startGame || title.hidden) return;
    const start = startGame;
    startGame = null;
    play.disabled = true;
    start();
  }
  function showPortrait() {
    return portraitReady ??= (async () => {
      await window.monolithTitleAudio?.whenBuffered;
      if (failed) return;
      veil.hidden = true;
      title.hidden = false;
      body.classList.add('title-staging');
    })();
  }
  function fail() {
    failed = true;
    startGame = null;
    play.disabled = true;
    title.hidden = true;
    body.classList.remove('title-staging', 'title-ready');
    veil.hidden = false;
    veil.classList.remove('done');
    veil.querySelector('.load-indicator').hidden = true;
    document.getElementById('load-sub').textContent = 'The game couldn’t load.';
    retry.hidden = false;
  }

  play.onclick = enter;
  retry.onclick = () => location.reload();
  window.addEventListener('keydown', event => {
    if (title.hidden || play.disabled || event.target?.closest('button') || event.repeat) return;
    if (['Space', 'Enter'].includes(event.code)) {
      event.preventDefault();
      enter();
    }
  });
  window.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement && event.target.src.endsWith('/app/bundle.js')) fail();
  }, true);
  window.monolithArrival = {
    showPortrait,
    progress(message) {
      if (!failed && !startGame && status.textContent !== message && message !== 'Ready') status.textContent = message;
    },
    movementProgress(loaded, total) {
      if (failed || startGame) return;
      const amount = loaded < total ? `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB` : '';
      if (bytes.textContent !== amount) bytes.textContent = amount;
    },
    fail,
    async complete(start) {
      if (failed || typeof start !== 'function') return;
      await showPortrait();
      if (failed) return;
      body.classList.replace('title-staging', 'title-ready');
      startGame = start;
      bytes.textContent = '';
      status.textContent = 'Ready to play';
      play.disabled = false;
    },
  };
})();
