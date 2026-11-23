import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { chromium } from 'playwright-core'

const { release } = JSON.parse(await readFile(new URL('../game/release.json', import.meta.url)))
const base = new URL(process.argv[2] || 'http://127.0.0.1:3000/return-of-the-king')
const path = '/return-of-the-king'
const prefix = path + '/releases/' + release + '/'
const res = await fetch(base)
assert.equal(res.status, 200)
assert.equal(new URL(res.url).pathname, path)
assert.ok(res.headers.get('content-type')?.includes('text/html'))
assert.match(res.headers.get('cache-control') || '', /max-age=0/)
const origin = new URL(res.url).origin
const timelines = await (await fetch(origin + prefix + 'assets/cutscenes.json')).json()
assert.equal((await fetch(origin + prefix + 'missing-for-smoke.bin')).status, 404)
const js = await fetch(origin + prefix + 'app/bundle.js')
assert.equal(js.status, 200)
assert.match(js.headers.get('cache-control') || '', /immutable/)
const expected = await readFile(new URL('../public' + prefix + 'app/bundle.js', import.meta.url))
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
assert.equal(hash(Buffer.from(await js.arrayBuffer())), hash(expected))

const browser = await chromium.launch({
  channel: 'chrome',
  args: [
    '--enable-unsafe-swiftshader', '--disable-dev-shm-usage',
    '--autoplay-policy=document-user-activation-required',
    '--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies',
  ],
})
try {
  const page = await browser.newPage({ viewport: { width: 800, height: 450 } })
  const errors = []
  const workers = []
  const music = new Set()
  page.on('pageerror', error => errors.push(error.message))
  page.on('worker', worker => workers.push(new URL(worker.url()).pathname))
  page.on('response', response => {
    const url = new URL(response.url())
    if (!['http:', 'https:'].includes(url.protocol)) return
    if (response.status() >= 400) errors.push(response.status() + ' ' + url.pathname)
    if (url.pathname === path) return
    if (url.origin !== origin || !url.pathname.startsWith(prefix)) errors.push('Unexpected asset ' + url.href)
    if (url.pathname.includes('/assets/music/')) music.add(url.pathname)
  })
  await page.addInitScript(() => {
    // Read the signal sent to the speakers without changing the game mixer.
    const connect = AudioNode.prototype.connect
    AudioNode.prototype.connect = function (destination, ...ports) {
      if (this.context instanceof AudioContext && destination === this.context.destination) {
        window.gameOutput = this.context.createAnalyser()
        connect.call(this, window.gameOutput)
      }
      return connect.call(this, destination, ...ports)
    }
    window.gameCspViolations = []
    document.addEventListener('securitypolicyviolation', event => {
      window.gameCspViolations.push([event.effectiveDirective, event.blockedURI])
    })
  })
  await page.goto(base.href, { waitUntil: 'domcontentloaded', timeout: 90000 })
  // Poll with no synthetic user gesture until the real first interaction below.
  const session = await page.context().newCDPSession(page)
  const started = Date.now()
  while (true) {
    const { result } = await session.send('Runtime.evaluate', {
      userGesture: false, returnByValue: true,
      expression: `!document.querySelector('#story-title')?.hidden && !!window.__MONOLITH_GAME__`,
    })
    if (result.value) break
    assert.ok(Date.now() - started < 150000, 'Title did not open')
    await new Promise(resolve => setTimeout(resolve, 250))
  }
  const { result } = await session.send('Runtime.evaluate', {
    userGesture: false,
    returnByValue: true,
    expression: `(() => {
      const audio = window.__MONOLITH_GAME__.audio
      const cue = audio.cue
      window.playedTitleCues = []
      audio.cue = (...args) => {
        const played = cue(...args)
        if (played) window.playedTitleCues.push(args[0])
        return played
      }
      return [document.title, audio.state().state, !document.getElementById('title-audio-hint').hidden]
    })()`,
  })
  assert.deepEqual(result.value, ['Return of the King', 'suspended', true])
  await page.keyboard.press('m')
  await page.waitForFunction(() => {
    const track = window.__VOYAGER__.app.campaign.titleMusic.path
    if (window.__MONOLITH_GAME__.audio.musicTime(track) < 2) return false
    const samples = new Float32Array(window.gameOutput.fftSize)
    window.gameOutput.getFloatTimeDomainData(samples)
    return samples.some(value => Math.abs(value) > 0.01)
  }, null, { timeout: 45000 })
  assert.equal(await page.locator('#title-audio-hint').isHidden(), true)
  assert.equal(await page.locator('#story-title').isVisible(), true)
  await page.locator('#title-play').hover()
  await page.locator('#title-play').click({ timeout: 150000 })
  await page.waitForFunction(() => window.__VOYAGER__?.getState().game?.ready, null, { timeout: 180000 })
  const opening = await page.evaluate(() => {
    const app = window.__VOYAGER__.app
    return {
      scene: window.__VOYAGER__.getState().game.scene,
      layers: app.giArrayTexture?.image.depth,
      atlases: app.loadedAtlases.length,
      glError: app.renderer.getContext().getError(),
    }
  })
  const titleCues = await page.evaluate(() => window.playedTitleCues.filter(name => name.startsWith('ui_')))
  for (const cue of ['ui_open', 'ui_hover', 'ui_click']) assert.ok(titleCues.includes(cue), 'Missing title cue ' + cue)
  assert.equal(opening.scene, 'CS-01')
  assert.ok(opening.layers > 0)
  assert.equal(opening.layers, opening.atlases)
  assert.equal(opening.glError, 0)
  await page.keyboard.down('Space')
  await page.waitForFunction(() => {
    const game = window.__VOYAGER__.getState().game
    return !game.scene && game.tick > 0 && game.health > 0 && game.enemies > 0
  }, null, { timeout: 120000 })
  await page.keyboard.up('Space')
  const openingTrack = timelines[0].afterMusic?.path || timelines[0].music
  await page.waitForFunction(track => window.__MONOLITH_GAME__.audio.musicTime(track) > 0, openingTrack)
  const combat = await page.evaluate(track => {
    const game = window.__VOYAGER__.getState().game
    const api = window.__MONOLITH_GAME__
    const audio = api.audio.state()
    const resources = performance.getEntriesByType('resource')
    const bytes = rows => rows.reduce((n, row) => n + row.encodedBodySize, 0)
    return {
      enemies: game.enemies, health: game.health, sfx: audio.sfxFiles,
      musicSeconds: api.audio.musicTime(track), bloodTextures: api.blood.stats().texturesLoaded,
      totalBytes: bytes(resources), musicBytes: bytes(resources.filter(r => r.name.includes('/assets/music/'))),
      csp: window.gameCspViolations,
    }
  }, openingTrack)
  assert.equal(combat.bloodTextures, 9)
  assert.equal(combat.sfx, 140)
  assert.ok(music.size > 0 && music.size <= 5, 'Opening downloaded ' + music.size + ' music files')
  assert.ok(combat.musicBytes < 10_000_000, 'Opening exceeded its soundtrack transfer budget')
  assert.ok(workers.filter(url => url === prefix + 'app/unpack-worker.js').length >= 2)
  assert.deepEqual(combat.csp, [])
  const openingMusicFiles = music.size
  const later = timelines.find(scene => scene.music && !music.has(prefix + scene.music))
  assert.ok(later, 'Expected deferred scene music')
  assert.ok(await page.evaluate(id => window.__MONOLITH_GAME__.startScene(id), later.id))
  await page.waitForFunction(track => window.__MONOLITH_GAME__.audio.musicTime(track) > 0, later.music, { timeout: 90000 })
  assert.deepEqual(await page.evaluate(() => window.gameCspViolations), [])
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ url: res.url, release, titleCues, opening, combat, musicFiles: openingMusicFiles, laterMusic: later.id, workers: workers.length }))
} finally {
  await browser.close()
}
