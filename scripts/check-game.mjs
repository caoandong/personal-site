import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
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
const [timelines, campaign] = await Promise.all(['cutscenes', 'campaigns'].map(async name =>
  (await fetch(origin + prefix + 'assets/' + name + '.json')).json()))
const titleTrack = campaign.levels[0].titleMusic.path
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
  const obsoleteMedia = []
  page.on('request', request => {
    const path = new URL(request.url()).pathname
    if (/\/assets\/arrival\/(portrait\.webp|approach\.(webp|mp4))$/.test(path)) obsoleteMedia.push(path)
  })
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
    // Count actual skinned-geometry draws on the game canvas while the motion pack is blocked.
    window.portraitDraws = 0
    window.portraitFrames = 0
    let lastSkinnedFrame
    const skinned = new WeakMap()
    for (const name of ['drawElements', 'drawArrays', 'drawElementsInstanced', 'drawArraysInstanced']) {
      const draw = WebGL2RenderingContext.prototype[name]
      WebGL2RenderingContext.prototype[name] = function (...args) {
        const result = draw.apply(this, args)
        if (this.canvas.id !== 'viewport' || window.portraitFrames >= 3) return result
        const program = this.getParameter(this.CURRENT_PROGRAM)
        if (program && !skinned.has(program)) {
          skinned.set(program, this.getAttribLocation(program, 'skinWeight') >= 0 &&
            this.getAttribLocation(program, 'skinIndex') >= 0)
        }
        if (program && skinned.get(program)) {
          if (window.portraitDraws === 0) {
            // Inspect on the next task, before the mobile media fallback timer.
            // A busy software GPU can delay a round trip from the test driver.
            setTimeout(() => {
              window.firstFrameCovered = document.querySelector('#story-title').hidden &&
                document.querySelector('#loading-veil').getBoundingClientRect().width > 0
            }, 0)
          }
          window.portraitDraws++
          const frame = document.timeline.currentTime
          if (frame !== lastSkinnedFrame) {
            lastSkinnedFrame = frame
            window.portraitFrames++
          }
        }
        return result
      }
    }
    window.playedTitleCues = []
    window.addEventListener('monolith:titlecue', event => window.playedTitleCues.push(event.detail))
    window.gameCspViolations = []
    document.addEventListener('securitypolicyviolation', event => {
      window.gameCspViolations.push([event.effectiveDirective, event.blockedURI])
    })
  })
  // Hold the renderer, score and movement data independently, even on a fast connection.
  let releaseBundle
  let releaseScore
  let releaseMotion
  const bundle = new Promise(resolve => { releaseBundle = resolve })
  const score = new Promise(resolve => { releaseScore = resolve })
  const motion = new Promise(resolve => { releaseMotion = resolve })
  await page.route('**/app/bundle.js', async route => { await bundle; await route.continue() })
  await page.route(origin + prefix + titleTrack, async route => { await score; await route.continue() })
  await page.route('**/pack.bin.zst', async route => { await motion; await route.continue() })
  await page.goto(base.href, { waitUntil: 'commit', timeout: 90000 })
  const session = await page.context().newCDPSession(page)
  const inspect = async expression => {
    const { result, exceptionDetails } = await session.send('Runtime.evaluate', {
      expression, userGesture: false, awaitPromise: true, returnByValue: true,
    })
    assert.equal(exceptionDetails, undefined)
    return result.value
  }
  const waitForInspection = async (expression, timeout = 30000) => {
    const deadline = Date.now() + timeout
    while (Date.now() < deadline) {
      if (await inspect(expression)) return
      await delay(100)
    }
    const status = await inspect("document.querySelector('#load-sub')?.textContent")
    throw new Error(`Timed out waiting for ${expression}. Loader: ${status}. Errors: ${errors.join(', ')}`)
  }
  const capture = async name => {
    const { data } = await session.send('Page.captureScreenshot', { format: 'png' })
    if (process.env.GAME_SCREENSHOTS_DIR) {
      await writeFile(join(process.env.GAME_SCREENSHOTS_DIR, name + '.png'), Buffer.from(data, 'base64'))
    }
    return data
  }
  const start = page.locator('#title-play')
  const sound = page.locator('#title-sound')
  // Avoid browser automation's synthetic activation before checking the autoplay policy.
  await waitForInspection("document.querySelector('#load-sub')?.getBoundingClientRect().width > 0 && document.querySelector('#load-sub').textContent.trim()")
  assert.equal(await inspect("document.querySelector('#story-title').getBoundingClientRect().width"), 0)
  // Measure the presented frame: nearly the entire loader must be dark.
  const data = await capture('01-dark-loader')
  const darkFraction = await inspect(`(async () => {
    const pixels = Uint8Array.from(atob('${data}'), c => c.charCodeAt(0))
    const image = await createImageBitmap(new Blob([pixels], {type: 'image/png'}))
    const ctx = new OffscreenCanvas(80, 45).getContext('2d')
    ctx.drawImage(image, 0, 0, 80, 45)
    image.close()
    const rgba = ctx.getImageData(0, 0, 80, 45).data
    let dark = 0
    for (let i = 0; i < rgba.length; i += 4) {
      if (Math.max(rgba[i], rgba[i+1], rgba[i+2]) < 60) dark++
    }
    return dark / (rgba.length / 4)
  })()`)
  assert.ok(darkFraction > 0.9, `Initial loader was only ${Math.round(darkFraction * 100)}% dark`)
  assert.deepEqual(obsoleteMedia, [])
  releaseBundle()

  await waitForInspection("typeof window.firstFrameCovered === 'boolean'", 150000)
  assert.equal(await inspect('window.firstFrameCovered'), true,
    'First hero frame bypassed the unbuffered title score')
  releaseScore()
  await waitForInspection("document.querySelector('#title-play')?.getBoundingClientRect().width > 0", 150000)
  assert.equal(await inspect("document.querySelector('#title-play').disabled"), true)
  // Fresh Chrome may download the audio range while deferring decoding until a tap.
  // Require buffered samples or a completed score transfer before touching the controls.
  await waitForInspection(`(() => {
    const media = window.monolithTitleAudio.media
    return media.readyState >= media.HAVE_CURRENT_DATA ||
      performance.getEntriesByName(media.currentSrc).some(entry => entry.encodedBodySize > 0)
  })()`, 45000)
  await waitForInspection('window.portraitFrames === 3')
  const portraitDraws = await inspect('window.portraitDraws')
  assert.equal(await inspect('!!window.__VOYAGER__'), false)
  await waitForInspection("document.querySelector('#title-sound')?.getBoundingClientRect().width > 0")
  assert.equal(await inspect("document.querySelector('#title-sound').textContent.trim()"), 'Enable sound')
  assert.equal(await inspect('window.monolithTitleAudio.context.state'), 'suspended')
  await sound.click()
  await page.waitForFunction(() => {
    const audio = window.monolithTitleAudio
    if (audio?.media.currentTime < 1 || !audio) return false
    if (!window.titleMeter) {
      window.titleMeter = audio.context.createAnalyser()
      audio.output.connect(window.titleMeter)
    }
    const samples = new Float32Array(window.titleMeter.fftSize)
    window.titleMeter.getFloatTimeDomainData(samples)
    return samples.some(value => Math.abs(value) > 0.01)
  }, null, { timeout: 45000 })
  assert.equal(await sound.isVisible(), true)
  assert.equal((await sound.textContent()).trim(), 'Mute sound')
  if (process.env.GAME_SCREENSHOTS_DIR) await capture('02-title-loading')

  // Use physical input on the disabled button. Nothing should start now or be queued for later.
  const box = await start.boundingBox()
  assert.ok(box)
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)
  await page.locator('#story-title-name').click()
  await page.keyboard.press('Enter')
  await page.keyboard.press('Space')
  await inspect('window.monolithTitleAudio.confirm()')
  assert.equal(await start.isDisabled(), true)
  assert.equal(await inspect("window.playedTitleCues.includes('ui_click')"), false)
  await page.mouse.move(1, 1)

  releaseMotion()
  await page.waitForFunction(() => {
    const game = window.__VOYAGER__?.getState().game
    return game?.titlePortrait && !document.getElementById('title-play').disabled
  }, null, { timeout: 180000 })
  assert.equal(await start.isVisible(), true)
  assert.equal(await inspect("!!window.__VOYAGER__.getState().game.scene"), false)
  if (process.env.GAME_SCREENSHOTS_DIR) await capture('03-title-ready')
  await start.hover()
  await start.click()
  await page.waitForFunction(() => window.__VOYAGER__?.getState().game?.scene === 'CS-01', null, { timeout: 180000 })
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
  assert.equal(titleCues.filter(cue => cue === 'ui_click').length, 1)
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
  await page.waitForFunction(() => window.__VOYAGER__.app.lightingDetail === 'full', null, { timeout: 120000 })
  assert.deepEqual(errors, [])
  assert.deepEqual(obsoleteMedia, [])
  console.log(JSON.stringify({ url: res.url, release, darkFraction, portraitDraws, titleCues, opening, combat, musicFiles: openingMusicFiles, laterMusic: later.id, workers: workers.length }))
} finally {
  await browser.close()
}
