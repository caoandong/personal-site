import createMDX from '@next/mdx'
import path from 'path'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const { release } = JSON.parse(readFileSync(new URL('./game/release.json', import.meta.url), 'utf8'))
const gamePath = '/return-of-the-king'
const gamePolicy = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "worker-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  "connect-src 'self' blob:",
  "frame-ancestors 'none'",
].join('; ')

/** @type {import('next').NextConfig} */
const nextConfig = {
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [{ source: gamePath, destination: `${gamePath}/releases/${release}/index.html` }]
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob:",
              "font-src 'self'",
              "connect-src 'self'",
              "frame-ancestors 'none'",
            ].join('; '),
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
      {
        source: `${gamePath}/:path*`,
        headers: [{ key: 'Content-Security-Policy', value: gamePolicy }],
      },
      {
        source: gamePath,
        headers: [{ key: 'Cache-Control', value: 'public, max-age=0, must-revalidate' }],
      },
      {
        source: `${gamePath}/releases/:release/:asset*`,
        headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
      },
    ]
  },
}

const withMDX = createMDX({
  options: {
    remarkPlugins: [
      'remark-math',
      'remark-gfm',
    ],
    rehypePlugins: [
      'rehype-katex',
      // rehype-pretty-code with serializable options for Turbopack compatibility
      ['rehype-pretty-code', {
        theme: {
          dark: 'github-dark-dimmed',
          light: 'github-light',
        },
        keepBackground: true,
        defaultLang: 'plaintext',
      }],
    ],
  },
})

export default withMDX(nextConfig)
