import createMDX from '@next/mdx'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const nextConfig = {
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
  turbopack: {
    root: __dirname,
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
