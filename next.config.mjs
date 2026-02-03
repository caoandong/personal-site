import createMDX from '@next/mdx'

/** @type {import('next').NextConfig} */
const nextConfig = {
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
  turbopack: {
    root: '/Users/antonio/flair/software/personal-site',
  },
}

// With Turbopack, plugins must be passed as strings (not imported functions)
// This allows the plugins to be serialized and passed to Rust
const withMDX = createMDX({
  options: {
    remarkPlugins: [
      'remark-math',
      'remark-gfm',
    ],
    rehypePlugins: [
      'rehype-katex',
    ],
  },
})

export default withMDX(nextConfig)
