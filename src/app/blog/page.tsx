import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'

const posts = [
  {
    slug: 'math-foundations',
    title: 'Mathematical Foundations',
    description: 'Demonstrating full LaTeX rendering with inline and block mathematics, including Maxwell equations and Fourier transforms.',
    date: '2025-02-02',
  },
]

export default function BlogIndex() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <header className="mb-12 flex items-center justify-between">
        <Link
          href="/"
          className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          &larr; Home
        </Link>
        <ThemeToggle />
      </header>

      <h1 className="text-4xl font-bold tracking-tight mb-8">Blog</h1>

      <div className="space-y-8">
        {posts.map((post) => (
          <article key={post.slug} className="group">
            <Link href={`/blog/${post.slug}`} className="block">
              <time className="text-sm text-muted-foreground">
                {new Date(post.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })}
              </time>
              <h2 className="text-xl font-semibold mt-1 group-hover:underline">
                {post.title}
              </h2>
              <p className="text-muted-foreground mt-2">{post.description}</p>
            </Link>
          </article>
        ))}
      </div>
    </div>
  )
}
