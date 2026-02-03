import Link from 'next/link'
import { cn } from '@/lib/utils'
import { typography, layout } from '@/lib/typography'

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
    <div className={cn(layout.sectionSpacing, 'not-prose')}>
      <h1 className={typography({ variant: 'h1' })}>
        Blog
      </h1>

      <div className="space-y-10 md:space-y-12">
        {posts.map((post) => (
          <article key={post.slug} className="group border-b border-stroke pb-10 md:pb-12 last:border-b-0">
            <Link href={`/blog/${post.slug}`} className="block">
              <time className={typography({ variant: 'small', color: 'muted' })}>
                {new Date(post.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })}
              </time>
              <h2 className={cn(typography({ variant: 'h2' }), 'mt-2 group-hover:text-grey-1 transition-colors duration-200')}>
                {post.title}
              </h2>
              <p className={cn(typography({ variant: 'body', color: 'muted' }), 'mt-3 max-w-2xl')}>{post.description}</p>
            </Link>
          </article>
        ))}
      </div>
    </div>
  )
}
