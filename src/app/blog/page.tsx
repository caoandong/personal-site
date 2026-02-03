import Link from 'next/link'
import { cn } from '@/lib/utils'
import { typography, layout } from '@/lib/typography'
import { borderColors, groupHoverTextColors, colorTransitions } from '@/lib/colors'

interface Post {
  slug: string
  title: string
  date: string
}

const posts: Post[] = [
  {
    slug: 'math-foundations',
    title: 'Mathematical Foundations',
    date: '2025-02-02',
  },
]

function groupPostsByYear(posts: Post[]): Record<string, Post[]> {
  return posts.reduce(
    (acc, post) => {
      const year = new Date(post.date).getFullYear().toString()
      if (!acc[year]) acc[year] = []
      acc[year].push(post)
      return acc
    },
    {} as Record<string, Post[]>
  )
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${month}/${day}`
}

export default function BlogIndex() {
  const groupedPosts = groupPostsByYear(posts)
  const sortedYears = Object.keys(groupedPosts).sort(
    (a, b) => Number(b) - Number(a)
  )

  return (
    <main className={layout.sectionSpacing}>
      <h1 className={typography({ variant: 'h1' })}>Writing</h1>

      <div className="space-y-6">
        {sortedYears.map((year) => (
          <div
            key={year}
            className="grid grid-cols-[4rem_1fr] gap-x-4 md:grid-cols-[5rem_1fr] md:gap-x-6"
          >
            <span
              className={cn(
                typography({ variant: 'small', color: 'muted' }),
                'pt-2'
              )}
            >
              {year}
            </span>
            <div>
              {groupedPosts[year].map((post, index) => (
                <Link
                  key={post.slug}
                  href={`/blog/${post.slug}`}
                  className={cn(
                    'group flex items-baseline justify-between py-2',
                    index !== groupedPosts[year].length - 1 &&
                      `border-b ${borderColors.stroke}`
                  )}
                >
                  <span
                    className={cn(
                      typography({ variant: 'body' }),
                      groupHoverTextColors.subtle,
                      colorTransitions.default
                    )}
                  >
                    {post.title}
                  </span>
                  <span
                    className={cn(
                      typography({ variant: 'small', color: 'muted' }),
                      'ml-4 shrink-0'
                    )}
                  >
                    {formatDate(post.date)}
                  </span>
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
