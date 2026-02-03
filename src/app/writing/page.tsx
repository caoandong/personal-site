import Link from 'next/link'
import { cn } from '@/lib/utils'
import { typography, layout } from '@/lib/typography'
import { borderColors, groupHoverTextColors, colorTransitions } from '@/lib/colors'
import { posts, groupPostsByYear, formatDate } from '@/lib/posts'

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
                  href={`/writing/${post.slug}`}
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
