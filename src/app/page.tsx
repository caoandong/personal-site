import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'
import { cn } from '@/lib/utils'
import { typography, layout } from '@/lib/typography'
import {
  hoverTextColors,
  borderColors,
  groupHoverTextColors,
  colorTransitions,
} from '@/lib/colors'
import { getRecentPosts, groupPostsByYear, formatDate } from '@/lib/posts'

const RECENT_POSTS_LIMIT = 5

export default function Home() {
  const recentPosts = getRecentPosts(RECENT_POSTS_LIMIT)
  const groupedPosts = groupPostsByYear(recentPosts)
  const sortedYears = Object.keys(groupedPosts).sort(
    (a, b) => Number(b) - Number(a)
  )

  return (
    <div className={layout.container}>
      <header className={cn(layout.headerEnd, layout.headerSpacing)}>
        <ThemeToggle />
      </header>

      <main className={layout.sectionSpacing}>
        <h1 className={typography({ variant: 'h1' })}>Personal Blog</h1>

        <p
          className={cn(
            typography({ variant: 'body', color: 'default' }),
            'max-w-2xl'
          )}
        >
          A minimalistic blog exploring mathematics, design engineering, and
          interactive visualizations.
        </p>

        <section className="pt-8">
          <div className="mb-6 flex items-baseline justify-between">
            <h2
              className={typography({
                variant: 'h2',
                color: 'muted',
              })}
            >
              Writings
            </h2>
            <Link
              href="/blog"
              className={cn(
                typography({ variant: 'nav', color: 'muted' }),
                hoverTextColors.default,
                colorTransitions.default
              )}
            >
              <span>View all</span>
              <span className="ml-4">→</span>
            </Link>
          </div>

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
        </section>
      </main>
    </div>
  )
}
