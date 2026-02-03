import Link from 'next/link'
import { ThemeToggle } from '@/components/ThemeToggle'
import { AnimateIn } from '@/components/AnimateIn'
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
      <AnimateIn as="header" stagger={0} className={cn(layout.headerEnd, layout.headerSpacing)}>
        <ThemeToggle />
      </AnimateIn>

      <main className={layout.sectionSpacing}>
        <AnimateIn stagger={1}>
          <h1 className={typography({ variant: 'h1' })}>Antonio Cao</h1>
        </AnimateIn>

        <AnimateIn stagger={2}>
          <p
            className={cn(
              typography({ variant: 'body', color: 'default' }),
              'max-w-2xl'
            )}
          >
            Co-founder of{' '}
            <a
              href="https://flair.ai"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                hoverTextColors.default,
                colorTransitions.default,
                'underline underline-offset-2'
              )}
            >
              flair.ai
            </a>
            . Exploring the space where functions become form.
          </p>
        </AnimateIn>

        <section className="pt-8">
          <AnimateIn stagger={3} className="mb-6 flex items-baseline justify-between">
            <h2
              className={typography({
                variant: 'h3',
                color: 'muted',
              })}
            >
              Writings
            </h2>
            <Link
              href="/writing"
              className={cn(
                typography({ variant: 'nav', color: 'muted' }),
                hoverTextColors.default,
                colorTransitions.default
              )}
            >
              <span>View all</span>
              <span className="ml-4">→</span>
            </Link>
          </AnimateIn>

          <AnimateIn stagger={4} className="space-y-6">
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
                          'ml-4 hidden shrink-0 md:inline'
                        )}
                      >
                        {formatDate(post.date)}
                      </span>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </AnimateIn>
        </section>
      </main>
    </div>
  )
}
