import { ThemeToggle } from '@/components/ThemeToggle'
import { BackButton } from '@/components/BackButton'
import { AnimateIn } from '@/components/AnimateIn'
import { cn } from '@/lib/utils'
import { layout } from '@/lib/typography'

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={layout.container}>
      <AnimateIn as="header" stagger={0} className={cn(layout.header, layout.headerSpacing)}>
        <BackButton fallback="/" />
        <ThemeToggle />
      </AnimateIn>
      {children}
    </div>
  )
}
